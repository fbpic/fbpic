# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the class LaserAntenna, which can be used to continuously
emit a laser during a simulation.
"""
import numpy as np
from scipy.constants import e, c, epsilon_0, physical_constants
r_e = physical_constants['classical electron radius'][0]
from fbpic.particles.utilities.utility_methods import weights
from fbpic.particles.deposition.numba_methods import deposit_field_numba

# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    import cupy
    from fbpic.utils.cuda import cuda, cuda_tpb_bpg_1d, compile_cupy

class LaserAntenna( object ):
    """
    Class that implements the emission of a laser by an antenna

    The antenna produces a current on the grid (in a thin slice along z), which
    matches the electric field to be emitted, according to the formula
    j(t) = 2 \epsilon_0 c E_emitted(z0, t)
    (The above formula is valid for an antenna at a fixed z0. When running in
    the boosted frame, the antenna is moving and the proportionality coefficient
    is modified accordingly.)

    This current is produced on the grid by using virtual macroparticles (this
    ensures that the charge conservation is properly satisfied), whose
    positions are stored in the LaserAntenna object. The motion of the virtual
    particles is prescribed, and their charge and currents are deposited
    at each timestep during the PIC loop.

    Note that the antenna is made of a set of matching positive and negative
    macroparticles, which are exactly superimposed when there is no emission,
    and which have opposite excursions when there is some emission.
    (Therefore, only the excursion of the positive particles is stored; the
    excursion of the negative is infered e.g. before depositing their current)

    Since the number of macroparticles is small, both updating their motion
    and depositing their charge/current is always done on the CPU.
    For GPU performance, the charge/current are deposited in a small-size array
    (corresponding to a thin slice in z) which is then transfered to the GPU
    and added into the full-size array of charge/current.
    Note that the antenna always uses linear shape factors (even when the
    rest of the simulation uses cubic shape factors.)
    """
    def __init__( self, laser_profile, z0_antenna, v_antenna,
                    dr_grid, Nr_grid, Nm, boost, npr=2, epsilon=0.01 ):
        """
        Initialize a LaserAntenna object (see class docstring for more info)

        Parameters
        ----------
        profile: a valid laser profile object
            Gives the value of the laser field in space and time

        z0_antenna: float (m)
            Initial position of the antenna *in the lab frame*

        v_antenna: float (m/s)
            Only used for the ``antenna`` method: velocity of the antenna
            (in the lab frame)

        dr_grid: float (m)
           Resolution of the grid which contains the fields

        Nr_grid: int
           Number of gridpoints radially

        Nm: int
           Number of azimuthal modes

        npr: int
           Number of virtual particles along the r axis, per cell

        nptheta: int
           Number of virtual particles in the theta direction
           (Particles are distributed along a star-pattern with
           nptheta arms in the transverse plane)

        epsilon: float
           Ratio between the maximum transverse excursion of any virtual
           particle of the laser antenna, and the transverse size of a cell
           (i.e. a virtual particle will not move by more than epsilon*dr)

        boost: a BoostConverter object or None
           Contains the information about the boost to be applied
        """
        # Register the properties of the laser injection
        self.laser_profile = laser_profile
        self.boost = boost

        # For now, boost and non-zero velocity are incompatible
        if (v_antenna != 0) and (boost is not None):
            if boost.gamma0 != 1.:
                raise ValueError("For now, the boosted frame is incompatible "
                    "with non-zero v_antenna.")

        # Initialize virtual particle with 2*Nm values of angle
        nptheta = 2*Nm

        # Porportionality coefficient between the weight of a particle
        # and its transverse position (in cylindrical geometry, particles
        # that are further away from the axis have a larger weight)
        # The larger the weight, the lower the excursion of the particles,
        # in order to emit a given laser (see definition of epsilon)
        alpha_weights = 2*np.pi / ( nptheta*npr*epsilon ) * dr_grid / r_e * e
        # Mobility coefficient: proportionality coefficient between the
        # velocity of the particles and the electric field to be emitted
        self.mobility_coef = 2*np.pi * \
          dr_grid**2 / ( nptheta*npr*alpha_weights ) * epsilon_0 * c
        # Tune the mobility for the boosted-frame
        if boost is not None:
            self.mobility_coef = self.mobility_coef / boost.gamma0
        # Tune the mobility for a moving antenna
        elif v_antenna is not None:
            self.mobility_coef *= \
                (1. - laser_profile.propag_direction*v_antenna/c)

        # Get total number of virtual particles
        Npr = Nr_grid * npr
        Ntot = Npr * nptheta
        # Get the baseline radius and angles of the virtual particles
        r_reg = dr_grid/npr * ( np.arange( Npr ) + 0.5 )
        theta_reg = 2*np.pi/nptheta * np.arange( nptheta )
        rp, thetap = np.meshgrid( r_reg, theta_reg, copy=True)
        self.baseline_r = rp.flatten()
        theta0 = thetap.flatten()

        # Baseline position of the particles and weights
        self.Ntot = Ntot
        self.baseline_x = self.baseline_r * np.cos( theta0 )
        self.baseline_y = self.baseline_r * np.sin( theta0 )
        self.baseline_z = z0_antenna * np.ones( Ntot )
        # NB: all virtual particles have the same baseline_z, but for
        # convenient reuse of other functions, baseline_z is still an array
        self.w = alpha_weights * self.baseline_r / dr_grid
        # Excursion with respect to the baseline position
        # (No excursion in z: the particles do not oscillate in this direction)
        self.excursion_x = np.zeros( Ntot )
        self.excursion_y = np.zeros( Ntot )
        # Particle velocities
        self.vx = np.zeros( Ntot )
        self.vy = np.zeros( Ntot )
        self.vz = np.zeros( Ntot )
        # If the simulation is performed in a boosted frame,
        # boost these quantities
        if boost is not None:
            self.baseline_z, = boost.static_length( [ self.baseline_z ] )
            self.vz, = boost.velocity( [ self.vz ] )
        # If there is a moving antenna, assign velocity
        elif v_antenna != 0:
            self.vz += v_antenna

        # Register whether the antenna deposits on the local domain
        # (gets updated by `update_current_rank`)
        self.deposit_on_this_rank = False

        # Initialize small-size buffers where the particles charge and currents
        # will be deposited before being added to the regular, large-size array
        # (esp. useful when running on GPU, for memory transfer)
        self.rho_buffer = np.empty( (Nm, 2, Nr_grid), dtype='complex' )
        self.Jr_buffer = np.empty( (Nm, 2, Nr_grid), dtype='complex' )
        self.Jt_buffer = np.empty( (Nm, 2, Nr_grid), dtype='complex' )
        self.Jz_buffer = np.empty( (Nm, 2, Nr_grid), dtype='complex' )
        if cuda_installed:
            self.d_rho_buffer = cupy.asarray( self.rho_buffer )
            self.d_Jr_buffer = cupy.asarray( self.Jr_buffer )
            self.d_Jt_buffer = cupy.asarray( self.Jt_buffer )
            self.d_Jz_buffer = cupy.asarray( self.Jz_buffer )

    def update_current_rank(self, comm):
        """
        Determine whether the antenna deposits on the local domain
        of this MPI rank.

        This function is typically called at the same as the
        particle exchange (i.e. at the beginning a PIC iteration).

        One alternative would be to check the antenna position before
        each call to `deposit` ; however this could result in rho^n and
        rho^{n+1} being deposited on different MPI rank, during the same
        PIC iteration - which would lead to spurious effects on the
        current correction.

        Parameters:
        -----------
        comm: a BoundaryCommunicator object
            Contains information on the local boundaries
        """
        # Check if the antenna is in the local physical domain
        # and update the flag `deposit_on_this_rank` accordingly
        zmin_local, zmax_local = comm.get_zmin_zmax(
            local=True, with_damp=True, with_guard=False, rank=comm.rank )
        z_antenna = self.baseline_z[0]
        if (z_antenna >= zmin_local) and (z_antenna < zmax_local):
            self.deposit_on_this_rank = True
        else:
            self.deposit_on_this_rank = False

    def push_x( self, dt, x_push=1., y_push=1., z_push=1. ):
        """
        Push the position of the virtual particles in the antenna
        over timestep `dt`, using their current velocity

        Parameters:
        -----------
        dt: float, seconds
            The timestep that should be used for the push
            (This can be typically be half of the simulation timestep)

        x_push, y_push, z_push: float, dimensionless
            Multiplying coefficient for the velocities in x, y and z
            e.g. if x_push=1., the particles are pushed forward in x
                 if x_push=-1., the particles are pushed backward in x
        """
        # Push transverse particle positions (element-wise array operation)
        self.excursion_x += (dt * x_push) * self.vx
        self.excursion_y += (dt * y_push) * self.vy
        # Move the position of the antenna (element-wise array operation)
        self.baseline_z += (dt * z_push) * self.vz

    def update_v( self, t ):
        """
        Update the particle velocities so that it corresponds to time t

        The updated value of the velocities is determined by calculating
        the electric field at the time t and at the position of the antenna
        and by multiplying this field by the mobility.

        Parameter
        ---------
        t: float (seconds)
            The time at which to calculate the velocities
        """
        # When running in a boosted frame, convert the position and time at
        # which to find the laser amplitude.
        if self.boost is not None:
            boost = self.boost
            inv_c = 1./c
            zlab = boost.gamma0*(  self.baseline_z + (c*boost.beta0)*t )
            tlab = boost.gamma0*( t + (inv_c*boost.beta0)* self.baseline_z )
        else:
            zlab = self.baseline_z
            tlab = t

        # Calculate the electric field to be emitted (in the lab-frame)
        # Eu is the amplitude along the polarization direction
        # Note that we neglect the (small) excursion of the particles when
        # calculating the electric field on the particles.
        Ex, Ey = self.laser_profile.E_field(
            self.baseline_x, self.baseline_y, zlab, tlab )

        # Calculate the corresponding velocity. This takes into account
        # lab-frame to boosted-frame conversion, through a modification
        # of the mobility coefficient: see the __init__ function
        self.vx = self.mobility_coef * Ex
        self.vy = self.mobility_coef * Ey

    def deposit( self, fld, fieldtype ):
        """
        Deposit the charge or current of the virtual particles onto the grid

        This function closely mirrors the deposit function of the regular
        macroparticles, but also introduces a few specific optimization:
        - use the particle velocities instead of the momenta for J
        - deposit the currents and charge into a small-size array

        Parameter
        ----------
        fld : a Field object
             Contains the list of InterpolationGrid objects with
             the field values as well as the prefix sum.

        fieldtype : string
             Indicates which field to deposit
             Either 'J' or 'rho'
        """
        # Interrupt this function if the antenna does not currently
        # deposit on the local domain (as determined by `update_current_rank`)
        if not self.deposit_on_this_rank:
            return

        # Shortcut for the list of InterpolationGrid objects
        grid = fld.interp

        # Set the buffers to zero
        if fieldtype == 'rho':
            self.rho_buffer[:,:,:] = 0.
        elif fieldtype == 'J':
            self.Jr_buffer[:,:,:] = 0.
            self.Jt_buffer[:,:,:] = 0.
            self.Jz_buffer[:,:,:] = 0.

        # Indices and weights in z:
        # same for both the negative and positive virtual particles
        iz, Sz = weights(self.baseline_z, grid[0].invdz, grid[0].zmin, grid[0].Nz,
                         direction='z', shape_order=1,
                         beta_n=grid[0].ruyten_linear_coef)
        # Find the z index where the small-size buffers should be added
        # to the large-size arrays rho, Jr, Jt, Jz
        iz_min = iz.min()
        iz_max = iz.max()
        # Since linear shape are used, and since the virtual particles all
        # have the same z position, iz_max is necessarily equal to iz_min+1
        # This is a sanity check, to avoid out-of-bound access later on.
        assert iz_max == iz_min+1
        # Substract from the array of indices in order to find the particle
        # index within the small-size buffers
        iz = iz - iz_min

        # Deposit the charge/current of positive and negative
        # virtual particles successively, into the small-size buffers
        for q in [-1, 1]:
            self.deposit_virtual_particles( q, fieldtype, grid, iz, Sz )

        # Copy the small-size buffers into the large-size arrays
        # (When running on the GPU, this involves copying the
        # small-size buffers from CPU to GPU)
        if fieldtype == 'rho':
            self.copy_rho_buffer( iz_min, grid )
        elif fieldtype == 'J':
            self.copy_J_buffer( iz_min, grid )

    def deposit_virtual_particles( self, q, fieldtype, grid, iz, Sz ):
        """
        Deposit the charge/current of the positive (q=+1) or negative
        (q=-1) virtual macroparticles

        Parameters
        ----------
        q: float (either +1 or -1)
            Indicates whether to deposit the charge/current
            of the positive or negative virtual macroparticles

        fieldtype: string (either 'rho' or 'J')
            Indicates whether to deposit the charge or current

        grid: a list of InterpolationGrid object
            The grids on which to the deposit the charge/current

        iz, ir : 2darray of ints
            Arrays of shape (shape_order+1, Ntot)
            where Ntot is the number of macroparticles.
            Contains the index of the cells that each virtual macroparticle
            will deposit to.
            (In the case of the laser antenna, these arrays are constant
            in principle; but they are kept as arrays for compatibility
            with the deposit_field_numba function.)

        Sz, Sr: 2darray of ints
            Arrays of shape (shape_order+1, Ntot)
            where Ntot is the number of macroparticles
            Contains the weight for respective cells from iz and ir,
            for each macroparticle.
        """
        # Position of the particles
        x = self.baseline_x + q*self.excursion_x
        y = self.baseline_y + q*self.excursion_y
        vx = q*self.vx
        vy = q*self.vy
        w = q*self.w

        # Preliminary arrays for the cylindrical conversion
        r = np.sqrt( x**2 + y**2 )
        # Avoid division by 0.
        invr = 1./np.where( r!=0., r, 1. )
        cos = np.where( r!=0., x*invr, 1. )
        sin = np.where( r!=0., y*invr, 0. )

        if fieldtype == 'rho' :
            # ---------------------------------------
            # Deposit the charge density mode by mode
            # ---------------------------------------
            # Prepare auxiliary matrix
            exptheta = np.ones( self.Ntot, dtype='complex')
            # exptheta takes the value exp(im theta) throughout the loop
            for m in range( len(grid) ) :
                # Increment exptheta (notice the + : forward transform)
                if m==1 :
                    exptheta[:].real = cos
                    exptheta[:].imag = sin
                elif m>1 :
                    exptheta[:] = exptheta*( cos + 1.j*sin )

                # Indices and weights in r
                ir, Sr = weights(r, grid[m].invdr, grid[m].rmin, grid[m].Nr,
                         direction='r', shape_order=1,
                         beta_n=grid[m].ruyten_linear_coef)

                # Deposit the fields into small-size buffer arrays
                deposit_field_numba( w*exptheta, self.rho_buffer[m,:],
                    iz, ir, Sz, Sr, (-1)**m )

        elif fieldtype == 'J' :
            # ----------------------------------------
            # Deposit the current density mode by mode
            # ----------------------------------------
            # Calculate the currents
            Jr = w * ( cos*vx + sin*vy )
            Jt = w * ( cos*vy - sin*vx )
            Jz = w * self.vz
            # Prepare auxiliary matrix
            exptheta = np.ones( self.Ntot, dtype='complex')
            # exptheta takes the value exp(im theta) throughout the loop
            for m in range( len(grid) ) :
                # Increment exptheta (notice the + : forward transform)
                if m==1 :
                    exptheta[:].real = cos
                    exptheta[:].imag = sin
                elif m>1 :
                    exptheta[:] = exptheta*( cos + 1.j*sin )

                # Indices and weights in r
                ir, Sr = weights(r, grid[m].invdr, grid[m].rmin, grid[m].Nr,
                         direction='r', shape_order=1,
                         beta_n=grid[m].ruyten_linear_coef)

                # Deposit the fields into small-size buffer arrays
                deposit_field_numba( Jr*exptheta, self.Jr_buffer[m,:],
                                     iz, ir, Sz, Sr, -(-1)**m )
                deposit_field_numba( Jt*exptheta, self.Jt_buffer[m,:],
                                     iz, ir, Sz, Sr, -(-1)**m )
                deposit_field_numba( Jz*exptheta, self.Jz_buffer[m,:],
                                     iz, ir, Sz, Sr, (-1)**m )

    def copy_rho_buffer( self, iz_min, grid ):
        """
        Add the small-size array rho_buffer into the full-size array rho

        Parameters
        ----------
        iz_min: int
            The z index in the full-size array, that corresponds to index 0
            in the small-size array (i.e. position at which to add the
            small-size array into the full-size one)

        grid: a list of InterpolationGrid objects
            Contains the full-size array rho
        """
        Nm = len(grid)
        if type(grid[0].rho) is np.ndarray:
            # The large-size array rho is on the CPU
            for m in range( Nm ):
                grid[m].rho[ iz_min:iz_min+2 ] += self.rho_buffer[m]
        else:
            # The large-size array rho is on the GPU
            # Copy the small-size buffer to the GPU
            self.d_rho_buffer.set( self.rho_buffer)
            # On the GPU: add the small-size buffers to the large-size array
            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( grid[0].Nr, TPB=64 )
            for m in range( Nm ):
                add_rho_to_gpu_array[dim_grid_1d, dim_block_1d]( iz_min,
                            self.d_rho_buffer, grid[m].rho, m )

    def copy_J_buffer( self, iz_min, grid ):
        """
        Add the small-size arrays Jr_buffer, Jt_buffer, Jz_buffer into
        the full-size arrays Jr, Jt, Jz

        Parameters
        ----------
        iz_min: int
            The z index in the full-size array, that corresponds to index 0
            in the small-size array (i.e. position at which to add the
            small-size array into the full-size one)

        grid: a list of InterpolationGrid objects
            Contains the full-size array Jr, Jt, Jz
        """
        Nm = len(grid)
        if type(grid[0].Jr) is np.ndarray:
            # The large-size arrays for J are on the CPU
            for m in range( Nm ):
                grid[m].Jr[ iz_min:iz_min+2 ] += self.Jr_buffer[m]
                grid[m].Jt[ iz_min:iz_min+2 ] += self.Jt_buffer[m]
                grid[m].Jz[ iz_min:iz_min+2 ] += self.Jz_buffer[m]
        else:
            # The large-size arrays for J are on the GPU
            # Copy the small-size buffers to the GPU
            self.d_Jr_buffer.set( self.Jr_buffer)
            self.d_Jt_buffer.set( self.Jt_buffer)
            self.d_Jz_buffer.set( self.Jz_buffer)
            # On the GPU: add the small-size buffers to the large-size array
            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( grid[0].Nr, TPB=64 )
            for m in range( Nm ):
                add_J_to_gpu_array[dim_grid_1d, dim_block_1d]( iz_min,
                    self.d_Jr_buffer, self.d_Jt_buffer, self.d_Jz_buffer,
                    grid[m].Jr, grid[m].Jt, grid[m].Jz, m )

if cuda_installed:

    @compile_cupy
    def add_rho_to_gpu_array( iz_min, rho_buffer, rho, m ):
        """
        Add the small-size array rho_buffer into the full-size array rho
        on the GPU

        Parameters
        ----------
        iz_min: int
            The index of the lowest cell in z that surrounds the antenna

        rho_buffer: 3darray of complexs
            Array of shape (Nm, 2, Nr) that stores the values of rho
            in the 2 cells that surround the antenna (for each mode).

        rho: 2darray of complexs
            Array of shape (Nz, Nr) that contains rho in the mode m

        m: int
           The index of the azimuthal mode involved
        """
        # Use one thread per radial cell
        ir = cuda.grid(1)

        # Add the values
        if ir < rho.shape[1]:
            rho[iz_min, ir] += rho_buffer[m, 0, ir]
            rho[iz_min+1, ir] += rho_buffer[m, 1, ir]

    @compile_cupy
    def add_J_to_gpu_array( iz_min, Jr_buffer, Jt_buffer,
                            Jz_buffer, Jr, Jt, Jz, m ):
        """
        Add the small-size arrays Jr_buffer, Jt_buffer, Jz_buffer into
        the full-size arrays Jr, Jt, Jz on the GPU

        Parameters:
        -----------
        iz_min: int

        Jr_buffer, Jt_buffer, Jz_buffer: 3darrays of complexs
            Arrays of shape (Nm, 2, Nr) that store the values of rho
            in the 2 cells that surround the antenna (for each mode).

        Jr, Jt, Jz: 2darrays of complexs
            Arrays of shape (Nz, Nr) that contain rho in the mode m

        m: int
           The index of the azimuthal mode involved
        """
        # Use one thread per radial cell
        ir = cuda.grid(1)

        # Add the values
        if ir < Jr.shape[1]:
            Jr[iz_min, ir] += Jr_buffer[m, 0, ir]
            Jr[iz_min+1, ir] += Jr_buffer[m, 1, ir]

            Jt[iz_min, ir] += Jt_buffer[m, 0, ir]
            Jt[iz_min+1, ir] += Jt_buffer[m, 1, ir]

            Jz[iz_min, ir] += Jz_buffer[m, 0, ir]
            Jz[iz_min+1, ir] += Jz_buffer[m, 1, ir]
