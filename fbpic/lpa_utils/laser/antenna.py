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
from .profiles import gaussian_profile
from fbpic.particles.utility_methods import weights
from fbpic.particles.numba_methods import deposit_field_numba

# Check if CUDA is available, then import CUDA functions
from fbpic.cuda_utils import cuda_installed
if cuda_installed:
    from fbpic.cuda_utils import cuda, cuda_tpb_bpg_1d

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
    def __init__( self, E0, w0, ctau, z0, zf, k0, cep_phase,
        phi2_chirp, theta_pol, z0_antenna, dr_grid, Nr_grid, Nm,
        npr=2, nptheta=4, epsilon=0.01, boost=None ):
        """
        Initialize a LaserAntenna object (see class docstring for more info)

        Parameters
        ----------
        E0: float (V.m^-1)
            The amplitude of the the electric field *in the lab frame*

        w0: float (m)
            The waist of the laser at focus

        ctau: float (m)
            The duration of the laser *in the lab frame*

        z0: float (m)
            The initial position of the laser centroid *in the lab frame*

        zf: float (m)
            The position of the focal plane *in the lab frame*

        k0: float (m^-1)
            Laser wavevector *in the lab frame*

        cep_phase: float (rad)
            Carrier Enveloppe Phase (CEP), i.e. the phase of the laser
            oscillations, at the position where the laser enveloppe is maximum.

        phi2_chirp: float (in second^2)
            The amount of temporal chirp, at focus *in the lab frame*
            Namely, a wave packet centered on the frequency (w0 + dw) will
            reach its peak intensity at a time z(dw) = z0 - c*phi2*dw.
            Thus, a positive phi2 corresponds to positive chirp, i.e. red part
            of the spectrum in the front of the pulse and blue part of the
            spectrum in the back.

        theta_pol: float (rad)
            Polarization angle of the laser

        z0_antenna: float (m)
            Initial position of the antenna *in the lab frame*

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
        if boost is not None:
            self.mobility_coef = self.mobility_coef / boost.gamma0

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

        # Record laser properties
        self.E0 = E0
        self.w0 = w0
        self.k0 = k0
        self.ctau = ctau
        self.z0 = z0
        self.zf = zf
        self.cep_phase = cep_phase
        self.phi2_chirp = phi2_chirp
        self.theta_pol = theta_pol
        self.boost = boost

        # Initialize small-size buffers where the particles charge and currents
        # will be deposited before being added to the regular, large-size array
        # (esp. useful when running on GPU, for memory transfer)
        self.rho_buffer = np.empty( (Nm, 2, Nr_grid), dtype='complex' )
        self.Jr_buffer = np.empty( (Nm, 2, Nr_grid), dtype='complex' )
        self.Jt_buffer = np.empty( (Nm, 2, Nr_grid), dtype='complex' )
        self.Jz_buffer = np.empty( (Nm, 2, Nr_grid), dtype='complex' )
        if cuda_installed:
            self.d_rho_buffer = cuda.device_array_like( self.rho_buffer )
            self.d_Jr_buffer = cuda.device_array_like( self.Jr_buffer )
            self.d_Jt_buffer = cuda.device_array_like( self.Jt_buffer )
            self.d_Jz_buffer = cuda.device_array_like( self.Jz_buffer )

    def halfpush_x( self, dt ):
        """
        Push the position of the virtual particles in the antenna
        over half a timestep, using their current velocity

        Parameter
        ---------
        dt: float (seconds)
            The (full) timestep of the simulation
        """
        # Half timestep
        hdt = 0.5*dt

        # Push transverse particle positions (element-wise array operation)
        self.excursion_x += hdt * self.vx
        self.excursion_y += hdt * self.vy
        # Move the position of the antenna (element-wise array operation)
        self.baseline_z += hdt * self.vz

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
        # Calculate the electric field to be emitted (in the lab-frame)
        # Eu is the amplitude along the polarization direction
        # Note that we neglect the (small) excursion of the particles when
        # calculating the electric field on the particles.
        Eu = self.E0 * gaussian_profile( self.baseline_z, self.baseline_r, t,
                        self.w0, self.ctau, self.z0, self.zf,
                        self.k0, self.cep_phase, self.phi2_chirp,
                        boost=self.boost, output_Ez_profile=False )

        # Calculate the corresponding velocity. This takes into account
        # lab-frame to boosted-frame conversion, through a modification
        # of the mobility coefficient: see the __init__ function
        self.vx = ( self.mobility_coef * np.cos(self.theta_pol) ) * Eu
        self.vy = ( self.mobility_coef * np.sin(self.theta_pol) ) * Eu

    def deposit( self, fld, fieldtype, comm ):
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

        comm : a BoundaryCommunicator object
             Allows to extract the boundaries of the physical domain
        """
        # Check if baseline_z is in the local physical domain
        # (This prevents out-of-bounds errors, and prevents 2 neighboring
        # processors from simultaneously depositing the laser antenna)
        zmin_local = fld.interp[0].zmin
        zmax_local = fld.interp[0].zmax
        # If a communicator is provided, remove the guard cells
        if comm is not None:
            dz = fld.interp[0].dz
            zmin_local += dz*comm.n_guard
            zmax_local -= dz*comm.n_guard
        # Interrupt this function if the antenna is not in the local domain
        z_antenna = self.baseline_z[0]
        if (z_antenna < zmin_local) or (z_antenna >= zmax_local):
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
                         direction='z', shape_order=1)
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

        # Indices and weights in r
        ir, Sr = weights(r, grid[0].invdr, grid[0].rmin, grid[0].Nr,
                         direction='r', shape_order=1)

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
                # Deposit the fields into small-size buffer arrays
                # (The sign -1 with which the guards are added is not
                # trivial to derive but avoids artifacts on the axis)
                deposit_field_numba( w*exptheta, self.rho_buffer[m,:],
                    iz, ir, Sz, Sr, -1.)

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
                # Deposit the fields into small-size buffer arrays
                # (The sign -1 with which the guards are added is not
                # trivial to derive but avoids artifacts on the axis)
                deposit_field_numba( Jr*exptheta, self.Jr_buffer[m,:],
                                     iz, ir, Sz, Sr, -1.)
                deposit_field_numba( Jt*exptheta, self.Jt_buffer[m,:],
                                     iz, ir, Sz, Sr, -1.)
                deposit_field_numba( Jz*exptheta, self.Jz_buffer[m,:],
                                     iz, ir, Sz, Sr, -1.)

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
        if type(grid[0].rho) is np.ndarray:
            # The large-size array rho is on the CPU
            for m in range( len(grid) ):
                grid[m].rho[ iz_min:iz_min+2 ] += self.rho_buffer[m]
        else:
            # The large-size array rho is on the GPU
            # Copy the small-size buffer to the GPU
            cuda.to_device( self.rho_buffer, to=self.d_rho_buffer )
            # On the GPU: add the small-size buffers to the large-size array
            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( grid[0].Nr, TPB=64 )
            add_rho_to_gpu_array[dim_grid_1d, dim_block_1d]( iz_min,
                            self.d_rho_buffer, grid[0].rho, grid[1].rho )

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
        if type(grid[0].Jr) is np.ndarray:
            # The large-size arrays for J are on the CPU
            for m in range( len(grid) ):
                grid[m].Jr[ iz_min:iz_min+2 ] += self.Jr_buffer[m]
                grid[m].Jt[ iz_min:iz_min+2 ] += self.Jt_buffer[m]
                grid[m].Jz[ iz_min:iz_min+2 ] += self.Jz_buffer[m]
        else:
            # The large-size arrays for J are on the GPU
            # Copy the small-size buffers to the GPU
            cuda.to_device( self.Jr_buffer, to=self.d_Jr_buffer )
            cuda.to_device( self.Jt_buffer, to=self.d_Jt_buffer )
            cuda.to_device( self.Jz_buffer, to=self.d_Jz_buffer )
            # On the GPU: add the small-size buffers to the large-size array
            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( grid[0].Nr, TPB=64 )
            add_J_to_gpu_array[dim_grid_1d, dim_block_1d]( iz_min,
                    self.d_Jr_buffer, self.d_Jt_buffer, self.d_Jz_buffer,
                    grid[0].Jr, grid[1].Jr, grid[0].Jt, grid[1].Jt,
                    grid[0].Jz, grid[1].Jz )

if cuda_installed:

    @cuda.jit()
    def add_rho_to_gpu_array( iz_min, rho_buffer, rho0, rho1 ):
        """
        Add the small-size array rho_buffer into the full-size array rho
        on the GPU
        """
        # Use one thread per radial cell
        ir = cuda.grid(1)

        # Add the values
        if ir < rho0.shape[1]:
            rho0[iz_min, ir] += rho_buffer[0, 0, ir]
            rho0[iz_min+1, ir] += rho_buffer[0, 1, ir]
            rho1[iz_min, ir] += rho_buffer[1, 0, ir]
            rho1[iz_min+1, ir] += rho_buffer[1, 1, ir]

    @cuda.jit()
    def add_J_to_gpu_array( iz_min, Jr_buffer, Jt_buffer, Jz_buffer,
            Jr0, Jr1, Jt0, Jt1, Jz0, Jz1 ):
        """
        Add the small-size arrays Jr_buffer, Jt_buffer, Jz_buffer into
        the full-size arrays Jr, Jt, Jz on the GPU
        """
        # Use one thread per radial cell
        ir = cuda.grid(1)

        # Add the values
        if ir < Jr0.shape[1]:
            Jr0[iz_min, ir] += Jr_buffer[0, 0, ir]
            Jr0[iz_min+1, ir] += Jr_buffer[0, 1, ir]
            Jr1[iz_min, ir] += Jr_buffer[1, 0, ir]
            Jr1[iz_min+1, ir] += Jr_buffer[1, 1, ir]

            Jt0[iz_min, ir] += Jt_buffer[0, 0, ir]
            Jt0[iz_min+1, ir] += Jt_buffer[0, 1, ir]
            Jt1[iz_min, ir] += Jt_buffer[1, 0, ir]
            Jt1[iz_min+1, ir] += Jt_buffer[1, 1, ir]

            Jz0[iz_min, ir] += Jz_buffer[0, 0, ir]
            Jz0[iz_min+1, ir] += Jz_buffer[0, 1, ir]
            Jz1[iz_min, ir] += Jz_buffer[1, 0, ir]
            Jz1[iz_min+1, ir] += Jz_buffer[1, 1, ir]
