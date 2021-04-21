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
from fbpic.particles.deposition.threading_methods import \
        deposit_rho_numba_linear, deposit_J_numba_linear
from fbpic.utils.threading import nthreads, get_chunk_indices

# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    import cupy
    from fbpic.utils.cuda import cuda_tpb_bpg_1d
    from fbpic.particles.deposition.cuda_methods_unsorted import \
        deposit_rho_gpu_unsorted, deposit_J_gpu_unsorted

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

    Not all laser profiles are available on the GPU, so the update of the
    virtual particle velocity is performed on either CPU or GPU depending on
    whether the laser profile is GPU enabled. The deposition of charge and
    current is then always performed on GPU in the usual way as long as
    CUDA is available. For this, the velocities are copied to the GPU if needed.
    Note that the antenna always uses linear shape factors (even when the rest of
    the simulation uses cubic shape factors.)
    """
    def __init__( self, laser_profile, z0_antenna, v_antenna,
                    dr_grid, Nr_grid, Nm, boost, npr=2, epsilon=0.01,
                    use_cuda=False ):
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

        use_cuda: bool
            Whether to use CUDA for the antenna
        """
        # Register the properties of the laser injection
        self.laser_profile = laser_profile
        self.boost = boost

        self.use_cuda = use_cuda

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
        # Inverse gamma; used only for the CPU deposition kernels
        self.inv_gamma = np.ones( Ntot )
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
    
        # Copy all required arrays to GPU if needed
        # Some of the arrays are kept as copies on CPU in the case that
        # the laser profile is not GPU capable
        if use_cuda:

            self.d_baseline_x = cupy.asarray( self.baseline_x )
            self.d_baseline_y = cupy.asarray( self.baseline_y )
            self.d_baseline_z = cupy.asarray( self.baseline_z )

            self.excursion_x = cupy.asarray( self.excursion_x )
            self.excursion_y = cupy.asarray( self.excursion_y )

            self.vx = cupy.asarray( self.vx )
            self.vy = cupy.asarray( self.vy )
            self.d_vz = cupy.asarray( self.vz )

            self.w = cupy.asarray( self.w )


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

        # If possible, the antenna position is updated on both GPU and CPU
        if self.use_cuda:
            self.d_baseline_z += (dt * z_push) * self.d_vz

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
        # If the laser profile supports it, do the whole calculation on GPU
        if self.use_cuda and self.laser_profile.gpu_capable:
            x = self.d_baseline_x
            y = self.d_baseline_y
            z = self.d_baseline_z
        else:
            x = self.baseline_x
            y = self.baseline_y
            z = self.baseline_z

        # When running in a boosted frame, convert the position and time at
        # which to find the laser amplitude.
        if self.boost is not None:
            boost = self.boost
            inv_c = 1./c
            zlab = boost.gamma0*(  z + (c*boost.beta0)*t )
            tlab = boost.gamma0*( t + (inv_c*boost.beta0)* z )
        else:
            zlab = z
            tlab = t

        # Calculate the electric field to be emitted (in the lab-frame)
        # Eu is the amplitude along the polarization direction
        # Note that we neglect the (small) excursion of the particles when
        # calculating the electric field on the particles.
        Ex, Ey = self.laser_profile.E_field(
            x, y, zlab, tlab )

        # Calculate the corresponding velocity. This takes into account
        # lab-frame to boosted-frame conversion, through a modification
        # of the mobility coefficient: see the __init__ function

        # Copy the velocities to GPU if the laser profile doesnt support GPU
        if self.use_cuda and not( self.laser_profile.gpu_capable ):
            self.vx.set( self.mobility_coef * Ex )
            self.vy.set( self.mobility_coef * Ey )
        else:
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

        # Deposit the charge/current of positive and negative
        # virtual particles successively
        for q in [-1, 1]:

            if self.use_cuda:
                self.deposit_virtual_particles_gpu( q, fieldtype, grid )
            else:
                self.deposit_virtual_particles_cpu( q, fieldtype, grid, fld )
            

    def deposit_virtual_particles_gpu( self, q, fieldtype, grid ):
        # Position of the particles
        x = self.d_baseline_x + q*self.excursion_x
        y = self.d_baseline_y + q*self.excursion_y

        if fieldtype == 'rho' :
            # ---------------------------------------
            # Deposit the charge density mode by mode
            # ---------------------------------------
            for m in range( len(grid) ) :
       
                dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( self.Ntot )
                deposit_rho_gpu_unsorted[
                    dim_grid_1d, dim_block_1d](
                    x, y, self.d_baseline_z, self.w, q,
                    grid[m].invdz, grid[m].zmin, grid[m].Nz,
                    grid[m].invdr, grid[m].rmin, grid[m].Nr,
                    grid[m].rho, m, grid[m].d_ruyten_linear_coef)

        elif fieldtype == 'J' :
            # Particle velocities
            vx = q*self.vx
            vy = q*self.vy
            # ---------------------------------------
            # Deposit the current density mode by mode
            # ---------------------------------------
            for m in range( len(grid) ) :
        
                dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( self.Ntot )
                deposit_J_gpu_unsorted[
                    dim_grid_1d, dim_block_1d](
                    x, y, self.d_baseline_z, self.w, q,
                    vx, vy, self.d_vz,
                    grid[m].invdz, grid[m].zmin, grid[m].Nz,
                    grid[m].invdr, grid[m].rmin, grid[m].Nr,
                    grid[m].Jr, grid[m].Jt, grid[m].Jz,
                    m, grid[m].d_ruyten_linear_coef)

    def deposit_virtual_particles_cpu( self, q, fieldtype, grid, fld ):
        x = self.baseline_x + q*self.excursion_x
        y = self.baseline_y + q*self.excursion_y

        # Divide particles in chunks (each chunk is handled by a different
        # thread) and register the indices that bound each chunks
        ptcl_chunk_indices = get_chunk_indices(self.Ntot, nthreads)

        # The set of Ruyten shape coefficients to use for higher modes. 
        # For Nm > 1, the set from mode 1 is used, since all higher modes have the
        # same coefficients. For Nm == 1, the coefficients from mode 0 are 
        # passed twice to satisfy the argument types for Numba JIT.
        if fld.Nm > 1:
            ruyten_m = 1
        else: 
            ruyten_m = 0

        if fieldtype == 'rho' :
            # ---------------------------------------
            # Deposit the charge density all modes at once
            # ---------------------------------------
            deposit_rho_numba_linear(
                x, y, self.baseline_z, self.w, q,
                grid[0].invdz, grid[0].zmin, grid[0].Nz,
                grid[0].invdr, grid[0].rmin, grid[0].Nr,
                fld.rho_global, fld.Nm,
                nthreads, ptcl_chunk_indices,
                grid[0].ruyten_linear_coef,
                grid[ruyten_m].ruyten_linear_coef )


        elif fieldtype == 'J' :
            # Calculate the relativistic momenta from the velocities.
            # The gamma is set to 1 both here and in the deposition kernel. 
            # This is alright since the deposition only depends on the products
            # ux*inv_gamma, uy*inv_gamma and uz*inv_gamma, which correspond to
            # vx/c, vy/c and vz/c, respectively. So as long as the products are
            # correct, passing inv_gamma = 1 is no issue.
            ux = q*self.vx / c
            uy = q*self.vy / c
            uz = self.vz / c

            # ---------------------------------------
            # Deposit the current density all modes at once
            # ---------------------------------------
            deposit_J_numba_linear(
                x, y, self.baseline_z, self.w, q,
                ux, uy, uz, self.inv_gamma,
                grid[0].invdz, grid[0].zmin, grid[0].Nz,
                grid[0].invdr, grid[0].rmin, grid[0].Nr,
                fld.Jr_global, fld.Jt_global, fld.Jz_global, fld.Nm,
                nthreads, ptcl_chunk_indices,
                grid[0].ruyten_linear_coef,
                grid[ruyten_m].ruyten_linear_coef )