# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure and methods associated with the particles.
"""
import numpy as np
from scipy.constants import c, e
from .ionization import Ionizer
from .tracking import ParticleTracker

# Load the utility methods
from .utility_methods import weights, unalign_angles
# Load the numba routines
from .numba_methods import push_p_numba, push_p_ioniz_numba, push_x_numba, \
        gather_field_numba, deposit_field_numba

# Check if CUDA is available, then import CUDA functions
from fbpic.cuda_utils import cuda_installed
if cuda_installed:
    from fbpic.cuda_utils import cuda, cuda_tpb_bpg_1d, cuda_tpb_bpg_2d
    from .cuda_methods import push_p_gpu, push_p_ioniz_gpu, push_x_gpu, \
        gather_field_gpu_linear, gather_field_gpu_cubic, \
        write_sorting_buffer, cuda_deposition_arrays, \
        get_cell_idx_per_particle, sort_particles_per_cell, \
        reset_prefix_sum, incl_prefix_sum
    from .cuda_deposition.cubic import deposit_rho_gpu_cubic, \
        deposit_J_gpu_cubic
    from .cuda_deposition.linear import deposit_rho_gpu_linear, \
        deposit_J_gpu_linear
    from .cuda_deposition.linear_non_atomic import deposit_rho_gpu, \
        deposit_J_gpu, add_rho, add_J

class Particles(object) :
    """
    Class that contains the particles data of the simulation

    Main attributes
    ---------------
    - x, y, z : 1darrays containing the Cartesian positions
                of the macroparticles (in meters)
    - uz, uy, uz : 1darrays containing the unitless momenta
                (i.e. px/mc, py/mc, pz/mc)
    At the end or start of any PIC cycle, the momenta should be
    one half-timestep *behind* the position.
    """
    def __init__(self, q, m, n, Npz, zmin, zmax,
                    Npr, rmin, rmax, Nptheta, dt,
                    ux_m=0., uy_m=0., uz_m=0.,
                    ux_th=0., uy_th=0., uz_th=0.,
                    dens_func=None, continuous_injection=True,
                    use_cuda=False, grid_shape=None, particle_shape='linear' ) :
        """
        Initialize a uniform set of particles

        Parameters
        ----------
        q : float (in Coulombs)
           Charge of the particle species

        m : float (in kg)
           Mass of the particle species

        n : float (in particles per m^3)
           Peak density of particles

        Npz : int
           Number of macroparticles along the z axis

        zmin, zmax : floats (in meters)
           z positions between which the particles are initialized

        Npr : int
           Number of macroparticles along the r axis

        rmin, rmax : floats (in meters)
           r positions between which the particles are initialized

        Nptheta : int
           Number of macroparticules along theta

        dt : float (in seconds)
           The timestep for the particle pusher

        ux_m, uy_m, uz_m: floats (dimensionless), optional
           Normalized mean momenta of the injected particles in each direction

        ux_th, uy_th, uz_th: floats (dimensionless), optional
           Normalized thermal momenta in each direction

        dens_func : callable, optional
           A function of the form :
           def dens_func( z, r ) ...
           where z and r are 1d arrays, and which returns
           a 1d array containing the density *relative to n*
           (i.e. a number between 0 and 1) at the given positions

        continuous_injection : bool, optional
           Whether to continuously inject the particles,
           in the case of a moving window

        use_cuda : bool, optional
            Wether to use the GPU or not.

        grid_shape: tuple, optional
            Needed when running on the GPU
            The shape of the local grid (including guard cells), i.e.
            a tuple of the form (Nz, Nr). This is needed in order
            to initialize the sorting of the particles per cell.

        particle_shape: str, optional
            Set the particle shape for the charge/current deposition.
            Possible values are 'cubic', 'linear' and 'linear_non_atomic'.
            While 'cubic' corresponds to third order shapes and 'linear'
            to first order shapes, 'linear_non_atomic' uses an equivalent
            deposition scheme to 'linear' which avoids atomics on the GPU.
        """
        # Register the timestep
        self.dt = dt

        # Define wether or not to use the GPU
        self.use_cuda = use_cuda
        if (self.use_cuda==True) and (cuda_installed==False) :
            print('*** Cuda not available for the particles.')
            print('*** Performing the particle operations on the CPU.')
            self.use_cuda = False

        # Register the properties of the particles
        # (Necessary for the pusher, and when adding more particles later, )
        Ntot = Npz*Npr*Nptheta
        self.Ntot = Ntot
        self.q = q
        self.m = m
        self.n = n
        self.rmin = rmin
        self.rmax = rmax
        self.Npr = Npr
        self.Nptheta = Nptheta
        self.dens_func = dens_func
        self.continuous_injection = continuous_injection

        # Initialize the momenta
        self.uz = uz_m * np.ones(Ntot) + uz_th * np.random.normal(size=Ntot)
        self.ux = ux_m * np.ones(Ntot) + ux_th * np.random.normal(size=Ntot)
        self.uy = uy_m * np.ones(Ntot) + uy_th * np.random.normal(size=Ntot)
        self.inv_gamma = 1./np.sqrt(
            1 + self.ux**2 + self.uy**2 + self.uz**2 )

        # Initilialize the fields array (at the positions of the particles)
        self.Ez = np.zeros( Ntot )
        self.Ex = np.zeros( Ntot )
        self.Ey = np.zeros( Ntot )
        self.Bz = np.zeros( Ntot )
        self.Bx = np.zeros( Ntot )
        self.By = np.zeros( Ntot )

        # Allocate the positions and weights of the particles,
        # and fill them with values if the array is not empty
        self.x = np.empty( Ntot )
        self.y = np.empty( Ntot )
        self.z = np.empty( Ntot )
        self.w = np.empty( Ntot )

        # By default, there is no particle tracking (see method track)
        self.tracker = None
        # By default, the species is not ionizable (see method make_ionizable)
        self.ionizer = None
        # Total number of quantities (necessary in MPI communications)
        self.n_integer_quantities = 0
        self.n_float_quantities = 8 # x, y, z, ux, uy, uz, inv_gamma, w

        if Ntot > 0:
            # Get the 1d arrays of evenly-spaced positions for the particles
            dz = (zmax-zmin)*1./Npz
            z_reg =  zmin + dz*( np.arange(Npz) + 0.5 )
            dr = (rmax-rmin)*1./Npr
            r_reg =  rmin + dr*( np.arange(Npr) + 0.5 )
            dtheta = 2*np.pi/Nptheta
            theta_reg = dtheta * np.arange(Nptheta)

            # Get the corresponding particles positions
            # (copy=True is important here, since it allows to
            # change the angles individually)
            zp, rp, thetap = np.meshgrid( z_reg, r_reg, theta_reg, copy=True)
            # Prevent the particles from being aligned along any direction
            unalign_angles( thetap, Npr, Npz, method='random' )
            # Flatten them (This performs a memory copy)
            r = rp.flatten()
            self.x[:] = r * np.cos( thetap.flatten() )
            self.y[:] = r * np.sin( thetap.flatten() )
            self.z[:] = zp.flatten()
            # Get the weights (i.e. charge of each macroparticle), which
            # are equal to the density times the volume r d\theta dr dz
            self.w[:] = q * n * r * dtheta*dr*dz
            # Modulate it by the density profile
            if dens_func is not None :
                self.w[:] = self.w * dens_func( self.z, r )

        # Allocate arrays and register variables when using CUDA
        if self.use_cuda:
            if grid_shape is None:
                raise ValueError("A `grid_shape` is needed when running "
                "on the GPU.\nPlease provide it when initializing particles.")
            # Register grid shape
            self.grid_shape = grid_shape
            # Allocate arrays for the particles sorting when using CUDA
            self.cell_idx = np.empty( Ntot, dtype=np.int32)
            self.sorted_idx = np.empty( Ntot, dtype=np.uint32)
            self.sorting_buffer = np.empty( Ntot, dtype=np.float64 )
            self.prefix_sum = np.empty( grid_shape[0]*grid_shape[1],
                                        dtype=np.int32 )
            # Register boolean that records if the particles are sorted or not
            self.sorted = False

        # Register particle shape
        self.particle_shape = particle_shape

    def send_particles_to_gpu( self ):
        """
        Copy the particles to the GPU.
        Particle arrays of self now point to the GPU arrays.
        """
        if self.use_cuda:
            # Send positions, velocities, inverse gamma and weights
            # to the GPU (CUDA)
            self.x = cuda.to_device(self.x)
            self.y = cuda.to_device(self.y)
            self.z = cuda.to_device(self.z)
            self.ux = cuda.to_device(self.ux)
            self.uy = cuda.to_device(self.uy)
            self.uz = cuda.to_device(self.uz)
            self.inv_gamma = cuda.to_device(self.inv_gamma)
            self.w = cuda.to_device(self.w)

            # Copy arrays on the GPU for the field
            # gathering and the particle push
            self.Ex = cuda.to_device(self.Ex)
            self.Ey = cuda.to_device(self.Ey)
            self.Ez = cuda.to_device(self.Ez)
            self.Bx = cuda.to_device(self.Bx)
            self.By = cuda.to_device(self.By)
            self.Bz = cuda.to_device(self.Bz)

            # Copy arrays on the GPU for the sorting
            self.cell_idx = cuda.to_device(self.cell_idx)
            self.sorted_idx = cuda.to_device(self.sorted_idx)
            self.prefix_sum = cuda.to_device(self.prefix_sum)
            self.sorting_buffer = cuda.to_device(self.sorting_buffer)
            if self.n_integer_quantities > 0:
                self.int_sorting_buffer = cuda.to_device(self.int_sorting_buffer)

            # Copy particle tracker data
            if self.tracker is not None:
                self.tracker.send_to_gpu()
            # Copy the ionization data
            if self.ionizer is not None:
                self.ionizer.send_to_gpu()

    def receive_particles_from_gpu( self ):
        """
        Receive the particles from the GPU.
        Particle arrays are accessible by the CPU again.
        """
        if self.use_cuda:
            # Copy the positions, velocities, inverse gamma and weights
            # to the GPU (CUDA)
            self.x = self.x.copy_to_host()
            self.y = self.y.copy_to_host()
            self.z = self.z.copy_to_host()
            self.ux = self.ux.copy_to_host()
            self.uy = self.uy.copy_to_host()
            self.uz = self.uz.copy_to_host()
            self.inv_gamma = self.inv_gamma.copy_to_host()
            self.w = self.w.copy_to_host()

            # Copy arrays on the CPU for the field
            # gathering and the particle push
            self.Ex = self.Ex.copy_to_host()
            self.Ey = self.Ey.copy_to_host()
            self.Ez = self.Ez.copy_to_host()
            self.Bx = self.Bx.copy_to_host()
            self.By = self.By.copy_to_host()
            self.Bz = self.Bz.copy_to_host()

            # Copy arrays on the CPU
            # that represent the sorting arrays
            self.cell_idx = self.cell_idx.copy_to_host()
            self.sorted_idx = self.sorted_idx.copy_to_host()
            self.prefix_sum = self.prefix_sum.copy_to_host()
            self.sorting_buffer = self.sorting_buffer.copy_to_host()
            if self.n_integer_quantities > 0:
                self.int_sorting_buffer = self.int_sorting_buffer.copy_to_host()

            # Copy particle tracker data
            if self.tracker is not None:
                self.tracker.receive_from_gpu()
            # Copy the ionization data
            if self.ionizer is not None:
                self.ionizer.receive_from_gpu()

    def track( self, comm ):
        """
        Activate particle tracking for the current species
        (i.e. allocates an array of ids, that is communicated through MPI
        and sorting, and is output in the openPMD file)

        Parameters:
        -----------
        comm: an fbpic.BoundaryCommunicator object
            Contains information about the number of processors
        """
        self.tracker = ParticleTracker( comm.size, comm.rank, self.Ntot )
        # Update the number of integer quantities
        self.n_integer_quantities += 1
        # Allocate the integer sorting buffer if needed
        if hasattr( self, 'int_sorting_buffer' ) is False and self.use_cuda:
            self.int_sorting_buffer = np.empty( self.Ntot, dtype=np.uint64 )

    def make_ionizable( self, element, target_species,
                        level_start=0, full_initialization=True ):
        """
        Make this species ionizable

        The implemented ionization model is the ADK model.
        See Chen, JCP 236 (2013), equation (2)

        Parameters
        ----------
        element: string
            The atomic symbol of the considered ionizable species
            (e.g. 'He', 'N' ;  do not use 'Helium' or 'Nitrogen')

        target_species: an fbpic.Particles object
            This object is not modified when creating the class, but
            it is modified when ionization occurs
            (i.e. more particles are created)

        level_start: int
            The ionization level at which the macroparticles are initially
            (e.g. 0 for initially neutral atoms)

        full_initialization: bool
            If True: initialize the parameters needed for the calculation
            of the ADK ionization rate. This is not needed when adding
            new particles to the same species (e.g. with the moving window).
        """
        # Initialize the ionizer module
        self.ionizer = Ionizer( element, self, target_species,
                                level_start, full_initialization )
        # Recalculate the weights to reflect the current ionization levels
        # (This is updated whenever further ionization happens)
        self.w[:] = e*self.ionizer.ionization_level*self.ionizer.neutral_weight

        # Update the number of float and int arrays
        self.n_float_quantities += 1 # neutral_weight
        self.n_integer_quantities += 1 # ionization_level
        # Allocate the integer sorting buffer if needed
        if hasattr( self, 'int_sorting_buffer' ) is False and self.use_cuda:
            self.int_sorting_buffer = np.empty( self.Ntot, dtype=np.uint64 )

    def handle_ionization( self ):
        """
        Ionize this species, and add new macroparticles to the target species
        """
        if self.ionizer is not None:
            if self.use_cuda:
                self.ionizer.handle_ionization_gpu( self )
            else:
                self.ionizer.handle_ionization_cpu( self )

    def rearrange_particle_arrays( self ):
        """
        Rearranges the particle data arrays to match with the sorted
        cell index array. The sorted index array is used to resort the
        arrays. A particle buffer is used to temporarily store
        the rearranged data.
        """
        # Get the threads per block and the blocks per grid
        dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( self.Ntot )
        # Iterate over (float) particle attributes
        attr_list = [ (self,'x'), (self,'y'), (self,'z'), \
                        (self,'ux'), (self,'uy'), (self,'uz'), \
                        (self, 'w'), (self,'inv_gamma') ]
        if self.ionizer is not None:
            attr_list += [ (self.ionizer,'neutral_weight') ]
        for attr in attr_list:
            # Get particle GPU array
            particle_array = getattr( attr[0], attr[1] )
            # Write particle data to particle buffer array while rearranging
            write_sorting_buffer[dim_grid_1d, dim_block_1d](
                self.sorted_idx, particle_array, self.sorting_buffer)
            # Assign the particle buffer to
            # the initial particle data array
            setattr( attr[0], attr[1], self.sorting_buffer)
            # Assign the old particle data array to the particle buffer
            self.sorting_buffer = particle_array
        # Iterate over (integer) particle attributes
        attr_list = [ ]
        if self.tracker is not None:
            attr_list += [ (self.tracker,'id') ]
        if self.ionizer is not None:
            attr_list += [ (self.ionizer,'ionization_level') ]
        for attr in attr_list:
            # Get particle GPU array
            particle_array = getattr( attr[0], attr[1] )
            # Write particle data to particle buffer array while rearranging
            write_sorting_buffer[dim_grid_1d, dim_block_1d](
                self.sorted_idx, particle_array, self.int_sorting_buffer)
            # Assign the particle buffer to
            # the initial particle data array
            setattr( attr[0], attr[1], self.int_sorting_buffer)
            # Assign the old particle data array to the particle buffer
            self.int_sorting_buffer = particle_array

    def push_p( self ) :
        """
        Advance the particles' momenta over one timestep, using the Vay pusher
        Reference : Vay, Physics of Plasmas 15, 056701 (2008)

        This assumes that the momenta (ux, uy, uz) are initially one
        half-timestep *behind* the positions (x, y, z), and it brings
        them one half-timestep *ahead* of the positions.
        """
        if self.use_cuda:
            # Get the threads per block and the blocks per grid
            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( self.Ntot )
            # Call the CUDA Kernel for the particle push
            if self.ionizer is None:
                push_p_gpu[dim_grid_1d, dim_block_1d](
                    self.ux, self.uy, self.uz, self.inv_gamma,
                    self.Ex, self.Ey, self.Ez,
                    self.Bx, self.By, self.Bz,
                    self.q, self.m, self.Ntot, self.dt )
            else:
                # Ionizable species can have a charge that depends on the
                # macroparticle, and hence require a different function
                push_p_ioniz_gpu[dim_grid_1d, dim_block_1d](
                    self.ux, self.uy, self.uz, self.inv_gamma,
                    self.Ex, self.Ey, self.Ez,
                    self.Bx, self.By, self.Bz,
                    self.m, self.Ntot, self.dt, self.ionizer.ionization_level )
        else :
            if self.ionizer is None:
                push_p_numba(self.ux, self.uy, self.uz, self.inv_gamma,
                    self.Ex, self.Ey, self.Ez, self.Bx, self.By, self.Bz,
                    self.q, self.m, self.Ntot, self.dt )
            else:
                # Ionizable species can have a charge that depends on the
                # macroparticle, and hence require a different function
                push_p_ioniz_numba(self.ux, self.uy, self.uz, self.inv_gamma,
                    self.Ex, self.Ey, self.Ez, self.Bx, self.By, self.Bz,
                    self.m, self.Ntot, self.dt, self.ionizer.ionization_level )

    def halfpush_x( self ) :
        """
        Advance the particles' positions over one half-timestep

        This assumes that the positions (x, y, z) are initially either
        one half-timestep *behind* the momenta (ux, uy, uz), or at the
        same timestep as the momenta.
        """
        if self.use_cuda:
            # Get the threads per block and the blocks per grid
            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( self.Ntot )
            # Call the CUDA Kernel for halfpush in x
            push_x_gpu[dim_grid_1d, dim_block_1d](
                self.x, self.y, self.z,
                self.ux, self.uy, self.uz,
                self.inv_gamma, self.dt )
            # The particle array is unsorted after the push in x
            self.sorted = False
        else :
            push_x_numba( self.x, self.y, self.z,
                self.ux, self.uy, self.uz,
                self.inv_gamma, self.Ntot, self.dt )

    def gather( self, grid ) :
        """
        Gather the fields onto the macroparticles

        This assumes that the particle positions are currently at
        the same timestep as the field that is to be gathered.

        Parameter
        ----------
        grid : a list of InterpolationGrid objects
             (one InterpolationGrid object per azimuthal mode)
             Contains the field values on the interpolation grid
        """
        if self.use_cuda == True:
            # Get the threads per block and the blocks per grid
            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( self.Ntot )
            # Call the CUDA Kernel for the gathering of E and B Fields
            # for Mode 0 and 1 only.
            if self.particle_shape == 'cubic':
                gather_field_gpu_cubic[dim_grid_1d, dim_block_1d](
                     self.x, self.y, self.z,
                     grid[0].invdz, grid[0].zmin, grid[0].Nz,
                     grid[0].invdr, grid[0].rmin, grid[0].Nr,
                     grid[0].Er, grid[0].Et, grid[0].Ez,
                     grid[1].Er, grid[1].Et, grid[1].Ez,
                     grid[0].Br, grid[0].Bt, grid[0].Bz,
                     grid[1].Br, grid[1].Bt, grid[1].Bz,
                     self.Ex, self.Ey, self.Ez,
                     self.Bx, self.By, self.Bz)
            else:
                gather_field_gpu_linear[dim_grid_1d, dim_block_1d](
                     self.x, self.y, self.z,
                     grid[0].invdz, grid[0].zmin, grid[0].Nz,
                     grid[0].invdr, grid[0].rmin, grid[0].Nr,
                     grid[0].Er, grid[0].Et, grid[0].Ez,
                     grid[1].Er, grid[1].Et, grid[1].Ez,
                     grid[0].Br, grid[0].Bt, grid[0].Bz,
                     grid[1].Br, grid[1].Bt, grid[1].Bz,
                     self.Ex, self.Ey, self.Ez,
                     self.Bx, self.By, self.Bz)
        else:
            # Preliminary arrays for the cylindrical conversion
            r = np.sqrt( self.x**2 + self.y**2 )
            # Avoid division by 0.
            invr = 1./np.where( r!=0., r, 1. )
            cos = np.where( r!=0., self.x*invr, 1. )
            sin = np.where( r!=0., self.y*invr, 0. )

            # Indices and weights
            if self.particle_shape == 'cubic':
                shape_order = 3
            else:
                shape_order = 1
            iz, Sz = weights(self.z, grid[0].invdz, grid[0].zmin, grid[0].Nz,
                             direction='z', shape_order=shape_order)
            ir, Sr = weights(r, grid[0].invdr, grid[0].rmin, grid[0].Nr,
                             direction='r', shape_order=shape_order)

            # Number of modes considered :
            # number of elements in the grid list
            Nm = len(grid)

            # -------------------------------
            # Gather the E field mode by mode
            # -------------------------------
            # Zero the previous fields
            self.Ex[:] = 0.
            self.Ey[:] = 0.
            self.Ez[:] = 0.
            # Prepare auxiliary matrices
            Ft = np.zeros(self.Ntot)
            Fr = np.zeros(self.Ntot)
            exptheta = np.ones(self.Ntot, dtype='complex')
            # exptheta takes the value exp(-im theta) throughout the loop
            for m in range(Nm) :
                # Increment exptheta (notice the - : backward transform)
                if m==1 :
                    exptheta[:].real = cos
                    exptheta[:].imag = -sin
                elif m>1 :
                    exptheta[:] = exptheta*( cos - 1.j*sin )
                # Gather the fields
                # (The sign with which the guards are added
                # depends on whether the fields should be zero on axis)
                gather_field_numba(
                    exptheta, m, grid[m].Er, Fr, iz, ir, Sz, Sr, -((-1.)**m))
                gather_field_numba(
                    exptheta, m, grid[m].Et, Ft, iz, ir, Sz, Sr, -((-1.)**m))
                gather_field_numba(
                    exptheta, m, grid[m].Ez, self.Ez, iz, ir, Sz, Sr, (-1.)**m)

            # Convert to Cartesian coordinates
            self.Ex[:] = cos*Fr - sin*Ft
            self.Ey[:] = sin*Fr + cos*Ft

            # -------------------------------
            # Gather the B field mode by mode
            # -------------------------------
            # Zero the previous fields
            self.Bx[:] = 0.
            self.By[:] = 0.
            self.Bz[:] = 0.
            # Prepare auxiliary matrices
            Ft[:] = 0.
            Fr[:] = 0.
            exptheta[:] = 1.
            # exptheta takes the value exp(-im theta) throughout the loop
            for m in range(Nm) :
                # Increment exptheta (notice the - : backward transform)
                if m==1 :
                    exptheta[:].real = cos
                    exptheta[:].imag = -sin
                elif m>1 :
                    exptheta[:] = exptheta*( cos - 1.j*sin )
                # Gather the fields
                # (The sign with which the guards are added
                # depends on whether the fields should be zero on axis)
                gather_field_numba(
                    exptheta, m, grid[m].Br, Fr, iz, ir, Sz, Sr, -((-1.)**m))
                gather_field_numba(
                    exptheta, m, grid[m].Bt, Ft, iz, ir, Sz, Sr, -((-1.)**m))
                gather_field_numba(
                    exptheta, m, grid[m].Bz, self.Bz, iz, ir, Sz, Sr, (-1.)**m)

            # Convert to Cartesian coordinates
            self.Bx[:] = cos*Fr - sin*Ft
            self.By[:] = sin*Fr + cos*Ft


    def deposit( self, fld, fieldtype ) :
        """
        Deposit the particles charge or current onto the grid

        This assumes that the particle positions (and momenta in the case of J)
        are currently at the same timestep as the field that is to be deposited

        Parameter
        ----------
        fld : a Field object
             Contains the list of InterpolationGrid objects with
             the field values as well as the prefix sum.

        fieldtype : string
             Indicates which field to deposit
             Either 'J' or 'rho'
        """
        # Shortcut for the list of InterpolationGrid objects
        grid = fld.interp

        if self.use_cuda == True:
            # Get the threads per block and the blocks per grid
            dim_grid_2d_flat, dim_block_2d_flat = cuda_tpb_bpg_1d(
                                                    grid[0].Nz*grid[0].Nr )
            dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d(
                                          grid[0].Nz, grid[0].Nr )

            # Create the helper arrays for deposition
            if self.particle_shape == 'linear_non_atomic':
                d_F0, d_F1, d_F2, d_F3 = cuda_deposition_arrays(
                    grid[0].Nz, grid[0].Nr, fieldtype=fieldtype)

            # Sort the particles
            if self.sorted is False:
                self.sort_particles(fld=fld)
                # The particles are now sorted and rearranged
                self.sorted = True

            # Call the CUDA Kernel for the deposition of rho or J
            # for Mode 0 and 1 only.
            # Rho
            if fieldtype == 'rho':
                # Deposit rho in each of four directions
                if self.particle_shape == 'linear_non_atomic':
                    deposit_rho_gpu[dim_grid_2d_flat, dim_block_2d_flat](
                        self.x, self.y, self.z, self.w,
                        grid[0].invdz, grid[0].zmin, grid[0].Nz,
                        grid[0].invdr, grid[0].rmin, grid[0].Nr,
                        d_F0, d_F1, d_F2, d_F3,
                        self.cell_idx, self.prefix_sum)
                    # Add the four directions together
                    add_rho[dim_grid_2d, dim_block_2d](
                        grid[0].rho, grid[1].rho,
                        d_F0, d_F1, d_F2, d_F3)
                elif self.particle_shape == 'cubic':
                    deposit_rho_gpu_cubic[dim_grid_2d_flat, dim_block_2d_flat](
                        self.x, self.y, self.z, self.w,
                        grid[0].invdz, grid[0].zmin, grid[0].Nz,
                        grid[0].invdr, grid[0].rmin, grid[0].Nr,
                        grid[0].rho, grid[1].rho,
                        self.cell_idx, self.prefix_sum)
                elif self.particle_shape == 'linear':
                    deposit_rho_gpu_linear[dim_grid_2d_flat, dim_block_2d_flat](
                        self.x, self.y, self.z, self.w,
                        grid[0].invdz, grid[0].zmin, grid[0].Nz,
                        grid[0].invdr, grid[0].rmin, grid[0].Nr,
                        grid[0].rho, grid[1].rho,
                        self.cell_idx, self.prefix_sum)
                else:
                    raise ValueError("`particle_shape` should be either 'linear', 'linear_atomic' \
                                      or 'cubic' but is `%s`" % self.particle_shape)
            # J
            elif fieldtype == 'J':
                # Deposit J in each of four directions
                if self.particle_shape == 'linear_non_atomic':
                    deposit_J_gpu[dim_grid_2d_flat, dim_block_2d_flat](
                        self.x, self.y, self.z, self.w,
                        self.ux, self.uy, self.uz, self.inv_gamma,
                        grid[0].invdz, grid[0].zmin, grid[0].Nz,
                        grid[0].invdr, grid[0].rmin, grid[0].Nr,
                        d_F0, d_F1, d_F2, d_F3,
                        self.cell_idx, self.prefix_sum)
                    # Add the four directions together
                    add_J[dim_grid_2d, dim_block_2d](
                        grid[0].Jr, grid[1].Jr,
                        grid[0].Jt, grid[1].Jt,
                        grid[0].Jz, grid[1].Jz,
                        d_F0, d_F1, d_F2, d_F3)
                elif self.particle_shape == 'cubic':
                    deposit_J_gpu_cubic[dim_grid_2d_flat, dim_block_2d_flat](
                        self.x, self.y, self.z, self.w,
                        self.ux, self.uy, self.uz, self.inv_gamma,
                        grid[0].invdz, grid[0].zmin, grid[0].Nz,
                        grid[0].invdr, grid[0].rmin, grid[0].Nr,
                        grid[0].Jr, grid[1].Jr,
                        grid[0].Jt, grid[1].Jt,
                        grid[0].Jz, grid[1].Jz,
                        self.cell_idx, self.prefix_sum)
                elif self.particle_shape == 'linear':
                    deposit_J_gpu_linear[dim_grid_2d_flat, dim_block_2d_flat](
                        self.x, self.y, self.z, self.w,
                        self.ux, self.uy, self.uz, self.inv_gamma,
                        grid[0].invdz, grid[0].zmin, grid[0].Nz,
                        grid[0].invdr, grid[0].rmin, grid[0].Nr,
                        grid[0].Jr, grid[1].Jr,
                        grid[0].Jt, grid[1].Jt,
                        grid[0].Jz, grid[1].Jz,
                        self.cell_idx, self.prefix_sum)
                else:
                    raise ValueError("`particle_shape` should be either \
                                      'linear', 'linear_atomic' or 'cubic' \
                                       but is `%s`" % self.particle_shape)
            else:
                raise ValueError("`fieldtype` should be either 'J' or \
                                  'rho', but is `%s`" % fieldtype)


        # CPU version
        else:
            # Preliminary arrays for the cylindrical conversion
            r = np.sqrt( self.x**2 + self.y**2 )
            # Avoid division by 0.
            invr = 1./np.where( r!=0., r, 1. )
            cos = np.where( r!=0., self.x*invr, 1. )
            sin = np.where( r!=0., self.y*invr, 0. )

            # Indices and weights
            if self.particle_shape == 'cubic':
                shape_order = 3
            else:
                shape_order = 1
            iz, Sz = weights(self.z, grid[0].invdz, grid[0].zmin, grid[0].Nz,
                             direction='z', shape_order=shape_order)
            ir, Sr = weights(r, grid[0].invdr, grid[0].rmin, grid[0].Nr,
                             direction='r', shape_order=shape_order)

            # Number of modes considered :
            # number of elements in the grid list
            Nm = len(grid)

            if fieldtype == 'rho' :
                # ---------------------------------------
                # Deposit the charge density mode by mode
                # ---------------------------------------
                # Prepare auxiliary matrix
                exptheta = np.ones( self.Ntot, dtype='complex')
                # exptheta takes the value exp(im theta) throughout the loop
                for m in range(Nm) :
                    # Increment exptheta (notice the + : forward transform)
                    if m==1 :
                        exptheta[:].real = cos
                        exptheta[:].imag = sin
                    elif m>1 :
                        exptheta[:] = exptheta*( cos + 1.j*sin )
                    # Deposit the fields
                    # (The sign -1 with which the guards are added is not
                    # trivial to derive but avoids artifacts on the axis)
                    deposit_field_numba(self.w*exptheta, grid[m].rho,
                                            iz, ir, Sz, Sr, -1.)

            elif fieldtype == 'J' :
                # ----------------------------------------
                # Deposit the current density mode by mode
                # ----------------------------------------
                # Calculate the currents
                Jr = self.w * c * self.inv_gamma*( cos*self.ux + sin*self.uy )
                Jt = self.w * c * self.inv_gamma*( cos*self.uy - sin*self.ux )
                Jz = self.w * c * self.inv_gamma*self.uz
                # Prepare auxiliary matrix
                exptheta = np.ones( self.Ntot, dtype='complex')
                # exptheta takes the value exp(im theta) throughout the loop
                for m in range(Nm) :
                    # Increment exptheta (notice the + : forward transform)
                    if m==1 :
                        exptheta[:].real = cos
                        exptheta[:].imag = sin
                    elif m>1 :
                        exptheta[:] = exptheta*( cos + 1.j*sin )
                    # Deposit the fields
                    # (The sign -1 with which the guards are added is not
                    # trivial to derive but avoids artifacts on the axis)
                    deposit_field_numba(Jr*exptheta, grid[m].Jr,
                                        iz, ir, Sz, Sr, -1.)
                    deposit_field_numba(Jt*exptheta, grid[m].Jt,
                                        iz, ir, Sz, Sr, -1.)
                    deposit_field_numba(Jz*exptheta, grid[m].Jz,
                                        iz, ir, Sz, Sr, -1.)

            else :
                raise ValueError(
        "`fieldtype` should be either 'J' or 'rho', but is `%s`" %fieldtype )

    def sort_particles(self, fld):
        """
        Sort the particles by performing the following steps:
        1. Get fied cell index
        2. Sort field cell index
        3. Parallel prefix sum
        4. Rearrange particle arrays

        Parameter
        ----------
        fld : a Field object
             Contains the list of InterpolationGrid objects with
             the field values as well as the prefix sum.
        """
        # Shortcut for interpolation grids
        grid = fld.interp
        # Get the threads per block and the blocks per grid
        dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( self.Ntot )
        dim_grid_2d_flat, dim_block_2d_flat = cuda_tpb_bpg_1d(
                                                grid[0].Nz*grid[0].Nr )
        # ------------------------
        # Sorting of the particles
        # ------------------------
        # Get the cell index of each particle
        # (defined by iz_lower and ir_lower)
        get_cell_idx_per_particle[dim_grid_1d, dim_block_1d](
            self.cell_idx,
            self.sorted_idx,
            self.x, self.y, self.z,
            grid[0].invdz, grid[0].zmin, grid[0].Nz,
            grid[0].invdr, grid[0].rmin, grid[0].Nr)
        # Sort the cell index array and modify the sorted_idx array
        # accordingly. The value of the sorted_idx array corresponds
        # to the index of the sorted particle in the other particle
        # arrays.
        sort_particles_per_cell(self.cell_idx, self.sorted_idx)
        # Reset the old prefix sum
        fld.prefix_sum_shift = 0
        reset_prefix_sum[dim_grid_2d_flat, dim_block_2d_flat](self.prefix_sum)
        # Perform the inclusive parallel prefix sum
        incl_prefix_sum[dim_grid_1d, dim_block_1d](
            self.cell_idx, self.prefix_sum)
        # Rearrange the particle arrays
        self.rearrange_particle_arrays()
