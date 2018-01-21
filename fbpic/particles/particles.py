# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure and methods associated with the particles.
"""
import numpy as np
import numba
from scipy.constants import e
from .tracking import ParticleTracker
from .elementary_process.ionization import Ionizer
from .elementary_process.compton import ComptonScatterer
# Load the utility methods
from .utilities.utility_methods import unalign_angles
# Load the numba methods
from .push.numba_methods import push_p_numba, push_p_ioniz_numba, push_x_numba
from .gathering.threading_methods import gather_field_numba_linear, \
        gather_field_numba_cubic
from .deposition.threading_methods import deposit_rho_numba_linear, \
        deposit_J_numba_linear, deposit_rho_numba_cubic, \
        deposit_J_numba_cubic, sum_reduce_2d_array

# Check if threading is enabled
from fbpic.utils.threading import threading_enabled, get_chunk_indices
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    # Load the CUDA methods
    from fbpic.utils.cuda import cuda, cuda_tpb_bpg_1d
    from .push.cuda_methods import push_p_gpu, push_p_ioniz_gpu, push_x_gpu
    from .deposition.cuda_methods import deposit_rho_gpu_linear, \
        deposit_J_gpu_linear, deposit_rho_gpu_cubic, deposit_J_gpu_cubic
    from .gathering.cuda_methods import gather_field_gpu_linear, \
        gather_field_gpu_cubic
    from .utilities.cuda_sorting import write_sorting_buffer, \
        get_cell_idx_per_particle, sort_particles_per_cell, \
        prefill_prefix_sum, incl_prefix_sum

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
                    grid_shape=None, particle_shape='linear',
                    use_cuda=False ) :
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

        grid_shape: tuple, optional
            Needed when running on the GPU
            The shape of the local grid (including guard cells), i.e.
            a tuple of the form (Nz, Nr). This is needed in order
            to initialize the sorting of the particles per cell.

        particle_shape: str, optional
            Set the particle shape for the charge/current deposition.
            Possible values are 'linear' and 'cubic' for first and third
            order particle shape factors.

        use_cuda : bool, optional
            Wether to use the GPU or not.
        """
        # Register the timestep
        self.dt = dt

        # Define whether or not to use the GPU
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
        # By default, the species experiences no elementary processes
        # (see method make_ionizable and activate_compton)
        self.ionizer = None
        self.compton_scatterer = None
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
            zp, rp, thetap = np.meshgrid( z_reg, r_reg, theta_reg,
                                        copy=True, indexing='ij' )
            # Prevent the particles from being aligned along any direction
            unalign_angles( thetap, Npz, Npr, method='random' )
            # Flatten them (This performs a memory copy)
            r = rp.flatten()
            self.x[:] = r * np.cos( thetap.flatten() )
            self.y[:] = r * np.sin( thetap.flatten() )
            self.z[:] = zp.flatten()
            # Get the weights (i.e. charge of each macroparticle), which
            # are equal to the density times the volume r d\theta dr dz
            self.w[:] = n * r * dtheta*dr*dz
            # Modulate it by the density profile
            if dens_func is not None :
                self.w[:] = self.w * dens_func( self.z, r )

        # Register particle shape
        self.particle_shape = particle_shape

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
            Nz, Nr = grid_shape
            self.prefix_sum = np.empty( Nz*(Nr+1), dtype=np.int32 )
            # Register boolean that records if the particles are sorted or not
            self.sorted = False

        # Register number of threads
        if threading_enabled:
            self.nthreads = numba.config.NUMBA_NUM_THREADS
        else:
            self.nthreads = 1

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

    def activate_compton( self, target_species, laser_energy, laser_wavelength,
        laser_waist, laser_ctau, laser_initial_z0, ratio_w_electron_photon=1,
        boost=None ):
        """
        Activate Compton scattering.

        This considers a counterpropagating Gaussian laser pulse (which is not
        represented on the grid, for compatibility with the boosted-frame,
        but is instead assumed to propagate rigidly along the z axis).
        Interaction between this laser and the current species results
        in the generation of photons, according to the Klein-Nishina formula.

        See the docstring of the class `ComptonScatterer` for more information
        on the physical model used, and its domain of validity.

        The API of this function is not stable, and may change in the future.

        Parameters:
        -----------
        target_species: a `Particles` object
            The photons species, to which new macroparticles will be added.

        laser_energy: float (in Joules)
            The energy of the counterpropagating laser pulse (in the lab frame)

        laser_wavelength: float (in meters)
            The wavelength of the laser pulse (in the lab frame)

        laser_waist, laser_ctau: floats (in meters)
            The waist and duration of the laser pulse (in the lab frame)
            Both defined as the distance, from the laser peak, where
            the *field* envelope reaches 1/e of its peak value.

        laser_initial_z0: float (in meters)
            The initial position of the laser pulse (in the lab frame)

        ratio_w_electron_photon: float
            The ratio of the weight of an electron macroparticle to the
            weight of the photon macroparticles that it will emit.
            Increasing this ratio increases the number of photon macroparticles
            that will be emitted and therefore improves statistics.
        """
        self.compton_scatterer = ComptonScatterer(
            self, target_species, laser_energy, laser_wavelength,
            laser_waist, laser_ctau, laser_initial_z0,
            ratio_w_electron_photon, boost )


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
        # Set charge to the elementary charge e (assumed by deposition kernel,
        # when using self.ionizer.w_times_level as the effective weight)
        self.q = e

        # Update the number of float and int arrays
        self.n_float_quantities += 1 # w_times_level
        self.n_integer_quantities += 1 # ionization_level
        # Allocate the integer sorting buffer if needed
        if hasattr( self, 'int_sorting_buffer' ) is False and self.use_cuda:
            self.int_sorting_buffer = np.empty( self.Ntot, dtype=np.uint64 )


    def handle_elementary_processes( self, t ):
        """
        Handle elementary processes for this species (e.g. ionization,
        Compton scattering) at simulation time t.
        """
        # Ionization
        if self.ionizer is not None:
            self.ionizer.handle_ionization( self )
        # Compton scattering
        if self.compton_scatterer is not None:
            self.compton_scatterer.handle_scattering( self, t )


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
            attr_list += [ (self.ionizer,'w_times_level') ]
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
        # Skip push for neutral particles (e.g. photons)
        if self.q == 0:
            return

        # GPU (CUDA) version
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
        # CPU version
        else:
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
        # GPU (CUDA) version
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
        # CPU version
        else:
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
        # Skip gathering for neutral particles (e.g. photons)
        if self.q == 0:
            return

        # GPU (CUDA) version
        if self.use_cuda:
            # Get the threads per block and the blocks per grid
            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( self.Ntot, TPB=64 )
            # Call the CUDA Kernel for the gathering of E and B Fields
            # for Mode 0 and 1 only.
            if self.particle_shape == 'linear':
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
            elif self.particle_shape == 'cubic':
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
                raise ValueError("`particle_shape` should be either \
                                  'linear' or 'cubic' \
                                   but is `%s`" % self.particle_shape)
        # CPU version
        else:
            if self.particle_shape == 'linear':
                gather_field_numba_linear(
                     self.x, self.y, self.z,
                     grid[0].invdz, grid[0].zmin, grid[0].Nz,
                     grid[0].invdr, grid[0].rmin, grid[0].Nr,
                     grid[0].Er, grid[0].Et, grid[0].Ez,
                     grid[1].Er, grid[1].Et, grid[1].Ez,
                     grid[0].Br, grid[0].Bt, grid[0].Bz,
                     grid[1].Br, grid[1].Bt, grid[1].Bz,
                     self.Ex, self.Ey, self.Ez,
                     self.Bx, self.By, self.Bz)
            elif self.particle_shape == 'cubic':
                # Divide particles into chunks (each chunk is handled by a
                # different thread) and return the indices that bound chunks
                ptcl_chunk_indices = get_chunk_indices(self.Ntot, self.nthreads)
                gather_field_numba_cubic(
                     self.x, self.y, self.z,
                     grid[0].invdz, grid[0].zmin, grid[0].Nz,
                     grid[0].invdr, grid[0].rmin, grid[0].Nr,
                     grid[0].Er, grid[0].Et, grid[0].Ez,
                     grid[1].Er, grid[1].Et, grid[1].Ez,
                     grid[0].Br, grid[0].Bt, grid[0].Bz,
                     grid[1].Br, grid[1].Bt, grid[1].Bz,
                     self.Ex, self.Ey, self.Ez,
                     self.Bx, self.By, self.Bz,
                     self.nthreads, ptcl_chunk_indices )
            else:
                raise ValueError("`particle_shape` should be either \
                                  'linear' or 'cubic' \
                                   but is `%s`" % self.particle_shape)

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
        # Skip deposition for neutral particles (e.g. photons)
        if self.q == 0:
            return

        # Shortcuts for the list of InterpolationGrid objects
        grid = fld.interp

        # When running on GPU: first sort the arrays of particles
        if self.use_cuda:
            # Sort the particles
            if not self.sorted:
                self.sort_particles(fld=fld)
                # The particles are now sorted and rearranged
                self.sorted = True

        # For ionizable atoms: set the effective weight to the weight
        # times the ionization level (on GPU, this needs to be done *after*
        # sorting, otherwise `weight` is not equal to the corresponding array)
        if self.ionizer is not None:
            weight = self.ionizer.w_times_level
        else:
            weight = self.w

        # GPU (CUDA) version
        if self.use_cuda:
            # Get the threads per block and the blocks per grid
            dim_grid_2d_flat, dim_block_2d_flat = \
                cuda_tpb_bpg_1d( self.prefix_sum.shape[0], TPB=64 )

            # Call the CUDA Kernel for the deposition of rho or J
            # for Mode 0 and 1 only.
            # Rho
            if fieldtype == 'rho':
                # Deposit rho in each of four directions
                if self.particle_shape == 'linear':
                    deposit_rho_gpu_linear[dim_grid_2d_flat, dim_block_2d_flat](
                        self.x, self.y, self.z, weight, self.q,
                        grid[0].invdz, grid[0].zmin, grid[0].Nz,
                        grid[0].invdr, grid[0].rmin, grid[0].Nr,
                        grid[0].rho, grid[1].rho,
                        self.cell_idx, self.prefix_sum)
                elif self.particle_shape == 'cubic':
                    deposit_rho_gpu_cubic[dim_grid_2d_flat, dim_block_2d_flat](
                        self.x, self.y, self.z, weight, self.q,
                        grid[0].invdz, grid[0].zmin, grid[0].Nz,
                        grid[0].invdr, grid[0].rmin, grid[0].Nr,
                        grid[0].rho, grid[1].rho,
                        self.cell_idx, self.prefix_sum)
                else:
                    raise ValueError("`particle_shape` should be either \
                                      'linear' or 'cubic' \
                                       but is `%s`" % self.particle_shape)
            # J
            elif fieldtype == 'J':
                # Deposit J in each of four directions
                if self.particle_shape == 'linear':
                    deposit_J_gpu_linear[dim_grid_2d_flat, dim_block_2d_flat](
                        self.x, self.y, self.z, weight, self.q,
                        self.ux, self.uy, self.uz, self.inv_gamma,
                        grid[0].invdz, grid[0].zmin, grid[0].Nz,
                        grid[0].invdr, grid[0].rmin, grid[0].Nr,
                        grid[0].Jr, grid[1].Jr,
                        grid[0].Jt, grid[1].Jt,
                        grid[0].Jz, grid[1].Jz,
                        self.cell_idx, self.prefix_sum)
                elif self.particle_shape == 'cubic':
                    deposit_J_gpu_cubic[dim_grid_2d_flat, dim_block_2d_flat](
                        self.x, self.y, self.z, weight, self.q,
                        self.ux, self.uy, self.uz, self.inv_gamma,
                        grid[0].invdz, grid[0].zmin, grid[0].Nz,
                        grid[0].invdr, grid[0].rmin, grid[0].Nr,
                        grid[0].Jr, grid[1].Jr,
                        grid[0].Jt, grid[1].Jt,
                        grid[0].Jz, grid[1].Jz,
                        self.cell_idx, self.prefix_sum)
                else:
                    raise ValueError("`particle_shape` should be either \
                                      'linear' or 'cubic' \
                                       but is `%s`" % self.particle_shape)
            else:
                raise ValueError("`fieldtype` should be either 'J' or \
                                  'rho', but is `%s`" % fieldtype)

        # CPU version
        else:
            # Divide particles in chunks (each chunk is handled by a different
            # thread) and register the indices that bound each chunks
            ptcl_chunk_indices = get_chunk_indices(self.Ntot, self.nthreads)

            # Multithreading functions for the deposition of rho or J
            # for Mode 0 and 1 only.
            if fieldtype == 'rho':
                # Generate temporary arrays for rho
                # (2 guard cells on each side in z and r, in order to store
                # contributions from, at most, cubic shape factors ; these
                # deposition guard cells are folded into the regular box
                # inside `sum_reduce_2d_array`)
                rho_global = np.zeros( dtype=np.complex128,
                    shape=(self.nthreads, fld.Nm, fld.Nz+4, fld.Nr+4) )
                # Deposit rho using CPU threading
                if self.particle_shape == 'linear':
                    deposit_rho_numba_linear(
                        self.x, self.y, self.z, weight, self.q,
                        grid[0].invdz, grid[0].zmin, grid[0].Nz,
                        grid[0].invdr, grid[0].rmin, grid[0].Nr,
                        rho_global, fld.Nm,
                        self.nthreads, ptcl_chunk_indices )
                elif self.particle_shape == 'cubic':
                    deposit_rho_numba_cubic(
                        self.x, self.y, self.z, weight, self.q,
                        grid[0].invdz, grid[0].zmin, grid[0].Nz,
                        grid[0].invdr, grid[0].rmin, grid[0].Nr,
                        rho_global, fld.Nm,
                        self.nthreads, ptcl_chunk_indices )
                else:
                    raise ValueError("`particle_shape` should be either \
                                      'linear' or 'cubic' \
                                       but is `%s`" % self.particle_shape)
                # Sum thread-local results to main field array
                for m in range(fld.Nm):
                    sum_reduce_2d_array( rho_global, grid[m].rho, m )

            elif fieldtype == 'J':
                # Generate temporary arrays for J
                # (2 guard cells on each side in z and r, in order to store
                # contributions from, at most, cubic shape factors ; these
                # deposition guard cells are folded into the regular box
                # inside `sum_reduce_2d_array`)
                Jr_global = np.zeros( dtype=np.complex128,
                    shape=(self.nthreads, fld.Nm, fld.Nz+4, fld.Nr+4) )
                Jt_global = np.zeros( dtype=np.complex128,
                    shape=(self.nthreads, fld.Nm, fld.Nz+4, fld.Nr+4) )
                Jz_global = np.zeros( dtype=np.complex128,
                    shape=(self.nthreads, fld.Nm, fld.Nz+4, fld.Nr+4) )
                # Deposit J using CPU threading
                if self.particle_shape == 'linear':
                    deposit_J_numba_linear(
                        self.x, self.y, self.z, weight, self.q,
                        self.ux, self.uy, self.uz, self.inv_gamma,
                        grid[0].invdz, grid[0].zmin, grid[0].Nz,
                        grid[0].invdr, grid[0].rmin, grid[0].Nr,
                        Jr_global, Jt_global, Jz_global, fld.Nm,
                        self.nthreads, ptcl_chunk_indices )
                elif self.particle_shape == 'cubic':
                    deposit_J_numba_cubic(
                        self.x, self.y, self.z, weight, self.q,
                        self.ux, self.uy, self.uz, self.inv_gamma,
                        grid[0].invdz, grid[0].zmin, grid[0].Nz,
                        grid[0].invdr, grid[0].rmin, grid[0].Nr,
                        Jr_global, Jt_global, Jz_global, fld.Nm,
                        self.nthreads, ptcl_chunk_indices )
                else:
                    raise ValueError("`particle_shape` should be either \
                                      'linear' or 'cubic' \
                                       but is `%s`" % self.particle_shape)
                # Sum thread-local results to main field array
                for m in range(fld.Nm):
                    sum_reduce_2d_array( Jr_global, grid[m].Jr, m )
                    sum_reduce_2d_array( Jt_global, grid[m].Jt, m )
                    sum_reduce_2d_array( Jz_global, grid[m].Jz, m )
            else:
                raise ValueError("`fieldtype` should be either 'J' or \
                                  'rho', but is `%s`" % fieldtype)

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
        dim_grid_2d_flat, dim_block_2d_flat = \
                cuda_tpb_bpg_1d( self.prefix_sum.shape[0] )

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
        prefill_prefix_sum[dim_grid_2d_flat, dim_block_2d_flat](
            self.cell_idx, self.prefix_sum, self.Ntot )
        # Perform the inclusive parallel prefix sum
        incl_prefix_sum[dim_grid_1d, dim_block_1d](
            self.cell_idx, self.prefix_sum)
        # Rearrange the particle arrays
        self.rearrange_particle_arrays()
