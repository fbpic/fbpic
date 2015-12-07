"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to implement the moving window.
"""
import numpy as np
from scipy.constants import c
from fbpic.particles import Particles
try:
    from fbpic.cuda_utils import cuda, cuda_tpb_bpg_2d, cuda_tpb_bpg_1d
    if cuda.is_available():
        cuda_installed = True
    else:
        cuda_installed = False
except ImportError:
    cuda_installed = False

class MovingWindow(object):
    """
    Class that contains the moving window's variables and methods

    One major problem of the moving window in a spectral code is that \
    the fields `wrap around` the moving window, .i.e the fields that
    disappear at the left end reappear at the right end, as a consequence
    of the periodicity of the Fourier transform.
    """
    
    def __init__( self, interp, comm, v=c, ux_m=0., uy_m=0., uz_m=0.,
                  ux_th=0., uy_th=0., uz_th=0. ):
        """
        Initializes a moving window object.

        Parameters
        ----------
        interp: a list of Interpolation object
            Contains the positions of the boundaries

        comm: a BoundaryCommunicator object
            Contains information about the MPI and about 
                    
        v: float (meters per seconds), optional
            The speed of the moving window

        ux_m, uy_m, uz_m: floats (dimensionless)
           Normalized mean momenta of the injected particles in each direction

        ux_th, uy_th, uz_th: floats (dimensionless)
           Normalized thermal momenta in each direction
        """
        # Momenta parameters
        self.ux_m = ux_m
        self.uy_m = uy_m
        self.uz_m = uz_m
        self.ux_th = ux_th
        self.uy_th = uy_th
        self.uz_th = uz_th
        
        # Attach moving window speed and period
        self.v = v
        self.exchange_period = comm.exchange_period

        # Attach reference position of moving window (only for the first proc)
        # (Determines by how many cells the window should be moves)
        if comm.rank == 0:
            self.zmin = interp[0].zmin
        
        # Attach injection position and speed (only for the last proc)
        if comm.rank == comm.size-1:
            ng = comm.n_guard
            self.z_inject = interp[0].zmax - ng*interp[0].dz
            self.z_end_plasma = interp[0].zmax - ng*interp[0].dz
            self.v_end_plasma = \
              c * uz_m / np.sqrt(1 + ux_m**2 + uy_m**2 + uz_m**2)

    def move_grids(self, fld, dt, mpi_comm):
        """
        Calculate by how many cells the moving window should be moved.
        If this is non-zero, shift the fields on the interpolation grid,
        and add new particles.

        NB: the spectral grid is not modified, as it is automatically
        updated after damping the fields (see main.py)
        
        Parameters
        ----------
        fld: a Fields object
            Contains the fields data of the simulation
    
        dt: float (in seconds)
            Timestep of the simulation

        mpi_comm: an mpi4py communicator
            This is typically the attribute `comm` of the BoundaryCommunicator
        """
        # To avoid discrepancies between processors, only the first proc
        # decides whether to send the data, and broadcasts the information.
        dz = fld.interp[0].dz
        if mpi_comm.rank==0:
            # Move the continuous position of the moving window object
            self.zmin += self.v * dt * self.exchange_period
            # Find the number of cells by which the window should move
            n_move = int( (self.zmin - fld.interp[0].zmin)/dz )
        else:
            n_move = None
        # Broadcast the information to all proc
        if mpi_comm.size > 0:
            n_move = mpi_comm.bcast( n_move )
    
        # Move the grids
        if n_move > 0:
            # Shift the fields
            Nm = len(fld.interp)
            for m in range(Nm):
                self.shift_interp_grid( fld.interp[m], n_move )

        # Because the grids have just been shifted, there is a shift
        # in the cell indices that are used for the prefix sum.
        if fld.use_cuda:
            fld.prefix_sum_shift = n_move * fld.Nr
            # This quantity is reset to 0 whenever d_prefix_sum is recalculated
                
        # Prepare the positions of injection for the particles
        # (The actual creation of particles is done when the routine
        # exchange_particles of boundary_communicator.py is called)
        if mpi_comm.rank == mpi_comm.size-1:
            # Move the injection position
            self.z_inject += self.v * dt * self.exchange_period
            # Take into account the motion of the end of the plasma
            self.z_end_plasma += self.v_end_plasma * dt * self.exchange_period

    ##         # Find the number of particle cells to add
    ##         n_inject = int( (self.z_inject - self.z_end_plasma)/dz )
    ##         # Add the new particle cells
    ##         if n_inject > 0:
    ##             for species in ptcl:
    ##                 if species.continuous_injection == True:
    ##                     if interp[0].use_cuda:
    ##                         # Add particles on the GPU
    ##                         add_particles_gpu( species, self.z_end_plasma,
    ##                             self.z_end_plasma + n_inject*dz, n_inject*p_nz,
    ##                             ux_m=self.ux_m, uy_m=self.uy_m, uz_m=self.uz_m,
    ##                             ux_th=self.ux_th, uy_th=self.uy_th,
    ##                             uz_th=self.uz_th)
    ##                     else:
    ##                         add_particles( species, self.z_end_plasma,
    ##                             self.z_end_plasma + n_inject*dz, n_inject*p_nz,
    ##                             ux_m=self.ux_m, uy_m=self.uy_m, uz_m=self.uz_m,
    ##                             ux_th=self.ux_th, uy_th=self.uy_th,
    ##                             uz_th=self.uz_th)
    ##         # Increment the position of the end of the plasma
    ##         self.z_end_plasma += n_inject*dz

    def shift_interp_grid( self, grid, n_move, shift_currents=False ):
        """
        Shift the interpolation grid by n_move cells. Shifting is done
        either on the CPU or the GPU, if use_cuda is True.
    
        Parameters
        ----------
        grid: an InterpolationGrid corresponding to one given azimuthal mode 
            Contains the values of the fields on the interpolation grid,
            and is modified by this function.

        n_move: int
            The number of cells by which the grid should be shifted

        shift_currents: bool, optional
            Whether to also shift the currents
            Default: False, since the currents are recalculated from
            scratch at each PIC cycle
        """
        # Modify the values of the corresponding z's 
        grid.z += n_move*grid.dz
        grid.zmin += n_move*grid.dz
        grid.zmax += n_move*grid.dz
        
        if grid.use_cuda:
            # Shift all the fields on the GPU
            grid.Er = self.shift_interp_field_gpu( grid.Er, n_move )
            grid.Et = self.shift_interp_field_gpu( grid.Et, n_move )
            grid.Ez = self.shift_interp_field_gpu( grid.Ez, n_move )
            grid.Br = self.shift_interp_field_gpu( grid.Br, n_move )
            grid.Bt = self.shift_interp_field_gpu( grid.Bt, n_move )
            grid.Bz = self.shift_interp_field_gpu( grid.Bz, n_move )
            if shift_currents:
                grid.Jr = self.shift_interp_field_gpu( grid.Jr, n_move )
                grid.Jt = self.shift_interp_field_gpu( grid.Jt, n_move )
                grid.Jz = self.shift_interp_field_gpu( grid.Jz, n_move )
                grid.rho = self.shift_interp_field_gpu( grid.rho, n_move )
        else:
            # Shift all the fields on the CPU
            self.shift_interp_field( grid.Er, n_move )
            self.shift_interp_field( grid.Et, n_move )
            self.shift_interp_field( grid.Ez, n_move )
            self.shift_interp_field( grid.Br, n_move )
            self.shift_interp_field( grid.Bt, n_move )
            self.shift_interp_field( grid.Bz, n_move )
            if shift_currents:
                self.shift_interp_field( grid.Jr, n_move )
                self.shift_interp_field( grid.Jt, n_move )
                self.shift_interp_field( grid.Jz, n_move )
                self.shift_interp_field( grid.rho, n_move )

    def shift_interp_field( self, field_array, n_move ):
        """
        Shift the field 'field_array' by n_move cells (backwards)
        
        Parameters
        ----------
        field_array: 2darray of complexs
            Contains the value of the fields, and is modified by
            this function

        n_move: int
            The number of cells by which the grid should be shifted
        """
        # Transfer the values to n_move cell before
        field_array[:-n_move,:] = field_array[n_move:,:]
        # Put the last cells to 0
        field_array[-n_move,:] = 0  

    def shift_interp_field_gpu( self, field_array, n_move):
        """
        Shift the field 'field_array' by n_move cells (backwards)
        on the GPU by applying a kernel that copies the shifted
        fields to a buffer array.
        
        Parameters
        ----------
        field_array: 2darray of complexs
            Contains the value of the fields, and is modified by
            this function

        n_move: int
            The number of cells by which the grid should be shifted

        Returns 
        -------
        The new shifted field array
        """
        # Get a 2D CUDA grid of the size of the grid
        dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d( 
            field_array.shape[0], field_array.shape[1] )
        # Initialize a field buffer to temporarily store the data
        field_buffer = cuda.device_array(
            (field_array.shape[0], field_array.shape[1]), dtype=np.complex128)
        # Shift the field array and copy it to the buffer
        shift_field_array_gpu[dim_grid_2d, dim_block_2d](
            field_array, field_buffer, n_move)
        # Assign the buffer to the original field array object
        field_array = field_buffer
        # Return the new shifted field array
        return( field_array )
            
# ---------------------------------------
# Utility functions for the moving window
# ---------------------------------------



def add_particles( species, zmin, zmax, Npz, ux_m=0., uy_m=0., uz_m=0.,
                  ux_th=0., uy_th=0., uz_th=0. ):
    """
    Create new particles between zmin and zmax, and add them to `species`

    Parameters
    ----------
    species: a Particles object
       Contains the particle data of that species

    zmin, zmax: floats (meters)
       The positions between which the new particles are created

    Npz: int
        The total number of particles to be added along the z axis
        (The number of particles along r and theta is the same as that of
        `species`)

    ux_m, uy_m, uz_m: floats (dimensionless)
        Normalized mean momenta of the injected particles in each direction

    ux_th, uy_th, uz_th: floats (dimensionless)
        Normalized thermal momenta in each direction     
    """
    # Create the particles that will be added
    new_ptcl = Particles( species.q, species.m, species.n,
        Npz, zmin, zmax, species.Npr, species.rmin, species.rmax,
        species.Nptheta, species.dt, species.dens_func,
        ux_m=ux_m, uy_m=uy_m, uz_m=uz_m,
        ux_th=ux_th, uy_th=uy_th, uz_th=uz_th )

    # Add the properties of these new particles to species object
    # Loop over the attributes of the species
    for key, attribute in vars(species).items():
        # Detect if it is an array
        if type(attribute) is np.ndarray:
            # Detect if it has one element per particle
            if attribute.shape == ( species.Ntot ,):
                # Concatenate the attribute of species and of new_ptcl
                new_attribute = np.hstack(
                    ( getattr(species, key), getattr(new_ptcl, key) )  )
                # Affect the resized array to the species object
                setattr( species, key, new_attribute )

    # Add the number of new particles to the global count of particles
    species.Ntot += new_ptcl.Ntot

def add_particles_gpu( species, zmin, zmax, Npz, ux_m=0., uy_m=0., uz_m=0.,
                  ux_th=0., uy_th=0., uz_th=0.):
    """
    Create new particles between zmin and zmax, and add them to `species`
    on the GPU.

    Parameters
    ----------
    species: a Particles object
       Contains the particle data of that species

    zmin, zmax: floats (meters)
       The positions between which the new particles are created

    Npz: int
        The total number of particles to be added along the z axis
        (The number of particles along r and theta is the same as that of
        `species`)

    ux_m, uy_m, uz_m: floats (dimensionless)
        Normalized mean momenta of the injected particles in each direction

    ux_th, uy_th, uz_th: floats (dimensionless)
        Normalized thermal momenta in each direction     
    """
    # Create the particles that will be added
    new_ptcl = Particles( species.q, species.m, species.n,
        Npz, zmin, zmax, species.Npr, species.rmin, species.rmax,
        species.Nptheta, species.dt, species.dens_func,
        ux_m=ux_m, uy_m=uy_m, uz_m=uz_m,
        ux_th=ux_th, uy_th=uy_th, uz_th=uz_th)
    # Calculate new total number of particles
    new_Ntot = species.Ntot + new_ptcl.Ntot
    # Get the threads per block and the blocks per grid
    dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( new_Ntot )
    # Iterate over particle attributes
    for attr in ['x', 'y', 'z', 'ux', 'uy', 'uz', 'w', 'inv_gamma']:
        # Initialize buffer array
        particle_buffer = cuda.device_array(new_Ntot, dtype=np.float64)
        # Get particle GPU array
        particle_array = getattr(species, attr)
        new_particle_array = getattr(new_ptcl, attr)
        # Add particle data to particle buffer array
        add_particle_data_gpu[dim_grid_1d, dim_block_1d](
            particle_array, new_particle_array, particle_buffer)
        # Assign the particle buffer to 
        # the initial particle data array 
        setattr(species, attr, particle_buffer)

    # Initialize empty arrays on the CPU for the field
    # gathering and the particle push
    species.Ex = np.zeros(new_Ntot, dtype = np.float64)
    species.Ey = np.zeros(new_Ntot, dtype = np.float64)
    species.Ez = np.zeros(new_Ntot, dtype = np.float64)
    species.Bx = np.zeros(new_Ntot, dtype = np.float64)
    species.By = np.zeros(new_Ntot, dtype = np.float64)
    species.Bz = np.zeros(new_Ntot, dtype = np.float64)
    # Initialize empty arrays on the CPU
    # that represent the sorting arrays
    species.cell_idx = np.empty(new_Ntot, dtype = np.int32)
    species.sorted_idx = np.arange(new_Ntot, dtype = np.uint32)
    species.particle_buffer = np.arange(new_Ntot, dtype = np.float64)
    # Initialize empty arrays on the GPU for the field
    # gathering and the particle push
    species.Ex = cuda.device_array_like(species.Ex)
    species.Ey = cuda.device_array_like(species.Ey)
    species.Ez = cuda.device_array_like(species.Ez)
    species.Bx = cuda.device_array_like(species.Bx)
    species.By = cuda.device_array_like(species.By)
    species.Bz = cuda.device_array_like(species.Bz)
    # Initialize empty arrays on the GPU for the sorting
    species.cell_idx = cuda.device_array_like(species.cell_idx)
    species.sorted_idx = cuda.device_array_like(species.sorted_idx)
    species.particle_buffer = cuda.device_array_like(species.particle_buffer)
    # Change the new total number of particles    
    species.Ntot = new_Ntot
    # The particles are unsorted after adding new particles.
    species.sorted = False

if cuda_installed:

    @cuda.jit('void(complex128[:,:], complex128[:,:], int32)')
    def shift_field_array_gpu( field_array, field_buffer, n_move ):
        """
        Shift a field array by reading the values from the field_array
        and writing them to the field_buffer on the GPU.

        Parameters:
        ------------
        field_array, field_buffer: 2darrays of complexs
            Contains the unshifted field (field_array)
            Contains the shifted field (field_buffer) afterwards

        n_move: int
            Amount of cells by which the field array should be 
            shifted in the longitudinal direction.
        """
        # Get a 2D CUDA grid
        i, j = cuda.grid(2)
        # Shift the values of the field array and copy them to the buffer
        if (i + n_move) < field_array.shape[0] and j < field_array.shape[1]:
            field_buffer[i, j] = field_array[i+n_move, j]
        # Set the remaining values to zero
        if (i + n_move) >= field_array.shape[0] and i < field_array.shape[0] \
          and j < field_array.shape[1]:
            field_buffer[i, j] = 0.



    @cuda.jit('void(float64[:], float64[:], float64[:])')
    def add_particle_data_gpu( particle_array, new_particle_array, particle_buffer ):
        """
        Add new particles by combining the old and the added particle data to a 
        new buffer array that then contains the new particles.

        Parameters:
        ------------
        particle_array, new_particle_array, particle_buffer: 1darrays of floats
            Contains the old particles (particle_array)
            Contains the added particles (new_particle_array)
            Contains the new particles (particle_buffer) afterwards
        """
        # Get a 1D CUDA grid
        i = cuda.grid(1)
        # Copy the old particle data to the buffer
        if i < particle_array.shape[0]:
            particle_buffer[i] = particle_array[i]
        # Copy the added particle data to the buffer
        if (i >= particle_array.shape[0]) and (i < (particle_buffer.shape[0])):
            particle_buffer[i] = new_particle_array[i-particle_array.shape[0]]
