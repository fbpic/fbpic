"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to handle mpi buffers for the particles
"""
import numpy as np
try:
    from fbpic.cuda_utils import cuda, cuda_tpb_bpg_1d
    if cuda.is_available():
        cuda_installed = True
    else:
        cuda_installed = False
except ImportError:
    cuda_installed = False

def remove_outside_particles(species, fld, nguard, left_proc, right_proc):
    """
    Remove the particles that are outside of the physical domain (i.e.
    in the guard cells). Store them in sending buffers, which are returned.

    Parameters
    ----------
    species: a Particles object
        Contains the data of this species

    fld: a Fields object
        Contains information about the dimension of the grid,
        and the prefix sum (when using the GPU)

    nguard: int
        Number of guard cells
        
    left_proc, right_proc: int or None
        Indicate whether there is a left or right processor or if the
        boundary is open (None). When the boundary is open, some time can be
        saved by returning an be empty, since it is not used anyway.

    Returns
    -------
    Two arrays of shape (8,Nptcl) where Nptcl is the number of particles
    that are sent to the left proc and right proc respectively.
    If left_proc or right_proc is None, the corresponding array has Nptcl=0
    """
    if species.use_cuda:
        # Remove outside particles on GPU, and copy buffers on CPU
        send_left, send_right = remove_particles_gpu( species, fld,
                                    nguard, left_proc, right_proc )
    else:
        # Remove outside particles on the CPU
        zbox_min = fld.interp[0].zmin + nguard*fld.interp[0].dz
        zbox_max = fld.interp[0].zmax - nguard*fld.interp[0].dz
        send_left, send_right = remove_particles_cpu( species,
                    zbox_min, zbox_max, left_proc, right_proc )

    return( send_left, send_right )
        
def remove_particles_cpu(species, zbox_min, zbox_max, left_proc, right_proc):
    """
    Removes the particles that are below `zbox_min` or beyond `zbox_max`
    from the particles arrays. Store them in sending buffers.

    Parameters
    ----------
    species: a Particles object
        Contains the data of this species

    zbox_min, zbox_max: float
        The lower and upper boundary of the physical box
        (i.e. does not include the guard cells)

    left_proc, right_proc: int or None
        Indicate whether there is a left or right processor or if the
        boundary is open (None). When the boundary is open, some time can be
        saved by returning an be empty, since it is not used anyway.
    """
    # Select the particles that are in the left or right guard cells,
    # and those that stay on the local process
    selec_left = ( species.z < zbox_min )
    selec_right = ( species.z > zbox_max )
    selec_stay = (np.logical_not(selec_left) & np.logical_not(selec_right))
        
    # Allocate and fill left sending buffer
    if left_proc is not None:
        N_send_l = selec_left.sum()
        send_left = np.empty((8, N_send_l), dtype = np.float64)
        send_left[0,:] = species.x[selec_left]
        send_left[1,:] = species.y[selec_left]
        send_left[2,:] = species.z[selec_left]
        send_left[3,:] = species.ux[selec_left]
        send_left[4,:] = species.uy[selec_left]
        send_left[5,:] = species.uz[selec_left]
        send_left[6,:] = species.inv_gamma[selec_left]
        send_left[7,:] = species.w[selec_left]
    else:
        # No need to allocate and copy data ; return an empty array
        send_left = np.empty((8, 0), dtype = np.float64)

    # Allocate and fill right sending buffer
    if right_proc is not None:
        N_send_r = selec_right.sum()
        send_right = np.empty((8, N_send_r), dtype = np.float64)
        send_right[0,:] = species.x[selec_right]
        send_right[1,:] = species.y[selec_right]
        send_right[2,:] = species.z[selec_right]
        send_right[3,:] = species.ux[selec_right]
        send_right[4,:] = species.uy[selec_right]
        send_right[5,:] = species.uz[selec_right]
        send_right[6,:] = species.inv_gamma[selec_right]
        send_right[7,:] = species.w[selec_right]
    else:
        # No need to allocate and copy data ; return an empty array
        send_right = np.empty((8, 0), dtype = np.float64)

    # Resize the particle arrays
    N_stay = selec_stay.sum()
    species.Ntot = N_stay
    species.x = species.x[selec_stay]
    species.y = species.y[selec_stay]
    species.z = species.z[selec_stay]
    species.ux = species.ux[selec_stay]
    species.uy = species.uy[selec_stay]
    species.uz = species.uz[selec_stay]
    species.inv_gamma = species.inv_gamma[selec_stay]
    species.w = species.w[selec_stay]
        
    # Return the sending buffers
    return( send_left, send_right )

def remove_particles_gpu(species, fld, nguard, left_proc, right_proc):
    """
    Removes the particles that are below `zbox_min` or beyond `zbox_max`
    from the particles arrays. Store them in sending buffers on the CPU.

    Parameters
    ----------
    species: a Particles object
        Contains the data of this species

    fld: a Fields object
        Contains information about the dimension of the grid,
        and the prefix sum (when using the GPU)

    nguard: int
        Number of guard cells
        
    left_proc, right_proc: int or None
        Indicate whether there is a left or right processor or if the
        boundary is open (None). When the boundary is open, some time can be
        saved by returning an be empty, since it is not used anyway.
    """
    # Check if particles are sorted, otherwise raise exception
    if species.sorted == False:
        raise ValueError('Removing particles: The particles are not sorted!')

    # Get the particle indices between which to remove the particles
    prefix_sum = fld.d_prefix_sum
    i_min = prefix_sum.getitem(nguard*fld.Nr + fld.prefix_sum_shift)
    i_max = prefix_sum.getitem(nguard*fld.Nr - fld.prefix_sum_shift)

    # Total number of particles in each particle group
    N_send_l = i_min
    new_Ntot = i_max - i_min
    N_send_r = species.Ntot - i_max

    # Allocate the sending buffers on the CPU
    if left_proc is not None:
        send_left = np.empty((8, N_send_l), dtype = np.float64)
    else:
        send_left = np.empty((8, 0), dtype = np.float64)
    if right_proc is not None:
        send_right = np.empty((8, N_send_r), dtype = np.float64)
    else:
        send_right = np.empty((8, 0), dtype = np.float64)
    
    # Get the threads per block and the blocks per grid
    dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( species.Ntot )
    # Iterate over particle attributes
    i_attr = 0
    for attr in ['x', 'y', 'z', 'ux', 'uy', 'uz', 'w', 'inv_gamma']:
        # Initialize 3 buffer arrays on the GPU
        left_buffer = cuda.device_array(N_send_l, dtype=np.float64)
        right_buffer = cuda.device_array(N_send_r, dtype=np.float64)
        stay_buffer = cuda.device_array(new_Ntot, dtype=np.float64)
        # Split the particle array into the 3 buffers on the GPU
        particle_array = getattr(species, attr)
        split_particles_to_buffers[dim_grid_1d, dim_block_1d](
            particle_array, left_buffer, stay_buffer, right_buffer)
        # Assign the stay_buffer to the initial particle data array
        # and fill the sending buffers (if needed for MPI)
        setattr(species, attr, stay_buffer)
        if left_proc is not None:
            left_buffer.copy_to_host( send_left[i_attr] )
        if right_proc is not None:
            right_buffer.copy_to_host( send_right[i_attr] )
        # Increment the buffer index
        i_attr += 1

    # Change the new total number of particles    
    species.Ntot = new_Ntot


def add_buffers_to_particles( species, recv_left, recv_right ):
    """
    DOCUMENTATION
    """
    # Copy the buffers to an enlarged array
    if species.use_cuda:
        add_buffers_gpu( species, recv_left, recv_right )
    else:
        add_buffers_cpu( species, recv_left, recv_right )        
    
    # Reallocate the particles field arrays. This needs to be done,
    # as the total number of particles in this domain has changed.
    species.Ex = np.empty(species.Ntot, dtype = np.float64)
    species.Ey = np.empty(species.Ntot, dtype = np.float64)
    species.Ez = np.empty(species.Ntot, dtype = np.float64)
    species.Bx = np.empty(species.Ntot, dtype = np.float64)
    species.By = np.empty(species.Ntot, dtype = np.float64)
    species.Bz = np.empty(species.Ntot, dtype = np.float64)
    if species.use_cuda:
        # Initialize empty arrays on the GPU for the field
        # gathering and the particle push
        species.Ex = cuda.device_array_like(species.Ex)
        species.Ey = cuda.device_array_like(species.Ey)
        species.Ez = cuda.device_array_like(species.Ez)
        species.Bx = cuda.device_array_like(species.Bx)
        species.By = cuda.device_array_like(species.By)
        species.Bz = cuda.device_array_like(species.Bz)
    
    # Reallocate the cell index and sorted index arrays on the CPU
    species.cell_idx = np.empty(species.Ntot, dtype = np.int32)
    species.sorted_idx = np.arange(species.Ntot, dtype = np.uint32)
    species.particle_buffer = np.arange(species.Ntot, dtype = np.float64)
    # The particles are unsorted after adding new particles.
    species.sorted = False

def add_buffers_cpu( species, recv_left, recv_right ):
    """
    DOCUMENTATION
    """    
    # Form the new particle arrays by adding the received particles
    # from the left and the right to the particles that stay in the domain
    species.x = np.hstack((recv_left[0], species.x, recv_right[0]))
    species.y = np.hstack((recv_left[1], species.y, recv_right[1]))
    species.z = np.hstack((recv_left[2], species.z, recv_right[2]))
    species.ux = np.hstack((recv_left[3], species.ux, recv_right[3]))
    species.uy = np.hstack((recv_left[4], species.uy, recv_right[4]))
    species.uz = np.hstack((recv_left[5], species.uz, recv_right[5]))
    species.inv_gamma = \
        np.hstack((recv_left[6], species.inv_gamma, recv_right[6]))
    species.w = np.hstack((recv_left[7], species.w, recv_right[7]))

    # Adapt the total number of particles
    species.Ntot = species.Ntot + recv_left.shape[1] + recv_right.shape[1]
    
    
def add_buffers_gpu( species, recv_left, recv_right ):
    """
    DOCUMENTATION
    """
    # Get the new number of particles
    new_Ntot = species.Ntot + recv_left.shape[1] + recv_right.shape[1]
    
    # Get the threads per block and the blocks per grid
    dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( new_Ntot )
    
    # Iterate over particle attributes
    i_attr = 0
    for attr in ['x', 'y', 'z', 'ux', 'uy', 'uz', 'w', 'inv_gamma']:
        # Copy the proper buffers to the GPU
        left_buffer = cuda.copy_to_device( recv_left[i_attr] )
        right_buffer = cuda.copy_to_device( recv_right[i_attr] )
        # Initialize the new particle array
        particle_array = cuda.device_array(new_Ntot, dtype=np.float64)
        # Merge the arrays on the GPU
        stay_buffer = getattr(species, attr)
        merge_buffers_to_particles[dim_grid_1d, dim_block_1d](
            particle_array, left_buffer, stay_buffer, right_buffer)
        # Assign the stay_buffer to the initial particle data array
        # and fill the sending buffers (if needed for MPI)
        setattr(species, attr, particle_array)
        # Increment the buffer index
        i_attr += 1

    # Adapt the total number of particles
    species.Ntot = new_Ntot
        
# Cuda routines
# -------------
if cuda_installed:

    @cuda.jit('void(float64[:], float64[:], float64[:], float64[:])')
    def split_particles_to_buffers( particle_array,
                    left_buffer, stay_buffer, right_buffer ):
        """
        Split the (sorted) particle array into the three arrays left_buffer,
        stay_buffer and right_buffer (in the same order)

        Parameters:
        ------------
        particle_array: 1d device arrays of floats
            Original array of particles
            (represents *one* of the particle quantities)

        left_buffer, right_buffer: 1d device arrays of floats
            Will contain the particles that are outside of the physical domain

        stay_buffer: 1d device array of floats
            Will contain the particles that are inside the physical domain
        """
        # Get a 1D CUDA grid (the index corresponds to a particle index)
        i = cuda.grid(1)

        # Define a few variables
        n_left = left_buffer.shape[0]
        n_stay = stay_buffer.shape[0]
        n_right = right_buffer.shape[0]

        # Copy the particles into the right buffer
        if i < n_left:
            left_buffer[i] = particle_array[i]
        elif i < n_left + n_stay:
            stay_buffer[i-n_left] = particle_array[i]
        elif i < n_left + n_stay + n_right:
            right_buffer[i-n_left-n_stay] = particle_array[i]

    @cuda.jit('void(float64[:], float64[:], float64[:], float64[:])')
    def merge_buffers_to_particles( particle_array,
                    left_buffer, stay_buffer, right_buffer ):
        """
        Copy left_buffer, stay_buffer and right_buffers into a single, larger
        array `particle_array`

        Parameters:
        ------------
        particle_array: 1d device arrays of floats
            Final array of particles
            (represents *one* of the particle quantities)

        left_buffer, right_buffer: 1d device arrays of floats
            Contain the particles received from the neighbor processors

        stay_buffer: 1d device array of floats
            Contain the particles that remained in the present processor
        """
        # Get a 1D CUDA grid (the index corresponds to a particle index)
        i = cuda.grid(1)

        # Define a few variables
        n_left = left_buffer.shape[0]
        n_stay = stay_buffer.shape[0]
        n_right = right_buffer.shape[0]

        # Copy the particles into the right buffer
        if i < n_left:
            particle_array[i] = left_buffer[i]
        elif i < n_left + n_stay:
            particle_array[i] = stay_buffer[i-n_left]
        elif i < n_left + n_stay + n_right:
            particle_array[i] = right_buffer[i-n_left-n_stay]
