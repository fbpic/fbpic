# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
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

    When the boundaries are open, only the particles that are in the
    outermost half of the guard cells are removed. The particles that
    are in the innermost half are kept.
    
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
        boundary is open (None).

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
        send_left, send_right = remove_particles_cpu( species, fld,
                                    nguard, left_proc, right_proc )

    return( send_left, send_right )
        
def remove_particles_cpu(species, fld, nguard, left_proc, right_proc):
    """
    Remove the particles that are outside of the physical domain (i.e.
    in the guard cells). Store them in sending buffers, which are returned.

    When the boundaries are open, only the particles that are in the
    outermost half of the guard cells are removed. The particles that
    are in the innermost half are kept.
    
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
        boundary is open (None).

    Returns
    -------
    Two arrays of shape (8,Nptcl) where Nptcl is the number of particles
    that are sent to the left proc and right proc respectively.
    If left_proc or right_proc is None, the corresponding array has Nptcl=0
    """
    # Calculate the positions between which to remove particles
    # For the open boundaries, only the particles in the outermost
    # half of the guard cells are removed
    zbox_min = fld.interp[0].zmin + nguard*fld.interp[0].dz
    zbox_max = fld.interp[0].zmax - nguard*fld.interp[0].dz
    if left_proc is None:
        zbox_min = fld.interp[0].zmin + int(nguard/2)*fld.interp[0].dz
    if right_proc is None:
        zbox_max = fld.interp[0].zmax - int(nguard/2)*fld.interp[0].dz
    
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
    Remove the particles that are outside of the physical domain (i.e.
    in the guard cells). Store them in sending buffers, which are returned.

    When the boundaries are open, only the particles that are in the
    outermost half of the guard cells are removed. The particles that
    are in the innermost half are kept.
    
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
        boundary is open (None).

    Returns
    -------
    Two arrays of shape (8,Nptcl) where Nptcl is the number of particles
    that are sent to the left proc and right proc respectively.
    If left_proc or right_proc is None, the corresponding array has Nptcl=0
    """
    # Check if particles are sorted  ; if not print a message and sort them
    # (The particles are usually expected to be sorted from the previous
    # iteration at this point - except in a restart from a checkpoint.)
    if species.sorted == False:
        print('Removing particles: Particles are unsorted. Sorting them now.')
        species.sort_particles(fld = fld)
        species.sorted = True

    # Get the particle indices between which to remove the particles
    # (Take into account the fact that the moving window may have
    # shifted the grid since the particles were last sorted: prefix_sum_shift)
    prefix_sum = species.prefix_sum
    Nz = fld.Nz
    Nr = fld.Nr
    i_min = prefix_sum.getitem( (nguard+fld.prefix_sum_shift)*Nr )
    i_max = prefix_sum.getitem( (Nz-nguard+fld.prefix_sum_shift)*Nr - 1 )
    # For the open boundaries, only the particles in the outermost
    # half of the guard cells are removed
    if left_proc is None:
        # Find the index in z below which particles are removed
        iz_min = max( int(nguard/2) + fld.prefix_sum_shift, 0 )
        i_min = prefix_sum.getitem( iz_min * Nr )
    if right_proc is None:
        # Find the index in z above which particles are removed
        iz_max = min( Nz - int(nguard/2) + fld.prefix_sum_shift, Nz )
        i_max = prefix_sum.getitem( iz_max * Nr - 1 )
    # Because of the way in which the prefix_sum is calculated, if the
    # cell that was requested for i_max is beyond the last non-empty cell,
    # i_max will be zero, but should in fact be species.Ntot
    if i_max == 0:
        i_max = species.Ntot

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
    for attr in ['x', 'y', 'z', 'ux', 'uy', 'uz', 'inv_gamma', 'w' ]:
        # Initialize 3 buffer arrays on the GPU
        left_buffer = cuda.device_array((N_send_l,), dtype=np.float64)
        right_buffer = cuda.device_array((N_send_r,), dtype=np.float64)
        stay_buffer = cuda.device_array((new_Ntot,), dtype=np.float64)
        # Split the particle array into the 3 buffers on the GPU
        particle_array = getattr(species, attr)
        split_particles_to_buffers[dim_grid_1d, dim_block_1d]( particle_array,
                    left_buffer, stay_buffer, right_buffer, i_min, i_max)
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
        
    # Return the sending buffers
    return( send_left, send_right )


def add_buffers_to_particles( species, recv_left, recv_right ):
    """
    Add the particles stored in recv_left and recv_right
    to the existing particle in species.

    Resize the auxiliary arrays of the particles Ex, Ey, Ez, Bx, By, Bz,
    as well as cell_idx, sorted_idx and sorting_buffer
    
    Parameters
    ----------
    species: a Particles object
        Contain the particles that stayed on the present processors

    recv_left, recv_right: 2darrays of floats
        Arrays of shape (8, Nptcl) that represent the particles that
        were received from the neighboring processors
        These arrays are always on the CPU (since they were used for MPI)
    """
    # Copy the buffers to an enlarged array
    if species.use_cuda:
        add_buffers_gpu( species, recv_left, recv_right )
    else:
        add_buffers_cpu( species, recv_left, recv_right )        
    
    # Reallocate the particles auxiliary arrays. This needs to be done,
    # as the total number of particles in this domain has changed.
    if species.use_cuda:
        shape = (species.Ntot,)
        # Reallocate empty field-on-particle arrays on the GPU
        species.Ex = cuda.device_array( shape, dtype=np.float64 )
        species.Ex = cuda.device_array( shape, dtype=np.float64 )
        species.Ey = cuda.device_array( shape, dtype=np.float64 )
        species.Ez = cuda.device_array( shape, dtype=np.float64 )
        species.Bx = cuda.device_array( shape, dtype=np.float64 )
        species.By = cuda.device_array( shape, dtype=np.float64 )
        species.Bz = cuda.device_array( shape, dtype=np.float64 )
        # Reallocate empty auxiliary sorting arrays on the GPU
        species.cell_idx = cuda.device_array( shape, dtype=np.int32 )
        species.sorted_idx = cuda.device_array( shape, dtype=np.int32 )
        species.sorting_buffer = cuda.device_array( shape, dtype=np.float64 )
    else:
        # Reallocate empty field-on-particle arrays on the CPU
        species.Ex = np.empty(species.Ntot, dtype=np.float64)
        species.Ey = np.empty(species.Ntot, dtype=np.float64)
        species.Ez = np.empty(species.Ntot, dtype=np.float64)
        species.Bx = np.empty(species.Ntot, dtype=np.float64)
        species.By = np.empty(species.Ntot, dtype=np.float64)
        species.Bz = np.empty(species.Ntot, dtype=np.float64)
        # Reallocate empty auxiliary sorting arrays on the CPU
        species.cell_idx = np.empty( species.Ntot, dtype=np.int32 )
        species.sorted_idx =np.empty( species.Ntot, dtype=np.int32 )
        species.sorting_buffer = np.empty( species.Ntot, dtype=np.float64 )

    # The particles are unsorted after adding new particles.
    species.sorted = False

def add_buffers_cpu( species, recv_left, recv_right ):
    """
    Add the particles stored in recv_left and recv_right
    to the existing particle in species.

    Parameters
    ----------
    species: a Particles object
        Contain the particles that stayed on the present processors

    recv_left, recv_right: 2darrays of floats
        Arrays of shape (8, Nptcl) that represent the particles that
        were received from the neighboring processors
        These arrays are always on the CPU (since they were used for MPI)
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
    Add the particles stored in recv_left and recv_right
    to the existing particle in species.

    Parameters
    ----------
    species: a Particles object
        Contain the particles that stayed on the present processors

    recv_left, recv_right: 2darrays of floats
        Arrays of shape (8, Nptcl) that represent the particles that
        were received from the neighboring processors
        These arrays are always on the CPU (since they were used for MPI)
    """
    # Get the new number of particles
    new_Ntot = species.Ntot + recv_left.shape[1] + recv_right.shape[1]
    
    # Get the threads per block and the blocks per grid
    dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( new_Ntot )
    
    # Iterate over particle attributes
    i_attr = 0
    for attr in ['x', 'y', 'z', 'ux', 'uy', 'uz', 'inv_gamma', 'w']:
        # Copy the proper buffers to the GPU
        left_buffer = cuda.to_device( recv_left[i_attr] )
        right_buffer = cuda.to_device( recv_right[i_attr] )
        # Initialize the new particle array
        particle_array = cuda.device_array( (new_Ntot,), dtype=np.float64)
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

    @cuda.jit('void(float64[:], float64[:], float64[:], \
                    float64[:], int32, int32)')
    def split_particles_to_buffers( particle_array, left_buffer,
                    stay_buffer, right_buffer, i_min, i_max ):
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
            Note: if the boundary is open, then these buffers have size 0
            and in this case, they will not be filled
            (the corresponding particles are simply lost)

        stay_buffer: 1d device array of floats
            Will contain the particles that are inside the physical domain

        i_min, i_max: int
            Indices of particle_array between which particles are kept
            (and thus copied to stay_buffer). The particles below i_min
            (resp. above i_max) are copied to left_buffer (resp. right_buffer)
        """
        # Get a 1D CUDA grid (the index corresponds to a particle index)
        i = cuda.grid(1)

        # Auxiliary variables
        n_left = left_buffer.shape[0]
        n_right = right_buffer.shape[0]
        Ntot = particle_array.shape[0]

        # Copy the particles into the right buffer
        if i < i_min:
            # Check whether buffer is not empty (open boundary)
            if (n_left != 0):
                left_buffer[i] = particle_array[i]
        elif i < i_max:
            stay_buffer[i-i_min] = particle_array[i]
        elif i < Ntot:
            # Check whether buffer is not empty (open boundary)
            if (n_right != 0):
                right_buffer[i-i_max] = particle_array[i]

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
