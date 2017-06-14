# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to handle mpi buffers for the particles
"""
import numpy as np
import numba
# Check if CUDA is available, then import CUDA functions
from fbpic.cuda_utils import cuda_installed
if cuda_installed:
    from fbpic.cuda_utils import cuda, cuda_tpb_bpg_1d

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
    float_send_left, float_send_right, uint_send_left, uint_send_right:
        arrays of shape (n_float,Nptcl) and (n_int,Nptcl) where Nptcl
        is the number of particles that are sent to the left
        proc and right proc respectively, and where n_float and n_int
        are the number of float and integer quantities respectively
    If left_proc or right_proc is None, the corresponding array has Nptcl=0
    """
    if species.use_cuda:
        # Remove outside particles on GPU, and copy buffers on CPU
        float_send_left, float_send_right, uint_send_left, uint_send_right = \
            remove_particles_gpu( species, fld, nguard, left_proc, right_proc )
    else:
        # Remove outside particles on the CPU
        float_send_left, float_send_right, uint_send_left, uint_send_right = \
            remove_particles_cpu( species, fld, nguard, left_proc, right_proc )

    return(float_send_left, float_send_right, uint_send_left, uint_send_right)

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
    float_send_left, float_send_right, uint_send_left, uint_send_right:
        arrays of shape (n_float,Nptcl) and (n_int,Nptcl) where Nptcl
        is the number of particles that are sent to the left
        proc and right proc respectively, and where n_float and n_int
        are the number of float and integer quantities respectively
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

    # Shortcuts
    n_float = species.n_float_quantities
    n_int = species.n_integer_quantities

    # Allocate and fill left sending buffer
    if left_proc is not None:
        N_send_l = selec_left.sum()
        float_send_left = np.empty((n_float, N_send_l), dtype=np.float64)
        uint_send_left = np.empty((n_int, N_send_l), dtype=np.uint64)
        float_send_left[0,:] = species.x[selec_left]
        float_send_left[1,:] = species.y[selec_left]
        float_send_left[2,:] = species.z[selec_left]
        float_send_left[3,:] = species.ux[selec_left]
        float_send_left[4,:] = species.uy[selec_left]
        float_send_left[5,:] = species.uz[selec_left]
        float_send_left[6,:] = species.inv_gamma[selec_left]
        float_send_left[7,:] = species.w[selec_left]
        i_attr = 0
        if species.tracker is not None:
            uint_send_left[i_attr,:] = species.tracker.id[selec_left]
            i_attr += 1
        if species.ionizer is not None:
            uint_send_left[i_attr,:] = \
                species.ionizer.ionization_level[selec_left]
            float_send_left[8,:] = species.ionizer.neutral_weight[selec_left]
    else:
        # No need to allocate and copy data ; return an empty array
        float_send_left = np.empty((n_float, 0), dtype=np.float64)
        uint_send_left = np.empty((n_int, 0), dtype=np.uint64)

    # Allocate and fill right sending buffer
    if right_proc is not None:
        N_send_r = selec_right.sum()
        float_send_right = np.empty((n_float, N_send_r), dtype=np.float64)
        uint_send_right = np.empty((n_int, N_send_r), dtype=np.float64)
        float_send_right[0,:] = species.x[selec_right]
        float_send_right[1,:] = species.y[selec_right]
        float_send_right[2,:] = species.z[selec_right]
        float_send_right[3,:] = species.ux[selec_right]
        float_send_right[4,:] = species.uy[selec_right]
        float_send_right[5,:] = species.uz[selec_right]
        float_send_right[6,:] = species.inv_gamma[selec_right]
        float_send_right[7,:] = species.w[selec_right]
        i_attr = 0
        if species.tracker is not None:
            uint_send_right[i_attr,:] = species.tracker.id[selec_right]
            i_attr += 1
        if species.ionizer is not None:
            uint_send_right[i_attr,:] = \
                species.ionizer.ionization_level[selec_right]
            float_send_right[8,:] = species.ionizer.neutral_weight[selec_right]
    else:
        # No need to allocate and copy data ; return an empty array
        float_send_right = np.empty((n_float, 0), dtype = np.float64)
        uint_send_right = np.empty((n_int, 0), dtype=np.float64)

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
    if species.tracker is not None:
        species.tracker.id = species.tracker.id[selec_stay]
    if species.ionizer is not None:
        species.ionizer.neutral_weight = \
            species.ionizer.neutral_weight[selec_stay]
        species.ionizer.ionization_level = \
            species.ionizer.ionization_level[selec_stay]

    # Return the sending buffers
    return(float_send_left, float_send_right, uint_send_left, uint_send_right)

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
    float_send_left, float_send_right, uint_send_left, uint_send_right:
        arrays of shape (n_float,Nptcl) and (n_int,Nptcl) where Nptcl
        is the number of particles that are sent to the left
        proc and right proc respectively, and where n_float and n_int
        are the number of float and integer quantities respectively
    """
    # Check if particles are sorted  ; if not print a message and sort them
    # (The particles are usually expected to be sorted from the previous
    # iteration at this point - except in a restart from a checkpoint.)
    if species.sorted == False:
        print(
            '\n Removing particles: Particles are unsorted. Sorting them now.')
        species.sort_particles(fld = fld)
        species.sorted = True

    # Get the particle indices between which to remove the particles
    # (Take into account the fact that the moving window may have
    # shifted the grid since the particles were last sorted: prefix_sum_shift)
    prefix_sum = species.prefix_sum
    Nz = fld.Nz
    Nr = fld.Nr
    # Find the z index of the first cell for which particles are kept
    if left_proc is None:
        # Open boundary: particles in outermost half of guard cells are removed
        iz_min = max( int(nguard/2) + fld.prefix_sum_shift, 0 )
    else:
        # Normal boundary: all particles in guard cells are removed
        iz_min = max( nguard + fld.prefix_sum_shift, 0 )
    # Find the z index of the first cell for which particles are removed again
    if right_proc is None:
        # Open boundary: particles in outermost half of guard cells are removed
        iz_max = min( Nz - int(nguard/2) + fld.prefix_sum_shift, Nz )
    else:
        # Normal boundary: all particles in guard cells are removed
        iz_max = min( Nz - nguard, Nz )
    # Find the corresponding indices in the particle array
    # Reminder: prefix_sum[i] is the cumulative sum of the number of particles
    # in cells 0 to i (where cell i is included)
    if iz_min*Nr - 1 >= 0:
        i_min = prefix_sum.getitem( iz_min*Nr - 1 )
    else:
        i_min = 0
    i_max = prefix_sum.getitem( iz_max*Nr - 1 )
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
    n_float = species.n_float_quantities
    n_int = species.n_integer_quantities
    if left_proc is not None:
        float_send_left = np.empty((n_float, N_send_l), dtype=np.float64)
        uint_send_left = np.empty((n_int, N_send_l), dtype=np.uint64)
    else:
        float_send_left = np.empty((n_float, 0), dtype=np.float64)
        uint_send_left = np.empty((n_int, 0), dtype=np.uint64)
    if right_proc is not None:
        float_send_right = np.empty((n_float, N_send_r), dtype=np.float64)
        uint_send_right = np.empty((n_int, N_send_r), dtype=np.uint64)
    else:
        float_send_right = np.empty((n_float, 0), dtype=np.float64)
        uint_send_right = np.empty((n_int, 0), dtype=np.uint64)

    # Get the threads per block and the blocks per grid
    dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( species.Ntot )
    # Float quantities:
    # Build list of float attributes to copy
    attr_list = [ (species,'x'), (species,'y'), (species,'z'),
                    (species,'ux'), (species,'uy'), (species,'uz'),
                    (species,'inv_gamma'), (species,'w') ]
    if species.ionizer is not None:
        attr_list.append( (species.ionizer,'neutral_weight') )
    # Loop through the float attributes
    for i_attr in range(n_float):
        # Initialize 3 buffer arrays on the GPU (need to be initialized
        # inside the loop, as `copy_to_host` invalidates these arrays)
        left_buffer = cuda.device_array((N_send_l,), dtype=np.float64)
        right_buffer = cuda.device_array((N_send_r,), dtype=np.float64)
        stay_buffer = cuda.device_array((new_Ntot,), dtype=np.float64)
        # Check that the buffers are still on GPU
        # (safeguard against automatic memory management)
        assert type(left_buffer) != np.ndarray
        assert type(right_buffer) != np.ndarray
        assert type(left_buffer) != np.ndarray
        # Split the particle array into the 3 buffers on the GPU
        particle_array = getattr( attr_list[i_attr][0], attr_list[i_attr][1] )
        split_particles_to_buffers[dim_grid_1d, dim_block_1d]( particle_array,
                    left_buffer, stay_buffer, right_buffer, i_min, i_max)
        # Assign the stay_buffer to the initial particle data array
        # and fill the sending buffers (if needed for MPI)
        setattr( attr_list[i_attr][0], attr_list[i_attr][1], stay_buffer)
        if left_proc is not None:
            left_buffer.copy_to_host( float_send_left[i_attr] )
        if right_proc is not None:
            right_buffer.copy_to_host( float_send_right[i_attr] )

    # Integer quantities:
    if n_int > 0:
        attr_list = []
    if species.tracker is not None:
        attr_list.append( (species.tracker,'id') )
    if species.ionizer is not None:
        attr_list.append( (species.ionizer,'ionization_level') )
    for i_attr in range(n_int):
        # Initialize 3 buffer arrays on the GPU (need to be initialized
        # inside the loop, as `copy_to_host` invalidates these arrays)
        left_buffer = cuda.device_array((N_send_l,), dtype=np.uint64)
        right_buffer = cuda.device_array((N_send_r,), dtype=np.uint64)
        stay_buffer = cuda.device_array((new_Ntot,), dtype=np.uint64)
        # Split the particle array into the 3 buffers on the GPU
        particle_array = getattr( attr_list[i_attr][0], attr_list[i_attr][1] )
        split_particles_to_buffers[dim_grid_1d, dim_block_1d]( particle_array,
            left_buffer, stay_buffer, right_buffer, i_min, i_max)
        # Assign the stay_buffer to the initial particle data array
        # and fill the sending buffers (if needed for MPI)
        setattr( attr_list[i_attr][0], attr_list[i_attr][1], stay_buffer)
        if left_proc is not None:
            left_buffer.copy_to_host( uint_send_left[i_attr] )
        if right_proc is not None:
            right_buffer.copy_to_host( uint_send_right[i_attr] )

    # Change the new total number of particles
    species.Ntot = new_Ntot

    # Return the sending buffers
    return(float_send_left, float_send_right, uint_send_left, uint_send_right)


def add_buffers_to_particles( species, float_recv_left, float_recv_right,
                                        uint_recv_left, uint_recv_right):
    """
    Add the particles stored in recv_left and recv_right
    to the existing particle in species.

    Resize the auxiliary arrays of the particles Ex, Ey, Ez, Bx, By, Bz,
    as well as cell_idx, sorted_idx and sorting_buffer

    Parameters
    ----------
    species: a Particles object
        Contain the particles that stayed on the present processors

    float_recv_left, float_recv_right, uint_recv_left, uint_recv_right:
        arrays of shape (n_float,Nptcl) and (n_int,Nptcl) where Nptcl
        is the number of particles that are received to the left
        proc and right proc respectively, and where n_float and n_int
        are the number of float and integer quantities respectively
        These arrays are always on the CPU (since they were used for MPI)
    """
    # Copy the buffers to an enlarged array
    if species.use_cuda:
        add_buffers_gpu( species, float_recv_left, float_recv_right,
                                uint_recv_left, uint_recv_right )
    else:
        add_buffers_cpu( species, float_recv_left, float_recv_right,
                                uint_recv_left, uint_recv_right )

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
        if species.n_integer_quantities > 0:
            species.int_sorting_buffer = \
                cuda.device_array( shape, dtype=np.uint64 )
    else:
        # Reallocate empty field-on-particle arrays on the CPU
        species.Ex = np.empty(species.Ntot, dtype=np.float64)
        species.Ey = np.empty(species.Ntot, dtype=np.float64)
        species.Ez = np.empty(species.Ntot, dtype=np.float64)
        species.Bx = np.empty(species.Ntot, dtype=np.float64)
        species.By = np.empty(species.Ntot, dtype=np.float64)
        species.Bz = np.empty(species.Ntot, dtype=np.float64)

    # The particles are unsorted after adding new particles.
    species.sorted = False

def add_buffers_cpu( species, float_recv_left, float_recv_right,
                            uint_recv_left, uint_recv_right):
    """
    Add the particles stored in recv_left and recv_right
    to the existing particle in species.

    Parameters
    ----------
    species: a Particles object
        Contain the particles that stayed on the present processors

    float_recv_left, float_recv_right, uint_recv_left, uint_recv_right:
        arrays of shape (n_float,Nptcl) and (n_int,Nptcl) where Nptcl
        is the number of particles that are received to the left
        proc and right proc respectively, and where n_float and n_int
        are the number of float and integer quantities respectively
        These arrays are always on the CPU (since they were used for MPI)
    """
    # Form the new particle arrays by adding the received particles
    # from the left and the right to the particles that stay in the domain
    species.x = np.hstack((float_recv_left[0], species.x, float_recv_right[0]))
    species.y = np.hstack((float_recv_left[1], species.y, float_recv_right[1]))
    species.z = np.hstack((float_recv_left[2], species.z, float_recv_right[2]))
    species.ux = np.hstack((float_recv_left[3],species.ux,float_recv_right[3]))
    species.uy = np.hstack((float_recv_left[4],species.uy,float_recv_right[4]))
    species.uz = np.hstack((float_recv_left[5],species.uz,float_recv_right[5]))
    species.inv_gamma = \
        np.hstack((float_recv_left[6], species.inv_gamma, float_recv_right[6]))
    species.w = np.hstack((float_recv_left[7], species.w, float_recv_right[7]))
    i_attr = 0
    if species.tracker is not None:
        species.tracker.id = np.hstack( (uint_recv_left[i_attr],
            species.tracker.id, uint_recv_right[i_attr]))
        i_attr += 1
    if species.ionizer is not None:
        species.ionizer.ionization_level = np.hstack( (uint_recv_left[i_attr],
            species.ionizer.ionization_level, uint_recv_right[i_attr]))
        species.ionizer.neutral_weight = np.hstack( (float_recv_left[8],
            species.ionizer.neutral_weight, float_recv_right[8]))

    # Adapt the total number of particles
    species.Ntot = species.Ntot + float_recv_left.shape[1] \
                                + float_recv_right.shape[1]


def add_buffers_gpu( species, float_recv_left, float_recv_right,
                            uint_recv_left, uint_recv_right):
    """
    Add the particles stored in recv_left and recv_right
    to the existing particle in species.

    Parameters
    ----------
    species: a Particles object
        Contain the particles that stayed on the present processors

    float_recv_left, float_recv_right, uint_recv_left, uint_recv_right:
        arrays of shape (n_float,Nptcl) and (n_int,Nptcl) where Nptcl
        is the number of particles that are received to the left
        proc and right proc respectively, and where n_float and n_int
        are the number of float and integer quantities respectively
        These arrays are always on the CPU (since they were used for MPI)
    """
    # Get the new number of particles
    new_Ntot = species.Ntot + float_recv_left.shape[1] \
                            + float_recv_right.shape[1]

    # Get the threads per block and the blocks per grid
    dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( new_Ntot )

    # Iterate over particle attributes
    # Build list of float attributes to copy
    attr_list = [ (species,'x'), (species,'y'), (species,'z'), \
                  (species,'ux'), (species,'uy'), (species,'uz'), \
                  (species,'inv_gamma'), (species,'w') ]
    if species.ionizer is not None:
        attr_list += [ (species.ionizer, 'neutral_weight') ]
    # Loop through the float quantities
    for i_attr in range( len(attr_list) ):
        # Copy the proper buffers to the GPU
        left_buffer = cuda.to_device( float_recv_left[i_attr] )
        right_buffer = cuda.to_device( float_recv_right[i_attr] )
        # Initialize the new particle array
        particle_array = cuda.device_array( (new_Ntot,), dtype=np.float64)
        # Merge the arrays on the GPU
        stay_buffer = getattr( attr_list[i_attr][0], attr_list[i_attr][1])
        merge_buffers_to_particles[dim_grid_1d, dim_block_1d](
            particle_array, left_buffer, stay_buffer, right_buffer)
        # Assign the stay_buffer to the initial particle data array
        # and fill the sending buffers (if needed for MPI)
        setattr(attr_list[i_attr][0], attr_list[i_attr][1], particle_array)

    # Build list of integer quantities to copy
    attr_list = []
    if species.tracker is not None:
        attr_list.append( (species.tracker,'id') )
    if species.ionizer is not None:
        attr_list.append( (species.ionizer,'ionization_level') )
    # Loop through the integer quantities
    for i_attr in range( len(attr_list) ):
        # Copy the proper buffers to the GPU
        left_buffer = cuda.to_device( uint_recv_left[i_attr] )
        right_buffer = cuda.to_device( uint_recv_right[i_attr] )
        # Initialize the new particle array
        particle_array = cuda.device_array( (new_Ntot,), dtype=np.uint64)
        # Merge the arrays on the GPU
        stay_buffer = getattr( attr_list[i_attr][0], attr_list[i_attr][1])
        merge_buffers_to_particles[dim_grid_1d, dim_block_1d](
            particle_array, left_buffer, stay_buffer, right_buffer)
        # Assign the stay_buffer to the initial particle data array
        # and fill the sending buffers (if needed for MPI)
        setattr(attr_list[i_attr][0], attr_list[i_attr][1], particle_array)

    # Adapt the total number of particles
    species.Ntot = new_Ntot


def shift_particles_periodic_subdomain( species, zmin, zmax ):
    """
    Assuming the local subdomain is periodic:
    Shift the particle positions by an integer number of box length,
    so that outside particle are back inside the physical domain

    Parameters:
    -----------
    species: an fbpic.Species object
        Contains the particle data
    zmin, zmax: floats
        Positions of the edges of the periodic box
    """
    # Perform the shift on the GPU
    if species.use_cuda:
        dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( species.Ntot )
        shift_particles_periodic_cuda[ dim_grid_1d, dim_block_1d ](
                                                    species.z, zmin, zmax )
    # Perform the shift on the CPU
    else:
        shift_particles_periodic_numba( species.z, zmin, zmax )

@numba.jit(nopython=True)
def shift_particles_periodic_numba( z, zmin, zmax ):
    """
    Shift the particle positions by an integer number of box length,
    so that outside particle are back inside the physical domain

    Parameters:
    -----------
    z: 1darray of floats
        The z position of the particles (one element per particle)
    zmin, zmax: floats
        Positions of the edges of the periodic box
    """
    # Get box length
    l_box = zmax - zmin
    # Loop through the particles and shift their positions
    for i in range(len(z)):
        while z[i] >= zmax:
            z[i] -= l_box
        while z[i] < zmin:
            z[i] += l_box

# Cuda routines
# -------------
if cuda_installed:

    @cuda.jit
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

    @cuda.jit
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

    @cuda.jit('void(float64[:], float64, float64)')
    def shift_particles_periodic_cuda( z, zmin, zmax ):
        """
        Shift the particle positions by an integer number of box length,
        so that outside particle are back inside the physical domain

        Parameters:
        -----------
        z: 1darray of floats
            The z position of the particles (one element per particle)
        zmin, zmax: floats
            Positions of the edges of the periodic box
        """
        # Get a 1D CUDA grid (the index corresponds to a particle index)
        i = cuda.grid(1)
        # Get box length
        l_box = zmax - zmin
        # Shift particle position
        if i < z.shape[0]:
            while z[i] >= zmax:
                z[i] -= l_box
            while z[i] < zmin:
                z[i] += l_box
