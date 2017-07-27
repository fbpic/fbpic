# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a number of methods that are useful for elementary processes
(e.g. ionization, Compton scattering) on CPU and GPU
"""
import numpy as np
from fbpic.threading_utils import njit_parallel, prange
# Check if CUDA is available, then import CUDA functions
from fbpic.cuda_utils import cuda_installed
if cuda_installed:
    from fbpic.cuda_utils import cuda_tpb_bpg_1d

def allocate_empty( N, use_cuda, dtype ):
    """
    # TODO
    """
    if use_cuda:
        cuda.device_array( (N,), dtype=dtype )
    else:
        np.empty( N, dtype=dtype )

def perform_cumsum( input_array, use_cuda ):
    """
    # TODO
    """
    cumulative_array = np.zeros( len(input_array)+1, dtype=np.int64 )
    np.cumsum( n_ionized, out=cumulative_array )

def reallocate_and_copy_old( species, use_cuda, old_Ntot, new_Ntot ):
    """
    # TODO
    """
    # On GPU, use one thread per particle
    if use_cuda:
        ptcl_grid_1d, ptcl_block_1d = cuda_tpb_bpg_1d( old_Ntot )

    # Iterate over particle attributes and copy the old particles
    for attr in ['x', 'y', 'z', 'ux', 'uy', 'uz', 'w', 'inv_gamma',
                    'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
        old_array = getattr(species, attr)
        new_array = allocate_empty( new_Ntot, use_cuda, dtype=np.float64 )
        if use_cuda:
            copy_particle_data_cuda[ ptcl_grid_1d, ptcl_block_1d ](
                old_Ntot, old_array, new_array )
        else:
            copy_particle_data_numba( old_Ntot, old_array, new_array )
        setattr( species, attr, new_array )
    # Copy the tracking id, if needed
    if species.tracker is not None:
        old_array = species.tracker.id
        new_array = allocate_empty( new_Ntot, use_cuda, dtype=np.uint64 )
        if use_cuda:
            copy_particle_data_cuda[ ptcl_grid_1d, ptcl_block_1d ](
                old_Ntot, old_array, new_array )
        else:
            copy_particle_data_numba( old_Ntot, old_array, new_array )
        species.tracker.id = new_array

    # Allocate the auxiliary arrays for GPU
    if use_cuda:
        species.cell_idx = cuda.device_array((new_Ntot,), dtype=np.int32)
        species.sorted_idx = cuda.device_array((new_Ntot,), dtype=np.uint32)
        species.sorting_buffer = cuda.device_array((new_Ntot,), dtype=np.float64)
        if species.n_integer_quantities > 0:
            species.int_sorting_buffer = \
                cuda.device_array( (new_Ntot,), dtype=np.uint64 )

    # Modify the total number of particles
    species.Ntot = new_Ntot


@njit_parallel
def copy_particle_data_numba( Ntot, old_array, new_array ):
    # Loop over single particles (in parallel if threading is enabled)
    for ip in prange( Ntot ):
        new_array[ip] = old_array[ip]
    return( new_array )

if cuda_installed:
    @cuda.jit()
    def copy_particle_data_cuda( Ntot, old_array, new_array ):
        # Loop over single particles
        ip = cuda.grid(1)
        if ip < Ntot:
            new_array[ip] = old_array[ip]
