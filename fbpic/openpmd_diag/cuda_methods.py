# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This files contains cuda methods that are used in the boosted-frame
diagnostics
"""
import numpy as np
from fbpic.cuda_utils import cuda, cuda_tpb_bpg_1d

@cuda.jit()
def extract_slice_from_gpu( pref_sum_curr, N_area, species ):
    """
    Extract the particles which have which have index between pref_sum_curr
    and pref_sum_curr + N_area, and return them in dictionaries.

    Parameters
    ----------
    pref_sum_curr: int
        The starting index needed for the extraction process
    N_area: int
        The number of particles to extract.
    species: an fbpic Species object
        The species from to extract data

    Returns
    -------
    particle_data : A dictionary of 1D float arrays (that are on the CPU)
        A dictionary that contains the particle data of
        the simulation (with normalized weigths).
    integer_data : A dictionary of 1D integer arrays (that are on the CPU
        A dictionary that contains the optional particle data
        (ionization level and particle id)
    """
    # Call kernel that extracts particles from GPU
    dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d(N_area)
    # - General particle quantities
    part_data = cuda.device_array( (8, N_area), dtype=np.float64 )
    extract_particles_from_gpu[dim_grid_1d, dim_block_1d]( pref_sum_curr,
         species.x, species.y, species.z, species.ux, species.uy, species.uz,
         species.w, species.inv_gamma, part_data )
    # - Optional integer particle arrays
    integer_data = {}
    if species.tracker is not None:
        integer_data['id'] = cuda.device_array( (N_area,), dtype=np.uint64 )
        extract_integers_from_gpu[dim_grid_1d, dim_block_1d](
            pref_sum_curr, species.tracker.id, integer_data['id'] )
    if species.ionizer is not None:
        integer_data['charge'] = cuda.device_array( (N_area,), dtype=np.uint64 )
        extract_integers_from_gpu[dim_grid_1d, dim_block_1d]( pref_sum_curr,
          species.ionizer.ionization_level, integer_data['charge'] )

    # Copy GPU arrays to the host
    part_data = part_data.copy_to_host()
    if species.ionizer is not None:
        integer_data['charge'] = \
            integer_data['charge'].copy_to_host()
    if species.tracker is not None:
        integer_data['id'] = \
            integer_data['id'].copy_to_host()

    # Return the data as dictionaries
    particle_data = { 'x':part_data[0], 'y':part_data[1], 'z':part_data[2],
        'ux':part_data[3], 'uy':part_data[4], 'uz':part_data[5],
        'w':part_data[6], 'inv_gamma':part_data[7] }

    return( particle_data, integer_data )


@cuda.jit()
def extract_particles_from_gpu( part_idx_start, x, y, z, ux, uy, uz, w,
                                inv_gamma, selected ):
    """
    Extract a selection of particles from the GPU and
    store them in a 2D array (8, N_part) in the following
    order: x, y, z, ux, uy, uz, w, inv_gamma.
    Selection goes from starting index (part_idx_start)
    to (part_idx_start + N_part-1), where N_part is derived
    from the shape of the 2D array (selected).

    Parameters
    ----------
    part_idx_start : int
        The starting index needed for the extraction process.
        ( minimum particle index to be extracted )

    x, y, z, ux, uy, uz, w, inv_gamma : 1D arrays of floats
        The GPU particle arrays for a given species.

    selected : 2D array of floats
        An empty GPU array to store the particles
        that are extracted.
    """
    i = cuda.grid(1)
    N_part = selected.shape[1]

    if i < N_part:
        ptcl_idx = part_idx_start+i
        selected[0, i] = x[ptcl_idx]
        selected[1, i] = y[ptcl_idx]
        selected[2, i] = z[ptcl_idx]
        selected[3, i] = ux[ptcl_idx]
        selected[4, i] = uy[ptcl_idx]
        selected[5, i] = uz[ptcl_idx]
        selected[6, i] = w[ptcl_idx]
        selected[7, i] = inv_gamma[ptcl_idx]

def extract_integers_from_gpu( part_idx_start, integer_array, selected ):
    """
    Extract a selection of particles from the GPU and
    store them in a 1D array (N_part,)
    Selection goes from starting index (part_idx_start)
    to (part_idx_start + N_part-1), where N_part is derived
    from the shape of the array `selected`.

    Parameters
    ----------
    part_idx_start : int
        The starting index needed for the extraction process.
        ( minimum particle index to be extracted )

    integer_array : 1D arrays of ints
        The GPU particle arrays for a given species. (e.g. particle id)

    selected : 1D array of ints
        An empty GPU array to store the particles that are extracted.
    """
    i = cuda.grid(1)
    N_part = selected.shape[1]

    if i < N_part:
        selected[i] = integer_array[part_idx_start+i]
