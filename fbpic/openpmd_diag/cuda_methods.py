# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This files contains cuda methods that are used in the boosted-frame
diagnostics
"""
import numpy as np
from fbpic.utils.cuda import cupy, cuda, cuda_tpb_bpg_1d, compile_cupy

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
        the simulation (with normalized weigths), including optional
        integer arrays (e.g. "id", "charge")
    """
    # Call kernel that extracts particles from GPU
    dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d(N_area)
    # - General particle quantities
    part_data = cupy.empty( (8, N_area), dtype=np.float64 )
    extract_particles_from_gpu[dim_grid_1d, dim_block_1d]( pref_sum_curr,
         species.x, species.y, species.z, species.ux, species.uy, species.uz,
         species.w, species.inv_gamma, part_data )
    # - Optional particle arrays
    if species.tracker is not None:
        selected_particle_id = cupy.empty( (N_area,), dtype=np.uint64 )
        extract_array_from_gpu[dim_grid_1d, dim_block_1d](
            pref_sum_curr, species.tracker.id, selected_particle_id )
    if species.ionizer is not None:
        selected_particle_charge = cupy.empty( (N_area,), dtype=np.uint64 )
        extract_array_from_gpu[dim_grid_1d, dim_block_1d]( pref_sum_curr,
          species.ionizer.ionization_level, selected_particle_charge )
        selected_particle_weight = cupy.empty( (N_area,), dtype=np.float64 )
        extract_array_from_gpu[dim_grid_1d, dim_block_1d]( pref_sum_curr,
          species.ionizer.w_times_level, selected_particle_weight )

    # Copy GPU arrays to the host
    part_data = part_data.get()
    particle_data = { 'x':part_data[0], 'y':part_data[1], 'z':part_data[2],
        'ux':part_data[3], 'uy':part_data[4], 'uz':part_data[5],
        'w':part_data[6], 'inv_gamma':part_data[7] }
    if species.tracker is not None:
        particle_data['id'] = selected_particle_id.get()
    if species.ionizer is not None:
        particle_data['charge'] = selected_particle_charge.get()
        # Replace particle weight
        particle_data['w'] = selected_particle_weight.get()

    # Return the data as dictionary
    return( particle_data )

@compile_cupy
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

@compile_cupy
def extract_array_from_gpu( part_idx_start, array, selected ):
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

    array : 1D arrays of ints or floats
        The GPU particle arrays for a given species. (e.g. particle id)

    selected : 1D array of ints or floats
        An empty GPU array to store the particles that are extracted.
    """
    i = cuda.grid(1)
    N_part = selected.shape[0]

    if i < N_part:
        selected[i] = array[part_idx_start+i]
