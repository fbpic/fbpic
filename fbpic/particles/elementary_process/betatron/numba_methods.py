# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines numba methods that are used in particle ionization.

Apart from synthactic, this file is very close to cuda_methods.py
"""
import numba
from scipy.constants import c
from fbpic.utils.threading import njit_parallel, prange
# Import inline functions
from .inline_functions import get_ionization_probability, \
    get_E_amplitude, copy_ionized_electrons_batch
# Compile the inline functions for CPU
get_ionization_probability = numba.njit(get_ionization_probability)
get_E_amplitude = numba.njit(get_E_amplitude)
copy_ionized_electrons_batch = numba.njit(copy_ionized_electrons_batch)

@njit_parallel
def ionize_ions_numba( N_batch, batch_size, Ntot,
    level_start, level_max, n_levels,
    n_ionized, ionized_from, ionization_level, random_draw,
    adk_prefactor, adk_power, adk_exp_prefactor,
    ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz, w, w_times_level ):
    """
    For each ion macroparticle, decide whether it is going to
    be further ionized during this timestep, based on the ADK rate.

    Increment the elements in `ionization_level` accordingly, and update
    `w_times_level` of the ions to take into account the change in level
    of the corresponding macroparticle.

    For the purpose of counting and creating the corresponding electrons,
    `ionized_from` (one element per macroparticle) is set to -1 at the position
    of the unionized ions, and to the level (before ionization) otherwise
    `n_ionized` (one element per batch, and per ionizable level that needs
    to be distinguished) counts the total number of ionized particles
    in the current batch.
    """
    # Loop over batches of particles (in parallel, if threading is enabled)
    for i_batch in prange( N_batch ):

        # Set the count of ionized particles in the batch to 0
        for i_level in range(n_levels):
            n_ionized[i_level, i_batch] = 0

        # Loop through the batch
        N_max = min( (i_batch+1)*batch_size, Ntot )
        for ip in range(i_batch*batch_size, N_max):

            # Skip the ionization routine, if the maximal ionization level
            # has already been reached for this macroparticle
            level = ionization_level[ip]
            if level >= level_max:
                ionized_from[ip] = -1
            else:
                # Calculate the amplitude of the electric field,
                # in the frame of the electrons (device inline function)
                E, gamma = get_E_amplitude( ux[ip], uy[ip], uz[ip],
                        Ex[ip], Ey[ip], Ez[ip], c*Bx[ip], c*By[ip], c*Bz[ip] )
                # Get ADK rate (device inline function)
                p = get_ionization_probability( E, gamma,
                  adk_prefactor[level], adk_power[level], adk_exp_prefactor[level])
                # Ionize particles
                if random_draw[ip] < p:
                    # Set the corresponding flag and update particle count
                    ionized_from[ip] = level-level_start
                    if n_levels == 1:
                        # No need to distinguish ionization levels
                        n_ionized[0, i_batch] += 1
                    else:
                        # Distinguish count for each ionizable level
                        n_ionized[level-level_start, i_batch] += 1
                    # Update the ionization level and the corresponding weight
                    ionization_level[ip] += 1
                    w_times_level[ip] = w[ip] * ionization_level[ip]
                else:
                    ionized_from[ip] = -1

    return( n_ionized, ionized_from, ionization_level, w_times_level )


@njit_parallel
def copy_ionized_electrons_numba(
    N_batch, batch_size, elec_old_Ntot, ion_Ntot,
    cumulative_n_ionized, ionized_from,
    i_level, store_electrons_per_level,
    elec_x, elec_y, elec_z, elec_inv_gamma,
    elec_ux, elec_uy, elec_uz, elec_w,
    elec_Ex, elec_Ey, elec_Ez, elec_Bx, elec_By, elec_Bz,
    ion_x, ion_y, ion_z, ion_inv_gamma,
    ion_ux, ion_uy, ion_uz, ion_w,
    ion_Ex, ion_Ey, ion_Ez, ion_Bx, ion_By, ion_Bz ):
    """
    Create the new electrons by copying the properties (position, momentum,
    etc) of the ions that they originate from.
    """
    #  Loop over batches of particles (in parallel, if threading is enabled)
    for i_batch in prange( N_batch ):
        copy_ionized_electrons_batch(
            i_batch, batch_size, elec_old_Ntot, ion_Ntot,
            cumulative_n_ionized, ionized_from,
            i_level, store_electrons_per_level,
            elec_x, elec_y, elec_z, elec_inv_gamma,
            elec_ux, elec_uy, elec_uz, elec_w,
            elec_Ex, elec_Ey, elec_Ez, elec_Bx, elec_By, elec_Bz,
            ion_x, ion_y, ion_z, ion_inv_gamma,
            ion_ux, ion_uy, ion_uz, ion_w,
            ion_Ex, ion_Ey, ion_Ez, ion_Bx, ion_By, ion_Bz )

    return( elec_x, elec_y, elec_z, elec_inv_gamma,
        elec_ux, elec_uy, elec_uz, elec_w,
        elec_Ex, elec_Ey, elec_Ez, elec_Bx, elec_By, elec_Bz )
