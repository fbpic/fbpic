# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines cuda methods that are used in particle ionization.

Apart from synthactic details, this file is very close to numba_methods.py
"""
from numba import cuda
from scipy.constants import c
# Import inline functions
from .inline_functions import get_ionization_probability, \
    get_E_amplitude, copy_ionized_electrons_batch
# Compile the inline functions for GPU
get_ionization_probability = cuda.jit( get_ionization_probability,
                                        device=True, inline=True )
get_E_amplitude = cuda.jit( get_E_amplitude,
                            device=True, inline=True )
copy_ionized_electrons_batch = cuda.jit( copy_ionized_electrons_batch,
                                            device=True, inline=True )

@cuda.jit()
def ionize_ions_cuda( N_batch, batch_size, Ntot, level_max,
    n_ionized, is_ionized, ionization_level, random_draw,
    adk_prefactor, adk_power, adk_exp_prefactor,
    ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz, w, w_times_level ):
    """
    For each ion macroparticle, decide whether it is going to
    be further ionized during this timestep, based on the ADK rate.

    Increment the elements in `ionization_level` accordingly, and update the
    `w_times_level` of the ions to take into account the change in level
    of the corresponding macroparticle.

    For the purpose of counting and creating the corresponding electrons,
    `is_ionized` (one element per macroparticle) is set to 1 at the position
    of the ionized ions, and `n_ionized` (one element per batch) counts
    the total number of ionized particles in the current batch.
    """
    # Loop over batches of particles
    i_batch = cuda.grid(1)
    if i_batch < N_batch:

        # Set the count of ionized particles in the batch to 0
        n_ionized[i_batch] = 0

        # Loop through the batch
        N_max = min( (i_batch+1)*batch_size, Ntot )
        for ip in range( i_batch*batch_size, N_max ):

            # Skip the ionization routine, if the maximal ionization level
            # has already been reached for this macroparticle
            level = ionization_level[ip]
            if level >= level_max:
                is_ionized[ip] = 0
                continue

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
                is_ionized[ip] = 1
                n_ionized[i_batch] += 1
                # Update the ionization level and the corresponding weight
                ionization_level[ip] += 1
                w_times_level[ip] = w[ip] * ionization_level[ip]
            else:
                is_ionized[ip] = 0

@cuda.jit()
def copy_ionized_electrons_cuda(
    N_batch, batch_size, elec_old_Ntot, ion_Ntot,
    cumulative_n_ionized, is_ionized,
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
    # Select the current batch
    i_batch = cuda.grid(1)
    if i_batch < N_batch:
        copy_ionized_electrons_batch(
            i_batch, batch_size, elec_old_Ntot, ion_Ntot,
            cumulative_n_ionized, is_ionized,
            elec_x, elec_y, elec_z, elec_inv_gamma,
            elec_ux, elec_uy, elec_uz, elec_w,
            elec_Ex, elec_Ey, elec_Ez, elec_Bx, elec_By, elec_Bz,
            ion_x, ion_y, ion_z, ion_inv_gamma,
            ion_ux, ion_uy, ion_uz, ion_w,
            ion_Ex, ion_Ey, ion_Ez, ion_Bx, ion_By, ion_Bz )
