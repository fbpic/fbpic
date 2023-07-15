# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines cuda methods that are used in particle ionization.

Apart from synthactic details, this file is very close to numba_methods.py
"""
from numba import cuda
from fbpic.utils.cuda import compile_cupy
from scipy.constants import c
# Import inline functions
from .inline_functions import get_gamma, get_particle_radiation

# Compile the inline functions for GPU
get_gamma = cuda.jit( get_gamma, device=True, inline=True)
get_particle_radiation = cuda.jit( get_particle_radiation, device=True, inline=True )

@compile_cupy
def gather_betatron_cuda(
    N_batch, batch_size, Ntot, ux, uy, uz, Ex, Ey, Ez,
    Bx, By, Bz, w, Larmore_factor, gamma_cutoff, radiation_data):
    """
    """
    # Loop over batches of particles
    i_batch = cuda.grid(1)
    if i_batch < N_batch:
        # Loop through the batch
        N_max = min( (i_batch+1)*batch_size, Ntot )
        for ip in range( i_batch*batch_size, N_max ):
            gamma_p = get_gamma(ux, uy, uz)
            if gamma_p < gamma_cutoff:
                continue

            theta_x, theta_y, omega_c, Energy_Larmor = get_particle_radiation(
                ux[ip], uy[ip], uz[ip], w[ip],
                Ex[ip], Ey[ip], Ez[ip], c*Bx[ip],
                c*By[ip], c*Bz[ip], gamma_p, Larmore_factor)

@compile_cupy
def copy_ionized_electrons_cuda(
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
    # Select the current batch
    i_batch = cuda.grid(1)
    if i_batch < N_batch:
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
