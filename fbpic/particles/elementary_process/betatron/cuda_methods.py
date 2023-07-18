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
from .inline_functions import get_angles_and_gamma, get_particle_radiation, \
    get_linear_coefficients, vector_interpolate


# Compile the inline functions for GPU
get_angles_and_gamma = cuda.jit( get_angles_and_gamma, device=True, inline=True)
get_particle_radiation = cuda.jit( get_particle_radiation, device=True, inline=True )
get_linear_coefficients = cuda.jit( get_linear_coefficients, device=True, inline=True )
vector_interpolate = cuda.jit( vector_interpolate, device=True, inline=True )

@compile_cupy
def gather_betatron_cuda(
    N_batch, batch_size, Ntot, ux, uy, uz, Ex, Ey, Ez,
    Bx, By, Bz, w, Larmore_factor, gamma_cutoff,
    omega_ax, SR_dxi, SR_xi_data,
    theta_x_min, theta_x_max, d_th_x,
    theta_y_min, theta_y_max, d_th_y,
    spect_batch, radiation_data):
    """
    """
    # Loop over batches of particles
    i_batch = cuda.grid(1)
    if i_batch < N_batch:
        # Loop through the batch
        N_max = min( (i_batch+1)*batch_size, Ntot )

        for ip in range( i_batch*batch_size, N_max ):

            theta_x, theta_y, gamma_p = get_angles_and_gamma(
                ux[ip], uy[ip], uz[ip]
            )

            if (gamma_p < gamma_cutoff) \
              or (theta_x > theta_x_max) \
              or (theta_y > theta_y_max) \
              or (theta_x < theta_x_min) \
              or (theta_y < theta_y_min):
                continue

            spect_loc = spect_batch[ip]
            omega_c, Energy_Larmor = get_particle_radiation(
                ux[ip], uy[ip], uz[ip], w[ip],
                Ex[ip], Ey[ip], Ez[ip], c*Bx[ip],
                c*By[ip], c*Bz[ip], gamma_p, Larmore_factor
            )

            th_ix, s0_x, s1_x = get_linear_coefficients(theta_x, theta_x_min, d_th_x)
            th_iy, s0_y, s1_y = get_linear_coefficients(theta_y, theta_y_min, d_th_y)

            spect_loc = vector_interpolate(
                omega_ax / omega_c, spect_loc, SR_dxi, SR_xi_data
            )
            spect_loc *= Energy_Larmor

            radiation_data[:, i_x, i_y] += spect_loc * s0_x * s0_y
            radiation_data[:, i_x+1, i_y] += spect_loc * s1_x * s0_y

            radiation_data[:, i_x, i_y+1] += spect_loc * s0_x * s1_y
            radiation_data[:, i_x+1, i_y+1] += spect_loc * s1_x * s1_y
