# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines cuda methods that are used in particle ionization.

Apart from synthactic details, this file is very close to numba_methods.py
"""
from numba import cuda

from numba.cuda.random import xoroshiro128p_normal_float64
from fbpic.utils.cuda import compile_cupy
from scipy.constants import c
import math
# Import inline functions
from .inline_functions import get_angles_and_gamma, get_particle_radiation, \
    get_linear_coefficients

# Compile the inline functions for GPU
get_angles_and_gamma = cuda.jit( get_angles_and_gamma, device=True, inline=True)
get_particle_radiation = cuda.jit( get_particle_radiation, device=True, inline=True )
get_linear_coefficients = cuda.jit( get_linear_coefficients, device=True, inline=True )

#@cuda.jit(device=True, inline=True )

# @compile_cupy
@cuda.jit
def gather_synchrotron_cuda(
    N_batch, batch_size, Ntot,
    ux, uy, uz, Ex, Ey, Ez,
    Bx, By, Bz, w,
    Larmore_factor_density,
    Larmore_factor_momentum,
    gamma_cutoff,
    omega_ax, SR_dxi, SR_xi_data,
    theta_x_min, theta_x_max, d_th_x,
    theta_y_min, theta_y_max, d_th_y,
    spect_batch, rng_states_batch, radiation_data):
    """
    doc
    """
    # Loop over batches of particles
    i_batch = cuda.grid(1)
    if i_batch < N_batch:
        # Loop through the batch
        spect_loc = spect_batch[i_batch]
        N_omega = spect_loc.size

        N_max = min( (i_batch+1)*batch_size, Ntot )
        for ip in range( i_batch*batch_size, N_max ):

            theta_x, theta_y, gamma_p = get_angles_and_gamma(
                ux[ip], uy[ip], uz[ip]
            )

            theta_diffusion = 2**-1.5 / gamma_p

            theta_x += theta_diffusion * xoroshiro128p_normal_float64(
                rng_states_batch, i_batch)

            theta_y += theta_diffusion * xoroshiro128p_normal_float64(
                rng_states_batch, i_batch)

            if (gamma_p <= gamma_cutoff) \
              or (theta_x >= theta_x_max) \
              or (theta_y >= theta_y_max) \
              or (theta_x <= theta_x_min) \
              or (theta_y <= theta_y_min):
                continue

            th_ix, s0_x, s1_x = get_linear_coefficients(
                theta_x, theta_x_min, d_th_x
            )
            th_iy, s0_y, s1_y = get_linear_coefficients(
                theta_y, theta_y_min, d_th_y
            )

            spect_loc, ux_ph, uy_ph, uz_ph = get_particle_radiation(
                    ux[ip], uy[ip], uz[ip], w[ip],
                    Ex[ip], Ey[ip], Ez[ip],
                    c*Bx[ip], c*By[ip], c*Bz[ip],
                    gamma_p,
                    Larmore_factor_density,
                    Larmore_factor_momentum,
                    SR_dxi, SR_xi_data,
                    omega_ax, spect_loc
            )

            for i_omega in range(N_omega):
                spect_loc_omega = spect_loc[i_omega]

                spect_proj_00 = spect_loc_omega * s0_x * s0_y
                cuda.atomic.add(
                    radiation_data, (th_ix, th_iy, i_omega),
                    spect_proj_00
                )

            for i_omega in range(N_omega):
                spect_loc_omega = spect_loc[i_omega]

                spect_proj_10 = spect_loc_omega * s1_x * s0_y
                cuda.atomic.add(
                    radiation_data, (th_ix+1, th_iy, i_omega),
                    spect_proj_10
                )

            for i_omega in range(N_omega):
                spect_loc_omega = spect_loc[i_omega]

                spect_proj_01 = spect_loc_omega * s0_x * s1_y
                cuda.atomic.add(
                    radiation_data, (th_ix, th_iy+1, i_omega),
                    spect_proj_01
                )

            for i_omega in range(N_omega):
                spect_loc_omega = spect_loc[i_omega]

                spect_proj_11 = spect_loc_omega * s1_x * s1_y
                cuda.atomic.add(
                    radiation_data, (th_ix+1, th_iy+1, i_omega),
                    spect_proj_11
                )
