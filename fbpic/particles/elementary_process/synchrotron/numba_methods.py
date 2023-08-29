# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines cuda methods that are used in particle ionization.

Apart from synthactic details, this file is very close to numba_methods.py
"""

import numba
from numba import prange
import random

from scipy.constants import c
# Import inline functions
from .inline_functions import get_angles, get_particle_radiation, \
    get_linear_coefficients

# Compile the inline functions for GPU
get_angles = numba.njit( get_angles)
get_particle_radiation = numba.njit( get_particle_radiation)
get_linear_coefficients = numba.njit( get_linear_coefficients )


@numba.njit(cache=True)
def gather_synchrotron_numba(
    N_tot,
    ux, uy, uz, Ex, Ey, Ez,
    Bx, By, Bz, w, gamma_inv,
    Larmore_factor_density,
    Larmore_factor_momentum,
    gamma_cutoff_inv,
    omega_ax, SR_dxi, SR_xi_data,
    theta_x_min, theta_x_max, d_th_x,
    theta_y_min, theta_y_max, d_th_y,
    spect_loc, radiation_data):
    """
    doc
    """
    for ip in prange( N_tot ):

        if  (gamma_inv[ip] >= gamma_cutoff_inv):
            continue

        theta_x, theta_y = get_angles(
            ux[ip], uy[ip], uz[ip]
        )

        theta_diffusion = 2**-1.5 * gamma_inv[ip]
        theta_x += random.gauss(0, theta_diffusion)
        theta_y += random.gauss(0, theta_diffusion)

        if   (theta_x >= theta_x_max) \
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

        spect_loc = get_particle_radiation(
                ux[ip], uy[ip], uz[ip], w[ip],
                Ex[ip], Ey[ip], Ez[ip],
                c*Bx[ip], c*By[ip], c*Bz[ip],
                gamma_inv[ip],
                Larmore_factor_density,
                Larmore_factor_momentum,
                SR_dxi, SR_xi_data,
                omega_ax, spect_loc
        )

        radiation_data[th_ix, th_iy, :] += spect_loc * s0_x * s0_y
        radiation_data[th_ix, th_iy+1, :] += spect_loc * s0_x * s1_y
        radiation_data[th_ix+1, th_iy, :] += spect_loc * s1_x * s0_y
        radiation_data[th_ix+1, th_iy+1, :] += spect_loc * s1_x * s1_y
