# Copyright 2023, FBPIC contributors
# Authors: Igor A Andriyash, Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines cuda methods that are used in calcualtion of synchrotron radiation.

Apart from synthactic details, this file is very close to cuda_methods.py
"""

from numba import jit, prange
import random
import math
from scipy.constants import c

# Import inline functions
from .inline_functions import get_angles, get_particle_radiation, \
    get_linear_coefficients

# Compile the inline functions for GPU
get_angles = jit( get_angles, nopython=True )
get_particle_radiation = jit( get_particle_radiation, nopython=True )
get_linear_coefficients = jit( get_linear_coefficients, nopython=True  )

@jit(cache=True, parallel=False, forceobj=True)
def gather_synchrotron_numba(
    N_tot,
    ux, uy, uz, Ex, Ey, Ez,
    Bx, By, Bz, w, gamma_inv,
    Larmore_factor_density,
    Larmore_factor_momentum,
    gamma_cutoff_inv, radiation_reaction,
    omega_ax, SR_dxi, SR_xi_data,
    theta_x_min, theta_x_max, d_th_x,
    theta_y_min, theta_y_max, d_th_y,
    spect_loc, radiation_data):
    """
    Calculate spectral-angular density of the energy emitted by
    the particle and add it to the radiation data array.

    Parameters
    ----------
    Ntot: integer
        Total number of particles

    ux, uy, uz, w: floats
        Components momentum and weight of the particle

    Ex, Ey, Ez: float
         Components of electric field on the particle (V/m)

    cBx, cBy, cBz: float
         Components of magnetic field on the particle multiplied by
         the speed of light (V/m)

    gamma_inv: float
        Reciprocal of particle Lorentz factor

    Larmore_factor_density: float
        Normalization factor for spectral-angular density,
        `e**2 * dt / (6 * np.pi * epsilon_0 * c * hbar * d_theta_x * d_theta_y)`

    Larmore_factor_momentum: float
        Normalization factor for the photon momentum,
        `e**2 * dt / ( 6 * np.pi * epsilon_0 * c**2 )`

    gamma_cutoff_inv: float
        Reciprocal of the Lorentz factor below which particles are discarded

    radiation_reaction: bool
        Whether to consider radiation reaction on the electrons

    omega_ax: 1D vector of floats
        frequencies on which spectrum is calculated

    SR_dxi: float
        Sampling step of the spectral profile function

    SR_xi_data: 1D vector of floats
        Sampling of the spectral profile function

    theta_x_min: float
        Lower limit of the `theta_x` angle axis

    theta_x_max: float
        Upper limit of the `theta_x` angle axis

    d_th_x: float
        Step of the `theta_x` angle axis

    theta_y_min: float
        Lower limit of the `theta_y` angle axis

    theta_y_max: float
        Upper limit of the `theta_y` angle axis

    d_th_y: float
        Step of the `theta_y` angle axis

    spect_loc: 1D array of floats
        Array for spectral profile of the particle

    radiation_data: 3D array of floats
        Global radiation data
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

        spect_loc, ux_ph, uy_ph, uz_ph = get_particle_radiation(
                ux[ip], uy[ip], uz[ip], w[ip],
                Ex[ip], Ey[ip], Ez[ip],
                c*Bx[ip], c*By[ip], c*Bz[ip],
                gamma_inv[ip],
                Larmore_factor_density,
                Larmore_factor_momentum,
                SR_dxi, SR_xi_data,
                omega_ax, spect_loc
        )

        if radiation_reaction:
            ux[ip] -= ux_ph
            uy[ip] -= uy_ph
            uz[ip] -= uz_ph
            gamma_inv[ip] = 1 / math.sqrt(
                1.0 + ux[ip] * ux[ip] + uy[ip] * uy[ip] + uz[ip] * uz[ip]
            )

        radiation_data[th_ix, th_iy, :] += spect_loc * s0_x * s0_y
        radiation_data[th_ix, th_iy+1, :] += spect_loc * s0_x * s1_y
        radiation_data[th_ix+1, th_iy, :] += spect_loc * s1_x * s0_y
        radiation_data[th_ix+1, th_iy+1, :] += spect_loc * s1_x * s1_y
