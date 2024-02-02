# Copyright 2023, FBPIC contributors
# Authors: Igor A Andriyash, Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines cuda methods that are used in calcualtion of synchrotron radiation.

Apart from synthactic details, this file is very close to numba_methods.py
"""

from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float64
from scipy.constants import c
import math

# Import inline functions
from .inline_functions import get_angles, get_particle_radiation, \
    get_linear_coefficients

# Compile the inline functions for GPU
get_angles = cuda.jit( get_angles, device=True, inline=True)
get_particle_radiation = cuda.jit( get_particle_radiation, device=True, inline=True )
get_linear_coefficients = cuda.jit( get_linear_coefficients, device=True, inline=True )

@cuda.jit
def gather_synchrotron_cuda(
    N_batch, batch_size, Ntot,
    ux, uy, uz, Ex, Ey, Ez,
    Bx, By, Bz, w, gamma_inv,
    Larmore_factor_density,
    Larmore_factor_momentum,
    gamma_cutoff_inv, radiation_reaction,
    omega_ax, SR_dxi, SR_xi_data,
    theta_x_min, theta_x_max, d_th_x,
    theta_y_min, theta_y_max, d_th_y,
    spect_batch, rng_states_batch, radiation_data):
    """
    Calculate spectral-angular density of the energy emitted by
    the particle and add it to the radiation data array.

    Parameters
    ----------
    N_batch: integer
        Total number of batches

    batch_size: integer
        Number of  particles in the current

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

    spect_batch: 2D array of floats
        Array for spectral profiles of particles in the batch

    rng_states_batch: rng_states
        States for random number generator

    radiation_data: 3D array of floats
        Global radiation data
    """
    # Loop over batches of particles
    i_batch = cuda.grid(1)
    if i_batch < N_batch:
        # Loop through the batch
        spect_loc = spect_batch[i_batch]
        N_omega = spect_loc.size

        N_max = min( (i_batch+1)*batch_size, Ntot )
        for ip in range( i_batch*batch_size, N_max ):

            if (gamma_inv[ip] >= gamma_cutoff_inv):
                continue

            theta_x, theta_y = get_angles( ux[ip], uy[ip], uz[ip] )

            theta_diffusion = 2**-1.5 * gamma_inv[ip]
            theta_x += theta_diffusion * xoroshiro128p_normal_float64(
                rng_states_batch, i_batch)
            theta_y += theta_diffusion * xoroshiro128p_normal_float64(
                rng_states_batch, i_batch)

            if   (theta_x >= theta_x_max) \
              or (theta_y >= theta_y_max) \
              or (theta_x <= theta_x_min) \
              or (theta_y <= theta_y_min):
                continue

            th_ix, s0_x, s1_x = get_linear_coefficients(
                theta_x, theta_x_min, d_th_x )
            th_iy, s0_y, s1_y = get_linear_coefficients(
                theta_y, theta_y_min, d_th_y )

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
