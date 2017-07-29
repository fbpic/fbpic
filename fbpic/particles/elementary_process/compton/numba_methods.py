# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines numba methods that are used in Compton scattering.
"""
import numba
import math
from scipy.constants import c
from fbpic.threading_utils import njit_parallel, prange

@numba.njit
def get_scattering_probability(
    dt, pi_re2, inv_mc, elec_ux, elec_uy, elec_uz, elec_inv_gamma,
    photon_n, photon_p, photon_beta_x, photon_beta_y, photon_beta_z ):
    """
    # TODO:
    Write transformations
    """
    # Get electron intermediate variable
    elec_gamma = 1./elec_inv_gamma

    # Get photon density and momentum in the rest frame of the electron
    transform_factor = elec_gamma \
        - elec_ux*photon_beta_x - elec_uy*photon_beta_y - elec_uz*photon_beta_z
    photon_n_rest = photon_n * transform_factor
    photon_p_rest = photon_p * transform_factor

    # Calculate the Klein-Nishina cross-section
    k = photon_p_rest * inv_mc
    f1 = 2 * ( 2 + k*(1+k)*(8+k) ) / ( k**2 * (1 + 2*k)**2 )
    f2 = ( 2 + k*(2-k) ) * math.log( 1 + 2*k ) / k**3
    sigma = pi_re2 * ( f1 - f2 )
    # Get the electron proper time
    proper_dt_rest = dt * elec_inv_gamma
    # Calculate the probability of scattering
    p = 1 - math.exp( - sigma * photon_n_rest * c * proper_dt_rest )

    return( p )


@njit_parallel
def determine_scatterings_numba( N_batch, batch_size, elec_Ntot,
    does_scatter, n_scatters, random_draw, dt, pi_re2, inv_mc,
    elec_ux, elec_uy, elec_uz, elec_inv_gamma,
    photon_n, photon_p, photon_beta_x, photon_beta_y, photon_beta_z ):
    """
    For each electron macroparticle, decide whether it is going to
    scatter, using the integrated Klein-Nishina formula.

    # TODO: Modify description below
    For the purpose of counting and creating the corresponding electrons,
    `is_ionized` (one element per macroparticle) is set to 1 at the position
    of the ionized ions, and `n_ionized` (one element per batch) counts
    the total number of ionized particles in the current batch.

    # TODO: Parameters

    """
    # Loop over batches of particles (in parallel, if threading is enabled)
    for i_batch in prange( N_batch ):

        # Set the count of scattered particles in the batch to 0
        n_scatters[i_batch] = 0

        # Loop through the batch
        # (Note: a while loop is used here, because numba 0.34 does
        # not support nested prange and range loops)
        N_max = min( (i_batch+1)*batch_size, elec_Ntot )
        ip = i_batch*batch_size
        while ip < N_max:

            # For each electron, calculate the probability of scattering
            p = get_scattering_probability( dt, pi_re2, inv_mc,
                elec_ux[ip], elec_uy[ip], elec_uz[ip], elec_inv_gamma[ip],
                photon_n, photon_p, photon_beta_x, photon_beta_y, photon_beta_z)

            # Determine whether the electron scatters
            if random_draw[ip] < p:
                # Set the corresponding flag and update particle count
                does_scatter[ip] = 1
                n_scatters[i_batch] += 1
            else:
                does_scatter[ip] = 0

            # Increment ip
            ip = ip + 1

    return( does_scatter, n_scatters )

@numba.njit
def scatter_photons_electrons_numba(
    N_batch, batch_size, photon_old_Ntot, elec_Ntot,
    cumulative_n_scatters, does_scatter,
    photon_x, photon_y, photon_z, photon_inv_gamma,
    photon_ux, photon_uy, photon_uz, photon_w,
    photon_Ex, photon_Ey, photon_Ez,
    photon_Bx, photon_By, photon_Bz,
    elec_x, elec_y, elec_z, elec_inv_gamma,
    elec_ux, elec_uy, elec_uz, elec_w,
    elec_Ex, elec_Ey, elec_Ez, elec_Bx, elec_By, elec_Bz ):
    """
    # TODO: so far, this is only copy the photons
    # One should a random additional angle
    """
    #  Loop over batches of particles (in parallel, if threading is enabled)
    for i_batch in range( N_batch ):
        # Photon index: this is incremented each time
        # an scattered photon is identified
        photon_index = photon_old_Ntot + cumulative_n_scatters[i_batch]

        # Loop through the electrons in this batch
        N_max = min( (i_batch+1)*batch_size, elec_Ntot )
        elec_index = i_batch*batch_size
        while elec_index < N_max:
            if does_scatter[elec_index] == 1:
                # Copy the elec data to the current photon_index
                photon_x[photon_index] = elec_x[elec_index]
                photon_y[photon_index] = elec_y[elec_index]
                photon_z[photon_index] = elec_z[elec_index]
                # TODO: For the moment, the momentum of the photon is wrong!
                photon_ux[photon_index] = elec_ux[elec_index]
                photon_uy[photon_index] = elec_uy[elec_index]
                photon_uz[photon_index] = elec_uz[elec_index]
                photon_inv_gamma[photon_index] = elec_inv_gamma[elec_index]
                photon_w[photon_index] = elec_w[elec_index]
                photon_Ex[photon_index] = elec_Ex[elec_index]
                photon_Ey[photon_index] = elec_Ey[elec_index]
                photon_Ez[photon_index] = elec_Ez[elec_index]
                photon_Bx[photon_index] = elec_Bx[elec_index]
                photon_By[photon_index] = elec_By[elec_index]
                photon_Bz[photon_index] = elec_Bz[elec_index]
                # Update the photon index
                photon_index += 1

            # Increment elecron index
            elec_index += 1

    return( photon_x, photon_y, photon_z, photon_inv_gamma,
        photon_ux, photon_uy, photon_uz, photon_w,
        photon_Ex, photon_Ey, photon_Ez, photon_Bx, photon_By, photon_Bz )
