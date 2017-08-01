# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines numba methods that are used in Compton scattering.
"""
import numba
import math, random
from scipy.constants import c
from fbpic.threading_utils import njit_parallel, prange

@numba.njit
def lorentz_transform( p_in, px_in, py_in, pz_in, gamma, beta, nx, ny, nz ):
    """
    Perform a Lorentz transform of the 4-momentum (p_in, px_in, py_in, pz_in)
    and return the results

    Parameters
    ----------
    p_in, px_in, py_in, pz_in: floats
        The coordinates of the 4-momentum
    gamma, beta: floats
        Lorentz factor and corresponding beta of the Lorentz transform
    nx, ny, nz: floats
        Coordinates of *normalized* vector that indicates
        the direction of the transform
    """
    p_parallel_in = nx*px_in + ny*py_in + nz*pz_in
    p_out = gamma * ( p_in - beta * p_parallel_in )
    p_parallel_out = gamma * ( p_parallel_in - beta * p_in )

    px_out = px_in + nx * ( p_parallel_out - p_parallel_in )
    py_out = py_in + ny * ( p_parallel_out - p_parallel_in )
    pz_out = pz_in + nz * ( p_parallel_out - p_parallel_in )

    return( p_out, px_out, py_out, pz_out )

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

# Note: This routine is necessarily serial on CPU, since there is
# no available thread-safe implementation of on-the-fly random generator.
@numba.njit
def scatter_photons_electrons_numba(
    N_batch, batch_size, photon_old_Ntot, elec_Ntot,
    cumulative_n_scatters, does_scatter, inv_mc,
    photon_p, photon_px, photon_py, photon_pz,
    photon_x, photon_y, photon_z, photon_inv_gamma,
    photon_ux, photon_uy, photon_uz, photon_w,
    elec_x, elec_y, elec_z, elec_inv_gamma,
    elec_ux, elec_uy, elec_uz, elec_w ):
    """
    # TODO: so far, this is only copy the photons
    # One should a random additional angle
    """
    #  Loop over batches of particles
    for i_batch in range( N_batch ):

        # Photon index: this is incremented each time
        # a scattered photon is identified
        i_photon = photon_old_Ntot + cumulative_n_scatters[i_batch]

        # Loop through the electrons in this batch
        N_max = min( (i_batch+1)*batch_size, elec_Ntot )
        for i_elec in range( i_batch*batch_size, N_max ):

            if does_scatter[i_elec] == 1:

                # Prepare Lorentz transformation to the electron rest frame
                elec_gamma = 1./elec_inv_gamma[i_elec]
                elec_u = math.sqrt( elec_gamma**2 - 1. )
                elec_beta = elec_u * elec_inv_gamma[i_elec]
                if elec_u != 0:
                    elec_inv_u = 1./elec_u
                    elec_nx = elec_inv_u * elec_ux[i_elec]
                    elec_ny = elec_inv_u * elec_uy[i_elec]
                    elec_nz = elec_inv_u * elec_uz[i_elec]
                else:
                    # Avoid division by 0; provide arbitrary direction
                    # for the Lorentz transform (since beta=0 anyway)
                    elec_nx = 0.
                    elec_ny = 0.
                    elec_nz = 1.

                # Transform momentum of photon to the electron rest frame
                photon_rest_p, photon_rest_px, \
                    photon_rest_py, photon_rest_pz = lorentz_transform(
                            photon_p, photon_px, photon_py, photon_pz,
                            elec_gamma, elec_beta, elec_nx, elec_ny, elec_nz )

                # Find cos and sin of the spherical angle that represent
                # the direction of the incoming photon in the rest frame
                cos_theta = photon_rest_pz/photon_rest_p
                sin_theta = math.sqrt( 1 - cos_theta**2 )
                if sin_theta != 0:
                    inv_photon_rest_pxy = 1./( sin_theta * photon_rest_p )
                    cos_phi = photon_rest_px * inv_photon_rest_pxy
                    sin_phi = photon_rest_py * inv_photon_rest_pxy
                else:
                    # Avoid division by 0; provide arbitrary direction
                    # for the phi angle (since theta is 0 or pi anyway)
                    cos_phi = 1.
                    sin_phi = 0.

                # Draw scattering angle in the rest frame, from the
                # Klein-Nishina cross-section (See Ozmutl, E. N.
                # "Sampling of Angular Distribution in Compton Scattering"
                # Appl. Radiat. Isot. 43, 6, pp. 713-715 (1992))
                k = photon_rest_p * inv_mc
                c0 = 2.*(2.*k**2 + 2.*k + 1.)/(2.*k + 1.)**3
                b = (2. + c0)/(2. - c0)
                a = 2.*b - 1.
                # Use rejection method to draw x
                reject = True
                while reject:
                    # - Draw x with an approximate probability distribution
                    r1 = random.random()
                    x = b - (b + 1.)*(0.5*c0)**r1
                    # - Calculate approximate probability distribution h
                    h = a/(b-x)
                    # - Calculate expected (exact) probability distribution f
                    factor = 1 + k*(1-x)
                    f = ( (1+x**2)*factor + k**2*(1-x)**2 )/factor**3
                    # - Keep x according to rejection rule
                    r2 = random.random()
                    if r2 < f/h:
                        reject = False

                # Get scattered momentum in the rest frame
                new_photon_rest_p = photon_rest_p/( 1 + k*(1-x) )
                # - First in a system of axes aligned with the incoming photon
                cos_theta_s = x
                sin_theta_s = math.sqrt( 1 - x**2 )
                phi_s = random.random()
                cos_phi_s = math.cos( phi_s )
                sin_phi_s = math.sin( phi_s )
                new_photon_rest_pX = new_photon_rest_p * sin_theta_s*cos_phi_s
                new_photon_rest_pY = new_photon_rest_p * sin_theta_s*sin_phi_s
                new_photon_rest_pZ = new_photon_rest_p * cos_theta_s
                # - Then rotate it to the original system of axes
                new_photon_rest_px = sin_theta * cos_phi * new_photon_rest_pZ \
                                   + cos_theta * cos_phi * new_photon_rest_pX \
                                               - sin_phi * new_photon_rest_pY
                new_photon_rest_py = sin_theta * sin_phi * new_photon_rest_pZ \
                                   + cos_theta * sin_phi * new_photon_rest_pX \
                                               + cos_phi * new_photon_rest_pY
                new_photon_rest_pz = cos_theta * new_photon_rest_pZ \
                                   - sin_theta * new_photon_rest_pX

                # Transform momentum of photon back to the simulation frame
                # (i.e. Lorentz transform with opposite direction)
                new_photon_p, new_photon_px, new_photon_py, new_photon_pz = \
                    lorentz_transform(
                        new_photon_rest_p, new_photon_rest_px,
                        new_photon_rest_py, new_photon_rest_pz,
                        elec_gamma, elec_beta, -elec_nx, -elec_ny, -elec_nz)

                # Create the new photon by copying the electron data
                photon_x[i_photon] = elec_x[i_elec]
                photon_y[i_photon] = elec_y[i_elec]
                photon_z[i_photon] = elec_z[i_elec]
                # The photon's ux, uy, uz corresponds to the actual px, py, pz
                photon_ux[i_photon] = new_photon_px
                photon_uy[i_photon] = new_photon_py
                photon_uz[i_photon] = new_photon_pz
                photon_inv_gamma[i_photon] = 1./new_photon_p
                photon_w[i_photon] = elec_w[i_elec]

                # Update the photon index
                i_photon += 1
