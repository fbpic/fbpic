# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines numba methods that are used in Compton scattering (on CPU).
"""
import numba
import math, random
from fbpic.utils.threading import njit_parallel, prange
# Import the inline functions
from .inline_functions import lorentz_transform, get_scattering_probability, \
    get_photon_density_gaussian, INV_MC
# Compile the inline functions for CPU
lorentz_transform = numba.njit( lorentz_transform )
get_scattering_probability = numba.njit( get_scattering_probability )
get_photon_density_gaussian = numba.njit( get_photon_density_gaussian )

@njit_parallel
def get_photon_density_gaussian_numba( photon_n, elec_Ntot,
    elec_x, elec_y, elec_z, ct, photon_n_lab_max, inv_laser_waist2,
    inv_laser_ctau2, laser_initial_z0, gamma_boost, beta_boost ):
    """
    Fill the array `photon_n` with the values of the photon density
    (in the simulation frame) in the scattering laser pulse, at
    the position of the electron macroparticles.

    Parameters
    ----------
    elec_x, elec_y, elec_z: 1d arrays of floats
        The position of the electrons (in the frame of the simulation)
    ct: float
        Current time in the simulation frame (multiplied by c)
    photon_n_lab_max: float
        Peak photon density (in the lab frame)
        (i.e. at the peak of the Gaussian pulse)
    inv_laser_waist2, inv_laser_ctau2, laser_initial_z0: floats
        Properties of the Gaussian laser pulse (in the lab frame)
    gamma_boost, beta_boost: floats
        Properties of the Lorentz boost between the lab and simulation frame.
    """
    # Loop over electrons (in parallel, if threading is enabled)
    for i_elec in prange( elec_Ntot ):

        photon_n[i_elec] = get_photon_density_gaussian(
            elec_x[i_elec], elec_y[i_elec], elec_z[i_elec], ct,
            photon_n_lab_max, inv_laser_waist2, inv_laser_ctau2,
            laser_initial_z0, gamma_boost, beta_boost )

    return( photon_n )


@njit_parallel
def determine_scatterings_numba( N_batch, batch_size, elec_Ntot,
    nscatter_per_elec, nscatter_per_batch, dt,
    elec_ux, elec_uy, elec_uz, elec_inv_gamma, ratio_w_electron_photon,
    photon_n, photon_p, photon_beta_x, photon_beta_y, photon_beta_z ):
    """
    For each electron macroparticle, decide how many photon macroparticles
    it will emit during `dt`, using the integrated Klein-Nishina formula.

    Note: this function uses a random generator within a `prange` loop.
    This implies that an indenpendent seed and random generator will be
    created for each thread.

    Electrons are processed in batches of size `batch_size`, with a parallel
    loop over batches. The batching allows quicker calculation of the
    total number of photons to be created.
    """
    # Loop over batches of particles (in parallel, if threading is enabled)
    for i_batch in prange( N_batch ):

        # Set the count of scattered particles in the batch to 0
        nscatter_per_batch[i_batch] = 0

        # Loop through the batch
        # (Note: a while loop is used here, because numba 0.34 does
        # not support nested prange and range loops)
        N_max = min( (i_batch+1)*batch_size, elec_Ntot )
        ip = i_batch*batch_size
        while ip < N_max:

            # Set the count of scattered photons for this electron to 0
            nscatter_per_elec[ip] = 0

            # For each electron, calculate the probability of scattering
            p = get_scattering_probability( dt, elec_ux[ip], elec_uy[ip],
                elec_uz[ip], elec_inv_gamma[ip], photon_n[ip],
                photon_p, photon_beta_x, photon_beta_y, photon_beta_z )

            # Determine the number of photons produced by this electron
            nscatter = int( p * ratio_w_electron_photon + random.random() )
            # Note: if p is 0, the above formula will return nscatter=0
            # since random_draw is in [0, 1). Similarly, if p is very small,
            # nscatter will be 1 with probabiliy p * ratio_w_electron_photon,
            # and 0 otherwise.
            nscatter_per_elec[ip] = nscatter
            nscatter_per_batch[i_batch] += nscatter

            # Increment ip
            ip = ip + 1

    return( nscatter_per_elec, nscatter_per_batch )


@njit_parallel
def scatter_photons_electrons_numba(
    N_batch, batch_size, photon_old_Ntot, elec_Ntot,
    cumul_nscatter_per_batch, nscatter_per_elec,
    photon_p, photon_px, photon_py, photon_pz,
    photon_x, photon_y, photon_z, photon_inv_gamma,
    photon_ux, photon_uy, photon_uz, photon_w,
    elec_x, elec_y, elec_z, elec_inv_gamma,
    elec_ux, elec_uy, elec_uz, elec_w, inv_ratio_w_elec_photon ):
    """
    Given the number of photons that are emitted by each electron
    macroparticle, determine the properties (momentum, energy) of
    each scattered photon and fill the arrays `photon_*` accordingly.

    Also, apply a recoil on the electrons.

    Note: this function uses a random generator within a `prange` loop.
    This implies that an indenpendent seed and random generator will be
    created for each thread.
    """
    #  Loop over batches of particles
    for i_batch in prange( N_batch ):

        # Photon index: this is incremented each time
        # a scattered photon is identified
        i_photon = photon_old_Ntot + cumul_nscatter_per_batch[i_batch]

        # Loop through the electrons in this batch
        N_max = min( (i_batch+1)*batch_size, elec_Ntot )
        for i_elec in range( i_batch*batch_size, N_max ):

            # Prepare calculation of scattered photons from this electron
            if nscatter_per_elec[i_elec] > 0:

                # Prepare Lorentz transformation to the electron rest frame
                elec_gamma = 1./elec_inv_gamma[i_elec]
                elec_u = math.sqrt(
                  elec_ux[i_elec]**2 + elec_uy[i_elec]**2 + elec_uz[i_elec]**2)
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
                if cos_theta**2 < 1:
                    sin_theta = math.sqrt( 1 - cos_theta**2 )
                    inv_photon_rest_pxy = 1./( sin_theta * photon_rest_p )
                    cos_phi = photon_rest_px * inv_photon_rest_pxy
                    sin_phi = photon_rest_py * inv_photon_rest_pxy
                else:
                    sin_theta = 0
                    # Avoid division by 0; provide arbitrary direction
                    # for the phi angle (since theta is 0 or pi anyway)
                    cos_phi = 1.
                    sin_phi = 0.

            # Loop through the number of scatterings for this electron
            for i_scat in range(nscatter_per_elec[i_elec]):

                # Draw scattering angle in the rest frame, from the
                # Klein-Nishina cross-section (See Ozmutl, E. N.
                # "Sampling of Angular Distribution in Compton Scattering"
                # Appl. Radiat. Isot. 43, 6, pp. 713-715 (1992))
                k = photon_rest_p * INV_MC
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
                phi_s = 2*math.pi*random.random()
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

                # Create the new photon by copying the electron position
                photon_x[i_photon] = elec_x[i_elec]
                photon_y[i_photon] = elec_y[i_elec]
                photon_z[i_photon] = elec_z[i_elec]
                photon_w[i_photon] = elec_w[i_elec] * inv_ratio_w_elec_photon
                # The photon's ux, uy, uz corresponds to the actual px, py, pz
                photon_ux[i_photon] = new_photon_px
                photon_uy[i_photon] = new_photon_py
                photon_uz[i_photon] = new_photon_pz
                # The photon's inv_gamma corresponds to 1./p (consistent
                # with the code for the particle pusher and for the
                # openPMD back-transformed diagnostics)
                photon_inv_gamma[i_photon] = 1./new_photon_p

                # Update the photon index
                i_photon += 1

            # Add recoil to electrons
            # Note: In order to reproduce the right distribution of electron
            # momentum, the electrons should recoil with the momentum
            # of *one single* photon, with a probability p (calculated by
            # get_scattering_probability). Here we reuse the momentum of
            # the last photon generated above. This requires that at least one
            # photon be created for this electron, which occurs with a
            # probability p*ratio_w_elec_photon. Thus, given that at least one
            # photon has been created, we should add recoil to the corresponding
            # electron only with a probability inv_ratio_w_elec_photon.
            if nscatter_per_elec[i_elec] > 0:
                if random.random() < inv_ratio_w_elec_photon:
                    elec_ux[i_elec] += INV_MC * (photon_px - new_photon_px)
                    elec_uy[i_elec] += INV_MC * (photon_py - new_photon_py)
                    elec_uz[i_elec] += INV_MC * (photon_pz - new_photon_pz)
