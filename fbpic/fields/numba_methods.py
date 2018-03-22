# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the optimized fields methods that use numba on a CPU
"""
from scipy.constants import c, epsilon_0, mu_0
c2 = c**2
from fbpic.utils.threading import njit_parallel, prange

@njit_parallel
def numba_correct_currents_curlfree_standard( rho_prev, rho_next, Jp, Jm, Jz,
                            kz, kr, inv_k2, inv_dt, Nz, Nr ):
    """
    Correct the currents in spectral space, using the curl-free correction
    which is adapted to the standard psatd
    """
    # Loop over the 2D grid (parallel in z, if threading is installed)
    for iz in prange(Nz):
        for ir in range(Nr):

            # Calculate the intermediate variable F
            F = - inv_k2[iz, ir] * (
                (rho_next[iz, ir] - rho_prev[iz, ir])*inv_dt \
                + 1.j*kz[iz, ir]*Jz[iz, ir] \
                + kr[iz, ir]*( Jp[iz, ir] - Jm[iz, ir] ) )

            # Correct the currents accordingly
            Jp[iz, ir] +=  0.5 * kr[iz, ir] * F
            Jm[iz, ir] += -0.5 * kr[iz, ir] * F
            Jz[iz, ir] += -1.j * kz[iz, ir] * F

    return

@njit_parallel
def numba_correct_currents_crossdeposition_standard( rho_prev, rho_next,
        rho_next_z, rho_next_xy, Jp, Jm, Jz, kz, kr, inv_dt, Nz, Nr ):
    """
    Correct the currents in spectral space, using the cross-deposition
    algorithm adapted to the standard psatd.
    """
    # Loop over the 2D grid
    for iz in prange(Nz):
        # Loop through the radial points
        # (Note: a while loop is used here, because numba 0.34 does
        # not support nested prange and range loops)
        ir = 0
        while ir < Nr:

            # Calculate the intermediate variable Dz and Dxy
            # (Such that Dz + Dxy is the error in the continuity equation)
            Dz = 1.j*kz[iz, ir]*Jz[iz, ir] + 0.5 * inv_dt * \
                ( rho_next[iz, ir] - rho_next_xy[iz, ir] + \
                  rho_next_z[iz, ir] - rho_prev[iz, ir] )
            Dxy = kr[iz, ir]*( Jp[iz, ir] - Jm[iz, ir] ) + 0.5 * inv_dt * \
                ( rho_next[iz, ir] - rho_next_z[iz, ir] + \
                  rho_next_xy[iz, ir] - rho_prev[iz, ir] )

            # Correct the currents accordingly
            if kr[iz, ir] != 0:
                inv_kr = 1./kr[iz, ir]
                Jp[iz, ir] += -0.5 * Dxy * inv_kr
                Jm[iz, ir] +=  0.5 * Dxy * inv_kr
            if kz[iz, ir] != 0:
                inv_kz = 1./kz[iz, ir]
                Jz[iz, ir] += 1.j * Dz * inv_kz

            # Increment ir
            ir += 1

    return

@njit_parallel
def numba_push_eb_standard( Ep, Em, Ez, Bp, Bm, Bz, Jp, Jm, Jz,
                       rho_prev, rho_next,
                       rho_prev_coef, rho_next_coef, j_coef,
                       C, S_w, kr, kz, dt,
                       use_true_rho, Nz, Nr) :
    """
    Push the fields over one timestep, using the standard psatd algorithm

    See the documentation of SpectralGrid.push_eb_with
    """
    # Loop over the 2D grid (parallel in z, if threading is installed)
    for iz in prange(Nz):
        for ir in range(Nr):

            # Save the electric fields, since it is needed for the B push
            Ep_old = Ep[iz, ir]
            Em_old = Em[iz, ir]
            Ez_old = Ez[iz, ir]

            # Calculate useful auxiliary arrays
            if use_true_rho:
                # Evaluation using the rho projected on the grid
                rho_diff = rho_next_coef[iz, ir] * rho_next[iz, ir] \
                        - rho_prev_coef[iz, ir] * rho_prev[iz, ir]
            else:
                # Evaluation using div(E) and div(J)
                divE = kr[iz, ir]*( Ep[iz, ir] - Em[iz, ir] ) \
                    + 1.j*kz[iz, ir]*Ez[iz, ir]
                divJ = kr[iz, ir]*( Jp[iz, ir] - Jm[iz, ir] ) \
                    + 1.j*kz[iz, ir]*Jz[iz, ir]

                rho_diff = (rho_next_coef[iz, ir] - rho_prev_coef[iz, ir]) \
                  * epsilon_0 * divE - rho_next_coef[iz, ir] * dt * divJ

            # Push the E field
            Ep[iz, ir] = C[iz, ir]*Ep[iz, ir] + 0.5*kr[iz, ir]*rho_diff \
                + c2*S_w[iz, ir]*( -1.j*0.5*kr[iz, ir]*Bz[iz, ir] \
                + kz[iz, ir]*Bp[iz, ir] - mu_0*Jp[iz, ir] )

            Em[iz, ir] = C[iz, ir]*Em[iz, ir] - 0.5*kr[iz, ir]*rho_diff \
                + c2*S_w[iz, ir]*( -1.j*0.5*kr[iz, ir]*Bz[iz, ir] \
                - kz[iz, ir]*Bm[iz, ir] - mu_0*Jm[iz, ir] )

            Ez[iz, ir] = C[iz, ir]*Ez[iz, ir] - 1.j*kz[iz, ir]*rho_diff \
                + c2*S_w[iz, ir]*( 1.j*kr[iz, ir]*Bp[iz, ir] \
                + 1.j*kr[iz, ir]*Bm[iz, ir] - mu_0*Jz[iz, ir] )

            # Push the B field
            Bp[iz, ir] = C[iz, ir]*Bp[iz, ir] \
                - S_w[iz, ir]*( -1.j*0.5*kr[iz, ir]*Ez_old \
                            + kz[iz, ir]*Ep_old ) \
                + j_coef[iz, ir]*( -1.j*0.5*kr[iz, ir]*Jz[iz, ir] \
                            + kz[iz, ir]*Jp[iz, ir] )

            Bm[iz, ir] = C[iz, ir]*Bm[iz, ir] \
                - S_w[iz, ir]*( -1.j*0.5*kr[iz, ir]*Ez_old \
                            - kz[iz, ir]*Em_old ) \
                + j_coef[iz, ir]*( -1.j*0.5*kr[iz, ir]*Jz[iz, ir] \
                            - kz[iz, ir]*Jm[iz, ir] )

            Bz[iz, ir] = C[iz, ir]*Bz[iz, ir] \
                - S_w[iz, ir]*( 1.j*kr[iz, ir]*Ep_old \
                            + 1.j*kr[iz, ir]*Em_old ) \
                + j_coef[iz, ir]*( 1.j*kr[iz, ir]*Jp[iz, ir] \
                            + 1.j*kr[iz, ir]*Jm[iz, ir] )

    return

@njit_parallel
def numba_correct_currents_curlfree_comoving( rho_prev, rho_next, Jp, Jm, Jz,
                            kz, kr, inv_k2,
                            j_corr_coef, T_eb, T_cc,
                            inv_dt, Nz, Nr ) :
    """
    Correct the currents in spectral space, using the curl-free correction
    which is adapted to the galilean/comoving-currents assumption
    """
    # Loop over the 2D grid (parallel in z, if threading is installed)
    for iz in prange(Nz):
        for ir in range(Nr):

            # Calculate the intermediate variable F
            F =  - inv_k2[iz, ir] * ( T_cc[iz, ir]*j_corr_coef[iz, ir] \
                * (rho_next[iz, ir] - rho_prev[iz, ir]*T_eb[iz, ir]) \
                + 1.j*kz[iz, ir]*Jz[iz, ir] \
                + kr[iz, ir]*( Jp[iz, ir] - Jm[iz, ir] ) )

            # Correct the currents accordingly
            Jp[iz, ir] +=  0.5 * kr[iz, ir] * F
            Jm[iz, ir] += -0.5 * kr[iz, ir] * F
            Jz[iz, ir] += -1.j * kz[iz, ir] * F

    return

@njit_parallel
def numba_correct_currents_crossdeposition_comoving(
        rho_prev, rho_next, rho_next_z, rho_next_xy, Jp, Jm, Jz,
        kz, kr, j_corr_coef, T_eb, T_cc, inv_dt, Nz, Nr ) :
    """
    Correct the currents in spectral space, using the cross-deposition
    algorithm adapted to the galilean/comoving-currents assumption.
    """
    # Loop over the 2D grid
    for iz in prange(Nz):
        # Loop through the radial points
        # (Note: a while loop is used here, because numba 0.34 does
        # not support nested prange and range loops)
        ir = 0
        while ir < Nr:

            # Calculate the intermediate variable Dz and Dxy
            # (Such that Dz + Dxy is the error in the continuity equation)

            Dz = 1.j*kz[iz, ir]*Jz[iz, ir] \
                + 0.5 * T_cc[iz, ir]*j_corr_coef[iz, ir] * \
                ( rho_next[iz, ir] - T_eb[iz, ir] * rho_next_xy[iz, ir] \
                  + rho_next_z[iz, ir] - T_eb[iz, ir] * rho_prev[iz, ir] )
            Dxy = kr[iz, ir]*( Jp[iz, ir] - Jm[iz, ir] ) \
                + 0.5 * T_cc[iz, ir]*j_corr_coef[iz, ir] * \
                ( rho_next[iz, ir] + T_eb[iz, ir] * rho_next_xy[iz, ir] \
                - rho_next_z[iz, ir] -  T_eb[iz, ir] * rho_prev[iz, ir] )

            # Correct the currents accordingly
            if kr[iz, ir] != 0:
                inv_kr = 1./kr[iz, ir]
                Jp[iz, ir] += -0.5 * Dxy * inv_kr
                Jm[iz, ir] +=  0.5 * Dxy * inv_kr
            if kz[iz, ir] != 0:
                inv_kz = 1./kz[iz, ir]
                Jz[iz, ir] += 1.j * Dz * inv_kz

            # Increment ir
            ir += 1

    return

@njit_parallel
def numba_push_eb_comoving( Ep, Em, Ez, Bp, Bm, Bz, Jp, Jm, Jz,
                       rho_prev, rho_next,
                       rho_prev_coef, rho_next_coef, j_coef,
                       C, S_w, T_eb, T_cc, T_rho,
                       kr, kz, dt, V, use_true_rho, Nz, Nr):
    """
    Push the fields over one timestep, using the psatd algorithm,
    with the assumptions of comoving currents
    (either with the galilean scheme or comoving scheme, depending on
    the values of the coefficients that are passed)

    See the documentation of SpectralGrid.push_eb_with
    """
    # Loop over the grid (parallel in z, if threading is installed)
    for iz in prange(Nz):
        for ir in range(Nr):

            # Save the electric fields, since it is needed for the B push
            Ep_old = Ep[iz, ir]
            Em_old = Em[iz, ir]
            Ez_old = Ez[iz, ir]

            # Calculate useful auxiliary arrays
            if use_true_rho:
                # Evaluation using the rho projected on the grid
                rho_diff = rho_next_coef[iz, ir] * rho_next[iz, ir] \
                        - rho_prev_coef[iz, ir] * rho_prev[iz, ir]
            else:
                # Evaluation using div(E) and div(J)
                divE = kr[iz, ir]*( Ep[iz, ir] - Em[iz, ir] ) \
                    + 1.j*kz[iz, ir]*Ez[iz, ir]
                divJ = kr[iz, ir]*( Jp[iz, ir] - Jm[iz, ir] ) \
                    + 1.j*kz[iz, ir]*Jz[iz, ir]

                rho_diff = ( T_eb[iz,ir] * rho_next_coef[iz, ir] \
                  - rho_prev_coef[iz, ir] ) \
                  * epsilon_0 * divE + T_rho[iz, ir] \
                  * rho_next_coef[iz, ir] * divJ

            # Push the E field
            Ep[iz, ir] = \
                T_eb[iz, ir]*C[iz, ir]*Ep[iz, ir] + 0.5*kr[iz, ir]*rho_diff \
                + j_coef[iz, ir]*1.j*kz[iz, ir]*V*Jp[iz, ir] \
                + c2*T_eb[iz, ir]*S_w[iz, ir]*( -1.j*0.5*kr[iz, ir]*Bz[iz, ir] \
                + kz[iz, ir]*Bp[iz, ir] - mu_0*T_cc[iz, ir]*Jp[iz, ir] )

            Em[iz, ir] = \
                T_eb[iz, ir]*C[iz, ir]*Em[iz, ir] - 0.5*kr[iz, ir]*rho_diff \
                + j_coef[iz, ir]*1.j*kz[iz, ir]*V*Jm[iz, ir] \
                + c2*T_eb[iz, ir]*S_w[iz, ir]*( -1.j*0.5*kr[iz, ir]*Bz[iz, ir] \
                - kz[iz, ir]*Bm[iz, ir] - mu_0*T_cc[iz, ir]*Jm[iz, ir] )

            Ez[iz, ir] = \
                T_eb[iz, ir]*C[iz, ir]*Ez[iz, ir] - 1.j*kz[iz, ir]*rho_diff \
                + j_coef[iz, ir]*1.j*kz[iz, ir]*V*Jz[iz, ir] \
                + c2*T_eb[iz, ir]*S_w[iz, ir]*( 1.j*kr[iz, ir]*Bp[iz, ir] \
                + 1.j*kr[iz, ir]*Bm[iz, ir] - mu_0*T_cc[iz, ir]*Jz[iz, ir] )

            # Push the B field
            Bp[iz, ir] = T_eb[iz, ir]*C[iz, ir]*Bp[iz, ir] \
                - T_eb[iz, ir]*S_w[iz, ir]*( -1.j*0.5*kr[iz, ir]*Ez_old \
                            + kz[iz, ir]*Ep_old ) \
                + j_coef[iz, ir]*( -1.j*0.5*kr[iz, ir]*Jz[iz, ir] \
                            + kz[iz, ir]*Jp[iz, ir] )

            Bm[iz, ir] = T_eb[iz, ir]*C[iz, ir]*Bm[iz, ir] \
                - T_eb[iz, ir]*S_w[iz, ir]*( -1.j*0.5*kr[iz, ir]*Ez_old \
                            - kz[iz, ir]*Em_old ) \
                + j_coef[iz, ir]*( -1.j*0.5*kr[iz, ir]*Jz[iz, ir] \
                            - kz[iz, ir]*Jm[iz, ir] )

            Bz[iz, ir] = T_eb[iz, ir]*C[iz, ir]*Bz[iz, ir] \
                - T_eb[iz, ir]*S_w[iz, ir]*( 1.j*kr[iz, ir]*Ep_old \
                            + 1.j*kr[iz, ir]*Em_old ) \
                + j_coef[iz, ir]*( 1.j*kr[iz, ir]*Jp[iz, ir] \
                            + 1.j*kr[iz, ir]*Jm[iz, ir] )

    return
