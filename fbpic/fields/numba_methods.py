# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the optimized fields methods that use numba on a CPU
"""
import numba
from scipy.constants import c, epsilon_0, mu_0
c2 = c**2

@numba.jit('void(complex128[:,:], complex128[:,:], \
           complex128[:,:], complex128[:,:], complex128[:,:], \
           float64[:,:], float64[:,:], float64[:,:], \
           float64, int32, int32)')
def numba_correct_currents_standard( rho_prev, rho_next, Jp, Jm, Jz,
                            kz, kr, inv_k2, inv_dt, Nz, Nr ):
    """
    Correct the currents in spectral space, using the standard pstad
    """
    # Loop over the 2D grid
    for iz in range(Nz):
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

@numba.jit('void(complex128[:,:], complex128[:,:], complex128[:,:], \
           complex128[:,:], complex128[:,:], complex128[:,:], \
           complex128[:,:], complex128[:,:], complex128[:,:], \
           complex128[:,:], complex128[:,:], \
           float64[:,:], float64[:,:], float64[:,:], \
           float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64, \
           int8, int32, int32)')
def numba_push_eb_standard( Ep, Em, Ez, Bp, Bm, Bz, Jp, Jm, Jz,
                       rho_prev, rho_next,
                       rho_prev_coef, rho_next_coef, j_coef,
                       C, S_w, kr, kz, dt,
                       use_true_rho, Nz, Nr) :
    """
    Push the fields over one timestep, using the standard psatd algorithm

    See the documentation of SpectralGrid.push_eb_with
    """
    # Loop over the 2D grid
    for iz in range(Nz):
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

@numba.jit
def numba_correct_currents_comoving( rho_prev, rho_next, Jp, Jm, Jz,
                            kz, kr, inv_k2,
                            j_corr_coef, T_eb, T_cc,
                            inv_dt, Nz, Nr ) :
    """
    Correct the currents in spectral space, using the assumption
    of comoving currents
    """
    # Loop over the 2D grid
    for iz in range(Nz):
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

@numba.jit
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
    # Loop over the grid
    for iz in range(Nz):
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
