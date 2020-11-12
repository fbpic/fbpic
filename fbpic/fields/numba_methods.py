# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the optimized fields methods that use numba on a CPU
"""
from scipy.constants import c, epsilon_0, mu_0
c2 = c**2
import numba
from fbpic.utils.threading import njit_parallel, prange


@njit_parallel
def numba_filter_scalar( field, Nz, Nr, filter_array_z, filter_array_r ) :
    """
    Multiply the input field by the filter_array

    Parameters :
    ------------
    field : 2darray of complexs
        An array that represent the fields in spectral space

    filter_array_z, filter_array_r : 1darray of reals
        An array that damps the fields at high k, in z and r respectively

    Nz, Nr : ints
        Dimensions of the arrays
    """
    # Loop over the 2D grid (parallel in z, if threading is installed)
    for iz in prange(Nz):
        for ir in range(Nr):

            field[iz,ir] = filter_array_z[iz]*filter_array_r[ir]*field[iz,ir]


@njit_parallel
def numba_filter_vector( fieldr, fieldt, fieldz, Nz, Nr,
                        filter_array_z, filter_array_r ):
    """
    Multiply the input field by the filter_array

    Parameters :
    ------------
    field : 2darray of complexs
        An array that represent the fields in spectral space

    filter_array_z, filter_array_r : 1darray of reals
        An array that damps the fields at high k, in z and r respectively

    Nz, Nr : ints
        Dimensions of the arrays
    """
    # Loop over the 2D grid (parallel in z, if threading is installed)
    for iz in prange(Nz):
        for ir in range(Nr):

            fieldr[iz,ir] = filter_array_z[iz]*filter_array_r[ir]*fieldr[iz,ir]
            fieldt[iz,ir] = filter_array_z[iz]*filter_array_r[ir]*fieldt[iz,ir]
            fieldz[iz,ir] = filter_array_z[iz]*filter_array_r[ir]*fieldz[iz,ir]


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
        for ir in range(Nr):

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
def numba_push_eb_pml_standard( Ep_pml, Em_pml, Bp_pml, Bm_pml,
                        Ez, Bz, C, S_w, kr, kz, Nz, Nr):
    """
    Push the PML split fields over one timestep, using the standard psatd algorithm

    See the documentation of SpectralGrid.push_eb_with
    """
    # Loop over the 2D grid
    for iz in prange(Nz):
        for ir in range(Nr):

            # Push the PML E field
            Ep_pml[iz, ir] = C[iz, ir]*Ep_pml[iz, ir] \
                + c2*S_w[iz, ir]*( -1.j*0.5*kr[iz, ir]*Bz[iz, ir] )

            Em_pml[iz, ir] = C[iz, ir]*Em_pml[iz, ir] \
                + c2*S_w[iz, ir]*( -1.j*0.5*kr[iz, ir]*Bz[iz, ir] )

            # Push the PML B field
            Bp_pml[iz, ir] = C[iz, ir]*Bp_pml[iz, ir] \
                - S_w[iz, ir]*( -1.j*0.5*kr[iz, ir]*Ez[iz, ir] )

            Bm_pml[iz, ir] = C[iz, ir]*Bm_pml[iz, ir] \
                - S_w[iz, ir]*( -1.j*0.5*kr[iz, ir]*Ez[iz, ir] )

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
        for ir in range(Nr):

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

@njit_parallel
def numba_push_eb_pml_comoving( Ep_pml, Em_pml, Bp_pml, Bm_pml,
                        Ez, Bz, C, S_w, T_eb, kr, kz, Nz, Nr):
    """
    Push the PML split fields over one timestep,
    using the galilean/comoving psatd algorithm

    See the documentation of SpectralGrid.push_eb_with
    """
    # Loop over the 2D grid
    for iz in prange(Nz):
        for ir in range(Nr):

            # Push the E field
            Ep_pml[iz, ir] = T_eb[iz, ir]*C[iz, ir]*Ep_pml[iz, ir] \
                + c2*T_eb[iz, ir]*S_w[iz, ir]*(-1.j*0.5*kr[iz, ir]*Bz[iz, ir])
            Em_pml[iz, ir] = T_eb[iz, ir]*C[iz, ir]*Em_pml[iz, ir] \
                + c2*T_eb[iz, ir]*S_w[iz, ir]*(-1.j*0.5*kr[iz, ir]*Bz[iz, ir])

            # Push the B field
            Bp_pml[iz, ir] = T_eb[iz, ir]*C[iz, ir]*Bp_pml[iz, ir] \
                - T_eb[iz, ir]*S_w[iz, ir]*( -1.j*0.5*kr[iz, ir]*Ez[iz, ir] )
            Bm_pml[iz, ir] = T_eb[iz, ir]*C[iz, ir]*Bm_pml[iz, ir] \
                - T_eb[iz, ir]*S_w[iz, ir]*( -1.j*0.5*kr[iz, ir]*Ez[iz, ir] )

    return


# -----------------------------------------------------------------------
# Parallel reduction of the global arrays for threads into a single array
# -----------------------------------------------------------------------

@njit_parallel
def numba_erase_threading_buffer( global_array ):
    """
    Set the threading buffer `global_array` to 0

    Parameter:
    ----------
    global_array: 4darray of complexs
        An array that contains the duplicated charge/current for each thread
    """
    nthreads, Nm, Nz, Nr = global_array.shape
    # Loop in parallel along nthreads
    for i_thread in prange(nthreads):
        # Loop through the modes and the grid
        for m in range(Nm):
            for iz in range(Nz):
                for ir in range(Nr):
                    # Erase values
                    global_array[i_thread, m, iz, ir] = 0.

@njit_parallel
def sum_reduce_2d_array( global_array, reduced_array, m ):
    """
    Sum the array `global_array` along its first axis and
    add it into `reduced_array`, and fold the deposition guard cells of
    global_array into the regular cells of reduced_array.

    Parameters:
    -----------
    global_array: 4darray of complexs
       Field array of shape (nthreads, Nm, 2+Nz+2, 2+Nr+2)
       where the additional 2's in z and r correspond to deposition guard cells
       that were used during the threaded deposition kernel.

    reduced array: 2darray of complex
      Field array of shape (Nz, Nr)

    m: int
       The azimuthal mode for which the reduction should be performed
    """
    # Extract size of each dimension
    Nz = reduced_array.shape[0]

    # Parallel loop over z
    for iz in prange(Nz):
        # Get index inside reduced_array
        iz_global = iz + 2
        reduce_slice( reduced_array, iz, global_array, iz_global, m )
    # Handle deposition guard cells in z
    reduce_slice( reduced_array, Nz-2, global_array, 0, m )
    reduce_slice( reduced_array, Nz-1, global_array, 1, m )
    reduce_slice( reduced_array, 0, global_array, Nz+2, m )
    reduce_slice( reduced_array, 1, global_array, Nz+3, m )

@numba.njit
def reduce_slice( reduced_array, iz, global_array, iz_global, m ):
    """
    Sum the array `global_array` into `reduced_array` for one given slice in z
    """
    Nreduce = global_array.shape[0]
    Nr = reduced_array.shape[1]
    # Loop over the reduction dimension (slow dimension)
    for it in range( Nreduce ):

        # First fold the low-radius deposition guard cells in
        reduced_array[iz, 1] += global_array[it, m, iz_global, 0]
        reduced_array[iz, 0] += global_array[it, m, iz_global, 1]
        # Then loop over regular cells
        for ir in range( Nr ):
            reduced_array[iz, ir] +=  global_array[it, m, iz_global, ir+2]
        # Finally fold the high-radius guard cells in
        reduced_array[iz, Nr-1] += global_array[it, m, iz_global, Nr+2]
        reduced_array[iz, Nr-1] += global_array[it, m, iz_global, Nr+3]
