# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of functions that are useful when converting the
fields from interpolation grid to the spectral grid and vice-versa
"""
from fbpic.utils.threading import prange, njit_parallel

@njit_parallel
def numba_copy_2dC_to_2dR( array_in, array_out ) :
    """
    Store the complex Nz x Nr array `array_in`
    into the real 2Nz x Nr array `array_out`,
    by storing the real part in the first Nz elements of `array_out` along z,
    and the imaginary part in the next Nz elements.

    Parameters :
    ------------
    array_in: 2darray of complexs
        Array of shape (Nz, Nr)
    array_out: 2darray of reals
        Array of shape (2*Nz, Nr)
    """
    Nz, Nr = array_in.shape

    # Loop over the 2D grid (parallel in z, if threading is installed)
    for iz in prange(Nz):
        for ir in range(Nr):
            array_out[iz, ir] = array_in[iz, ir].real
            array_out[iz+Nz, ir] = array_in[iz, ir].imag

@njit_parallel
def numba_copy_2dR_to_2dC( array_in, array_out ) :
    """
    Reconstruct the complex Nz x Nr array `array_out`,
    from the real 2Nz x Nr array `array_in`,
    by interpreting the first Nz elements of `array_in` along z as
    the real part, and the next Nz elements as the imaginary part.

    Parameters :
    ------------
    array_in: 2darray of reals
        Array of shape (2*Nz, Nr)
    array_out: 2darray of complexs
        Array of shape (Nz, Nr)
    """
    Nz, Nr = array_out.shape

    # Loop over the 2D grid (parallel in z, if threading is installed)
    for iz in prange(Nz):
        for ir in range(Nr):
            array_out[iz, ir] = array_in[iz, ir] + 1.j*array_in[iz+Nz, ir]

# ----------------------------------------------------
# Functions that combine components in spectral space
# ----------------------------------------------------

@njit_parallel
def numba_rt_to_pm( buffer_r, buffer_t, buffer_p, buffer_m ) :
    """
    Combine the arrays buffer_r and buffer_t to produce the
    arrays buffer_p and buffer_m, according to the rules of
    the Fourier-Hankel decomposition (see associated paper)
    """
    Nz, Nr = buffer_r.shape

    # Loop over the 2D grid (parallel in z, if threading is installed)
    for iz in prange(Nz):
        for ir in range(Nr):

            # Use intermediate variables, as the arrays
            # buffer_r and buffer_t may actually point to the same
            # object as buffer_p and buffer_m, for economy of memory
            value_r = buffer_r[iz, ir]
            value_t = buffer_t[iz, ir]
            # Combine the values
            buffer_p[iz, ir] = 0.5*( value_r - 1.j*value_t )
            buffer_m[iz, ir] = 0.5*( value_r + 1.j*value_t )


@njit_parallel
def numba_pm_to_rt( buffer_p, buffer_m, buffer_r, buffer_t ) :
    """
    Combine the arrays buffer_p and buffer_m to produce the
    arrays buffer_r and buffer_t, according to the rules of
    the Fourier-Hankel decomposition (see associated paper)
    """
    Nz, Nr = buffer_p.shape

    # Loop over the 2D grid (parallel in z, if threading is installed)
    for iz in prange(Nz):
        for ir in range(Nr):

            # Use intermediate variables, as the arrays
            # buffer_r and buffer_t may actually point to the same
            # object as buffer_p and buffer_m, for economy of memory
            value_p = buffer_p[iz, ir]
            value_m = buffer_m[iz, ir]
            # Combine the values
            buffer_r[iz, ir] =     ( value_p + value_m )
            buffer_t[iz, ir] = 1.j*( value_p - value_m )
