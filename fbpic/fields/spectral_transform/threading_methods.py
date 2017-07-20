# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of functions that are useful when converting the
fields from interpolation grid to the spectral grid and vice-versa
"""
from fbpic.threading_utils import prange, njit_parallel

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
