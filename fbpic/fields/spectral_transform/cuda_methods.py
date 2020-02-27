# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of functions that are useful when converting the
fields from interpolation grid to the spectral grid and vice-versa
"""
from numba import cuda
from fbpic.utils.cuda import compile_cupy

# ------------------
# Copying functions
# ------------------

@compile_cupy
def cuda_copy_2dC_to_2dR( array_in, array_out ) :
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
    # Set up cuda grid
    iz, ir = cuda.grid(2)
    Nz, Nr = array_in.shape

    # Copy from array_in to array_out
    if (iz < Nz) and (ir < Nr) :
        array_out[iz, ir] = array_in[iz, ir].real
        array_out[iz+Nz, ir] = array_in[iz, ir].imag

@compile_cupy
def cuda_copy_2dR_to_2dC( array_in, array_out ) :
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
    # Set up cuda grid
    iz, ir = cuda.grid(2)
    Nz, Nr = array_out.shape

    # Copy from array_in to array_out
    if (iz < Nz) and (ir < Nr) :
        array_out[iz, ir] = array_in[iz, ir] + 1.j*array_in[iz+Nz, ir]

@compile_cupy
def cuda_copy_2d_to_1d( array_2d, array_1d ) :
    """
    Copy array_2d to array_1d, so that the first axis of array_2d
    is contiguous in array_1d

    This is typically done before using the cuFFT API,
    which requires 1d arrays to do FFT in only one dimension.

    Parameters :
    ------------
    array_2d : 2darray of complexs
        Array of shape (Nz, Nr)

    array_1d : 1d array of complexs
        Array of shape (Nz*Nr,)
    """

    # Set up cuda grid
    iz, ir = cuda.grid(2)

    # Copy from array_2d to array_1d
    if (iz < array_2d.shape[0]) and (ir < array_2d.shape[1]) :
        i = iz + array_2d.shape[0]*ir
        array_1d[i] = array_2d[iz, ir]

@compile_cupy
def cuda_copy_1d_to_2d( array_1d, array_2d ) :
    """
    Copy array_1d to array_2d, so that the first axis of array_2d
    is contiguous in array_1d

    This is typically done after using the cuFFT API,
    since the the output of cuFFT is a 1d array

    Parameters :
    ------------
    array_2d : 2darray of complexs
        Array of shape (Nz, Nr)

    array_1d : 1d array of complexs
        Array of shape (Nz*Nr,)
    """

    # Set up cuda grid
    iz, ir = cuda.grid(2)

    # Copy from array_1d to array_2d
    if (iz < array_2d.shape[0]) and (ir < array_2d.shape[1]) :
        i = iz + array_2d.shape[0]*ir
        array_2d[iz, ir] = array_1d[i]

# ----------------------------------------------------
# Functions that combine components in spectral space
# ----------------------------------------------------

@compile_cupy
def cuda_rt_to_pm( buffer_r, buffer_t, buffer_p, buffer_m ) :
    """
    Combine the arrays buffer_r and buffer_t to produce the
    arrays buffer_p and buffer_m, according to the rules of
    the Fourier-Hankel decomposition (see associated paper)
    """
    # Set up cuda grid
    iz, ir = cuda.grid(2)

    if (iz < buffer_r.shape[0]) and (ir < buffer_r.shape[1]) :
        # Use intermediate variables, as the arrays
        # buffer_r and buffer_t may actually point to the same
        # object as buffer_p and buffer_m, for economy of memory
        value_r = buffer_r[iz, ir]
        value_t = buffer_t[iz, ir]
        # Combine the values
        buffer_p[iz, ir] = 0.5*( value_r - 1.j*value_t )
        buffer_m[iz, ir] = 0.5*( value_r + 1.j*value_t )


@compile_cupy
def cuda_pm_to_rt( buffer_p, buffer_m, buffer_r, buffer_t ) :
    """
    Combine the arrays buffer_p and buffer_m to produce the
    arrays buffer_r and buffer_t, according to the rules of
    the Fourier-Hankel decomposition (see associated paper)
    """
    # Set up cuda grid
    iz, ir = cuda.grid(2)

    if (iz < buffer_p.shape[0]) and (ir < buffer_p.shape[1]) :
        # Use intermediate variables, as the arrays
        # buffer_r and buffer_t may actually point to the same
        # object as buffer_p and buffer_m, for economy of memory
        value_p = buffer_p[iz, ir]
        value_m = buffer_m[iz, ir]
        # Combine the values
        buffer_r[iz, ir] =     ( value_p + value_m )
        buffer_t[iz, ir] = 1.j*( value_p - value_m )
