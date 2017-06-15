# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Soeren Jalas
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure and methods associated with the fields.
"""
import numpy as np

def get_filter_array( kz, kr, dz, dr ) :
    """
    Return the array that multiplies the fields in k space

    The filtering function is 1-sin( k/kmax * pi/2 )**2.
    (equivalent to a one-pass binomial filter in real space,
    for the longitudinal direction)

    Parameters
    ----------
    kz: 1darray
        The true wavevectors of the longitudinal, spectral grid
        (i.e. not the kz modified by finite order)

    kr: 1darray
        The transverse wavevectors on the spectral grid

    dz, dr: float
        The grid spacings (needed to calculate
        precisely the filtering function in spectral space)

    Returns
    -------
    A 2darray of shape ( len(kz), len(kr) )
    """
    # Find the 1D filter in z
    filt_z = 1. - np.sin( 0.5 * kz * dz )**2

    # Find the 1D filter in r
    filt_r = 1. - np.sin( 0.5 * kr * dr )**2

    # Build the 2D filter by takin the product
    filter_array = filt_z[:, np.newaxis] * filt_r[np.newaxis, :]

    return( filter_array )


def get_modified_k(k, n_order, dz):
    """
    Calculate the modified k that corresponds to a finite-order stencil

    The modified k are given by the formula
    $$ [k] = \sum_{n=1}^{m} a_n \,\frac{\sin(nk\Delta z)}{n\Delta z}$$
    with
    $$a_{n} = - \left(\frac{m+1-n}{m+n}\right) a_{n-1}$$

    Parameter:
    ----------
    k: 1darray
       Values of the real k at which to calculate the modified k

    n_order: int
       The order of the stencil
           Use -1 for infinite order
           Otherwise use a positive, even number. In this case
           the stencil extends up to n_order/2 cells on each side.

    dz: double
       The spacing of the grid in z

    Returns:
    --------
    A 1d array of the same length as k, which contains the modified ks
    """
    # Check the order
    # - For n_order = -1, do not go through the function.
    if n_order==-1:
        return( k )
    # - Else do additional checks
    elif n_order%2==1 or n_order<=0 :
        raise ValueError('Invalid n_order: %d' %n_order)
    else:
        m = int(n_order/2)

    # Calculate the stencil coefficients a_n by recurrence
    # (See definition of the a_n in the docstring)
    # $$ a_{n} = - \left(\frac{m+1-n}{m+n}\right) a_{n-1} $$
    stencil_coef = np.zeros(m+1)
    stencil_coef[0] = -2.
    for n in range(1,m+1):
        stencil_coef[n] = - (m+1-n)*1./(m+n) * stencil_coef[n-1]

    # Array of values for n: from 1 to m
    n_array = np.arange(1,m+1)
    # Array of values of sin:
    # first axis corresponds to k and second axis to n (from 1 to m)
    sin_array = np.sin( k[:,np.newaxis] * n_array[np.newaxis,:] * dz ) / \
                ( n_array[np.newaxis,:] * dz )

    # Modified k
    k_array = np.tensordot( sin_array, stencil_coef[1:], axes=(-1,-1))

    return( k_array )

def stencil_reach(kz, kperp, cdt):
    """
    Return the stencil reach (in spatial space) for a given modified kz
    (finite order stencil in the spectral domain) at a single kperp.
    The stencil reach is needed to define the number of guard cells
    between MPI domains when running in parallel.

    Parameters:
    ----------
    kz: array
        The modified longitudinal k (kz) array

    kperp: float
        Transverse k to use for calculations
        (The stencil reach is calculated at a single kperp)

    cdt: float
        Timestep times speed of light

    Returns:
    -------
    Number of cells needed for the stencil to decrease to machine precision
    """
    k = np.sqrt(kz**2 + kperp**2)
    # Calculation of the stencils for the three C/S coefficients
    # in the cylindrical PSATD equations
    cos_stencil = np.fft.ifft(
        np.cos( k * cdt) )
    sin_z_stencil = np.fft.ifft(
        np.where(k == 0, kz, np.sin(k * cdt) / (k) * kz) )
    sin_perp_stencil = np.fft.ifft(
        np.where(k == 0, kperp, np.sin(k * cdt) / (k) * kperp) )

    # Combination of the stencil function of all three C/S coefficients
    alpha = np.sqrt(np.abs(cos_stencil)**2 +
                    np.abs(sin_z_stencil)**2 +
                    np.abs(sin_perp_stencil)**2)
    # Reach of the stencil defined at the position, when the signal
    # decreased to machine precision
    stencil_reach = np.where(np.abs(alpha)[:int(alpha.shape[0]/2)] < 1.e-16)[0][0]

    return int(stencil_reach)

def get_stencil_reach(Nz, dz, cdt, n_order):
    """
    Return the stencil reach (in spatial space) for a given finite order
    stencil and for a given simulation setup with Nz cells and spacing dz.
    The stencil reach depends only slightly on the transverse k (kr) and will
    be calculated at a fixed kperp of 0.5 . The stencil reach is needed to
    define the number of guard cells between domains when running in parallel.

    Parameters:
    ----------
    Nz: int
        The number of cells in the logintudinal direction

    dz: float (microns)
        The cell spacing in the longitudinal direction

    cdt: float
        Timestep times speed of light

    n_order: int (multiple of 2)
        The (finite) order of the arbitrary order spectral,
        Maxwell (PSATD) solver, which defines the stencil reach.

    Returns:
    -------
    Number of cells needed for the stencil to decrease to
    machine precision at kperp = 0.5
    """
    # Calculate the real kz for the given grid (Nz)
    real_kz = 2 * np.pi * np.fft.fftfreq(Nz, d=dz)
    # Get the modified, finite order kz
    kz = get_modified_k(real_kz, n_order, dz=dz)

    # Calculate the stencil reach at an arbitrary kperp = 0.5
    # (Note: The stencil reach depends only weakly on kperp)
    return stencil_reach(kz, 0.5, cdt)
