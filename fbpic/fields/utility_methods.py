# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Soeren Jalas
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure and methods associated with the fields.
"""
import numpy as np
from scipy.constants import c

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


def stencil_reach(kz, kperp, cdt, v_comoving, use_galilean):
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

    v_comoving: float or None, optional
        If this variable is None, the standard PSATD is used (default).
        Otherwise, the current is assumed to be "comoving",
        i.e. constant with respect to (z - v_comoving * t).
        This can be done in two ways: either by
        - Using a PSATD scheme that takes this hypothesis into account
        - Solving the PSATD scheme in a Galilean frame

    use_galilean: bool, optional
        Determines which one of the two above schemes is used
        When use_galilean is true, the whole grid moves
        with a speed v_comoving

    Returns:
    -------
    Number of cells needed for the stencil to decrease to machine precision
    """
    k = np.sqrt(kz**2 + kperp**2)
    # Calculation of the Theta coefficient if the Galilean scheme is used
    if use_galilean is True:
        # When using the galilean scheme the stencil reach is always larger
        # in the direction of v_comoving. Use abs(v_comoving) to have the
        # maximum stencil extent
        abs_v_comoving = np.abs(v_comoving)
        theta = np.exp(1.j * abs_v_comoving * kz * cdt / c / 2)
    else:
        theta = np.ones_like(kz)
    # Calculation of the stencils for the three C/S coefficients
    # in the cylindrical PSATD equations
    cos_stencil = np.fft.ifft(
        theta ** 2 * np.cos( k * cdt) )
    sin_z_stencil = np.fft.ifft(
        np.where(k == 0, kz, theta ** 2 * np.sin(k * cdt) / (k) * kz) )
    sin_perp_stencil = np.fft.ifft(
        np.where(k == 0, kperp, theta ** 2 * np.sin(k * cdt) / (k) * kperp) )

    # Combination of the stencil function of all three C/S coefficients
    alpha = np.sqrt(np.abs(cos_stencil)**2 +
                    np.abs(sin_z_stencil)**2 +
                    np.abs(sin_perp_stencil)**2)
    # Reach of the stencil defined at the position, when the signal
    # decreased to machine precision
    stencil_reach = np.where(np.abs(alpha)
                             [:int(alpha.shape[0] / 2)] < 1.e-16)[0][0]

    return int(stencil_reach)


def get_stencil_reach(Nz, dz, cdt, n_order, v_comoving, use_galilean):
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

    v_comoving: float or None, optional
        If this variable is None, the standard PSATD is used (default).
        Otherwise, the current is assumed to be "comoving",
        i.e. constant with respect to (z - v_comoving * t).
        This can be done in two ways: either by
        - Using a PSATD scheme that takes this hypothesis into account
        - Solving the PSATD scheme in a Galilean frame

    use_galilean: bool, optional
        Determines which one of the two above schemes is used
        When use_galilean is true, the whole grid moves
        with a speed v_comoving

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

    return stencil_reach(kz, 0.5, cdt, v_comoving, use_galilean)
