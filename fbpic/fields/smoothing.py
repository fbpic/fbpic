# Copyright 2018, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Soeren Jalas
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a class for field smoothing.
"""
import numpy as np

class BinomialSmoother( object ):
    """
    Class that defines a binomial smoother on a grid,
    potentially with compensator.
    """

    def __init__( self, n_passes=1, compensator=False):
        """
        Initialize a binomial smoother

        Parameters
        ----------
        n_passes: int or dictionary of ints
            Number of passes of binomial smoothing in r and z.
            (More passes results in smoother fields)
            If `n_passes` is an integer, than the same number of passes
            are applied in r and z.
            If `n_passes` is a dictionary, it should be of the form
            {'z': <int>, 'r': <int>} to indicate the number of passes
            in each direction.

        compensator: bool, or dictionary of bools
            Whether to apply a compensator in r and z.
            (Applying a compensator mitigates the impact of the smoother
            on low and intermediate frequencies)
            If `compensator` is a boolean, than this applies to r and z.
            If `n_passes` is a dictionary, it should be of the form
            {'z': <bool>, 'r': <bool>} to indicate whether a compensator
            is applied in r and/or in z.
        """
        # Register smoothing parameters
        if type(n_passes) is int:
            self.n_passes = {'z': n_passes, 'r': n_passes}
        elif type(n_passes) is dict:
            self.n_passes = n_passes
        else:
            raise ValueError('Invalid argument `n_passes`')

        # Register compensator
        if type(compensator) is bool:
            self.compensator = {'z': compensator, 'r': compensator}
        elif type(compensator) is dict:
            self.compensator = compensator
        else:
            raise ValueError('Invalid argument `compensator`')


    def get_filter_array( self, kz, kr, dz, dr ) :
        """
        Return the array that multiplies the fields in k space

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
        # Equivalent to nz passes of binomial filter in real space
        sz2 = np.sin( 0.5 * kz * dz )**2
        nz = self.n_passes['z']
        filt_z = ( 1. - sz2 )**nz
        # Add compensator
        if self.compensator['z']:
            filt_z *= ( 1. + nz*sz2 )

        # Equivalent to nr passes of binomial filter in real space
        sr2 = np.sin( 0.5 * kr * dr )**2
        nr = self.n_passes['r']
        filt_r = ( 1. - sr2 )**nr
        # Add compensator
        if self.compensator['r']:
            filt_r *= ( 1. + nr*sr2 )

        return( filt_z, filt_r )
