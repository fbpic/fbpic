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
    TODO
    """

    def __init__( self, n_smoothing_passes, compensator=False):
        """
        TODO
        """
        # Register smoothing parameters
        if type(n_smoothing_passes) is int:
            self.n_smoothing_passes = {'z': n_smoothing_passes,
                                       'r': n_smoothing_passes}
        elif type(n_smoothing_passes) is dict:
            self.n_smoothing_passes = n_smoothing_passes
        else:
            raise ValueError('Invalid argument `n_smoothing_passes`')

        # Register compensator
        if type(compensator) is bool:
            self.compensator = {'z': compensator,
                                'r': compensator}
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
        nz = self.n_smoothing_passes['z']
        filt_z = ( 1. - sz2 )**nz
        # Add compensator
        if self.compensator['z']:
            filt_z *= ( 1. + nz*sz2 )

        # Equivalent to nr passes of binomial filter in real space
        sr2 = np.sin( 0.5 * kr * dr )**2
        nr = self.n_smoothing_passes['r']
        filt_r = ( 1. - sr2 )**nr
        # Add compensator
        if self.compensator['r']:
            filt_r *= ( 1. + nr*sr2 )

        return( filt_z, filt_r )
