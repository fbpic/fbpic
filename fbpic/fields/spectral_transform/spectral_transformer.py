# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the SpectralTransformer class, which handles conversion of
the fields from the interpolation grid to the spectral grid and vice-versa.
"""
import numpy as np
from .hankel import DHT
from .fourier import FFT

from .numba_methods import numba_rt_to_pm, numba_pm_to_rt
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    import cupy
    from fbpic.utils.cuda import cuda_tpb_bpg_2d
    from .cuda_methods import cuda_rt_to_pm, cuda_pm_to_rt

class SpectralTransformer(object) :
    """
    Object that allows to transform the fields back and forth between the
    spectral and interpolation grid.

    Attributes :
    - dht0, dhtm, dhtp : the discrete Hankel transform objects
       that operates along r
    - fft : the discrete Fourier transform object that operates along z

    Main methods :
    - spect2interp_scal :
        converts a scalar field from the spectral to the interpolation grid
    - spect2interp_vect :
        converts a vector field from the spectral to the interpolation grid
    - interp2spect_scal :
        converts a scalar field from the interpolation to the spectral grid
    - interp2spect_vect :
        converts a vector field from the interpolation to the spectral grid
    """

    def __init__(self, Nz, Nr, m, rmax, use_cuda=False ) :
        """
        Initializes the dht and fft attributes, which contain auxiliary
        matrices allowing to transform the fields quickly

        Parameters
        ----------
        Nz, Nr : int
            Number of points along z and r respectively

        m : int
            Index of the mode (needed for the Hankel transform)

        rmax : float
            The size of the simulation box along r.
        """
        # Check whether to use the GPU
        self.use_cuda = use_cuda
        if (self.use_cuda is True) and (cuda_installed is False) :
            self.use_cuda = False
        if self.use_cuda:
            # Initialize the dimension of the grid and blocks
            self.dim_grid, self.dim_block = cuda_tpb_bpg_2d( Nz, Nr, 1, 32 )

        # Initialize the DHT (local implementation, see hankel.py)
        self.dht0 = DHT(  m, m, Nr, Nz, rmax, use_cuda=self.use_cuda )
        self.dhtp = DHT(m+1, m, Nr, Nz, rmax, use_cuda=self.use_cuda )
        self.dhtm = DHT(m-1, m, Nr, Nz, rmax, use_cuda=self.use_cuda )

        # Initialize the FFT
        self.fft = FFT( Nr, Nz, use_cuda=self.use_cuda )

        # Initialize the spectral buffers
        if self.use_cuda:
            self.spect_buffer_r = cupy.empty(
                (Nz, Nr), dtype=np.complex128)
            self.spect_buffer_t = cupy.empty(
                (Nz, Nr), dtype=np.complex128)
        else:
            # Initialize the spectral buffers
            self.spect_buffer_r = np.zeros( (Nz, Nr), dtype=np.complex128 )
            self.spect_buffer_t = np.zeros( (Nz, Nr), dtype=np.complex128 )

        # Different names for same object (for economy of memory)
        self.spect_buffer_p = self.spect_buffer_r
        self.spect_buffer_m = self.spect_buffer_t

    def spect2interp_scal( self, spect_array, interp_array ) :
        """
        Convert a scalar field from the spectral grid
        to the interpolation grid.

        Parameters
        ----------
        spect_array : 2darray of complexs
           A complex array representing the fields in spectral space, from
           which to compute the values of the interpolation grid
           The first axis should correspond to z and the second axis to r.

        interp_array : 2darray of complexs
           A complex array representing the fields on the interpolation
           grid, and which is overwritten by this function.
        """
        # Perform the inverse DHT (along axis -1, which corresponds to r)
        self.dht0.inverse_transform( spect_array, self.spect_buffer_r )

        # Then perform the inverse FFT (along axis 0, which corresponds to z)
        self.fft.inverse_transform( self.spect_buffer_r, interp_array )

    def spect2interp_vect( self, spect_array_p, spect_array_m,
                          interp_array_r, interp_array_t ) :
        """
        Convert a transverse vector field in the spectral space (e.g. Ep, Em)
        to the interpolation grid (e.g. Er, Et)

        Parameters
        ----------
        spect_array_p, spect_array_m : 2darray
           Complex arrays representing the fields in spectral space, from
           which to compute the values of the interpolation grid
           The first axis should correspond to z and the second axis to r.

        interp_array_r, interp_array_t : 2darray
           Complex arrays representing the fields on the interpolation
           grid, and which are overwritten by this function.
        """
        # Perform the inverse DHT (along axis -1, which corresponds to r)
        self.dhtp.inverse_transform( spect_array_p, self.spect_buffer_p )
        self.dhtm.inverse_transform( spect_array_m, self.spect_buffer_m )

        # Combine the p and m components to obtain the r and t components
        if self.use_cuda :
            # Combine them on the GPU
            # (self.spect_buffer_r and self.spect_buffer_t are
            # passed in the following line, in order to make things
            # explicit, but they actually point to the same object
            # as self.spect_buffer_p, self.spect_buffer_m,
            # for economy of memory)
            cuda_pm_to_rt[self.dim_grid, self.dim_block](
                self.spect_buffer_p, self.spect_buffer_m,
                self.spect_buffer_r, self.spect_buffer_t )
        else :
            # Combine them on the CPU
            # (self.spect_buffer_r and self.spect_buffer_t are
            # passed in the following line, in order to make things
            # explicit, but they actually point to the same object
            # as self.spect_buffer_p, self.spect_buffer_m,
            # for economy of memory)
            numba_pm_to_rt( self.spect_buffer_p, self.spect_buffer_m,
                            self.spect_buffer_r, self.spect_buffer_t )

        # Finally perform the FFT (along axis 0, which corresponds to z)
        self.fft.inverse_transform( self.spect_buffer_r, interp_array_r )
        self.fft.inverse_transform( self.spect_buffer_t, interp_array_t )

    def interp2spect_scal( self, interp_array, spect_array ) :
        """
        Convert a scalar field from the interpolation grid
        to the spectral grid.

        Parameters
        ----------
        interp_array : 2darray
           A complex array representing the fields on the interpolation
           grid, from which to compute the values of the interpolation grid
           The first axis should correspond to z and the second axis to r.

        spect_array : 2darray
           A complex array representing the fields in spectral space,
           and which is overwritten by this function.
        """
        # Perform the FFT first (along axis 0, which corresponds to z)
        self.fft.transform( interp_array, self.spect_buffer_r )

        # Then perform the DHT (along axis -1, which corresponds to r)
        self.dht0.transform( self.spect_buffer_r, spect_array )

    def interp2spect_vect( self, interp_array_r, interp_array_t,
                           spect_array_p, spect_array_m ) :
        """
        Convert a transverse vector field from the interpolation grid
        (e.g. Er, Et) to the spectral space (e.g. Ep, Em)

        Parameters
        ----------
        interp_array_r, interp_array_t : 2darray
           Complex arrays representing the fields on the interpolation
           grid, from which to compute the values in spectral space
           The first axis should correspond to z and the second axis to r.

        spect_array_p, spect_array_m : 2darray
           Complex arrays representing the fields in spectral space,
           and which are overwritten by this function.
        """
        # Perform the FFT first (along axis 0, which corresponds to z)
        self.fft.transform( interp_array_r, self.spect_buffer_r )
        self.fft.transform( interp_array_t, self.spect_buffer_t )

        # Combine the r and t components to obtain the p and m components
        if self.use_cuda :
            # Combine them on the GPU
            # (self.spect_buffer_p and self.spect_buffer_m are
            # passed in the following line, in order to make things
            # explicit, but they actually point to the same object
            # as self.spect_buffer_r, self.spect_buffer_t,
            # for economy of memory)
            cuda_rt_to_pm[self.dim_grid, self.dim_block](
                self.spect_buffer_r, self.spect_buffer_t,
                self.spect_buffer_p, self.spect_buffer_m )
        else :
            # Combine them on the GPU
            # (self.spect_buffer_p and self.spect_buffer_m are
            # passed in the following line, in order to make things
            # explicit, but they actually point to the same object
            # as self.spect_buffer_r, self.spect_buffer_t,
            # for economy of memory)
            numba_rt_to_pm( self.spect_buffer_r, self.spect_buffer_t,
                            self.spect_buffer_p, self.spect_buffer_m )

        # Perform the inverse DHT (along axis -1, which corresponds to r)
        self.dhtp.transform( self.spect_buffer_p, spect_array_p )
        self.dhtm.transform( self.spect_buffer_m, spect_array_m )
