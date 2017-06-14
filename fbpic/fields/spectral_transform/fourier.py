# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Partile-In-Cell code (FB-PIC)
It defines the FFT object, which performs Fourier transforms along the axis 0,
and is used in spectral_transformer.py
"""
import numpy as np
import pyfftw
# Check if CUDA is available, then import CUDA functions
from fbpic.cuda_utils import cuda_installed
if cuda_installed:
    from accelerate.cuda import fft as cufft, blas as cublas
    from fbpic.cuda_utils import cuda, cuda_tpb_bpg_2d
    from .cuda_methods import cuda_copy_2d_to_1d, cuda_copy_1d_to_2d

class FFT(object):
    """
    Object that performs Fourier transform of 2D arrays along the z axis,
    (axis 0) either on the CPU (using pyfftw) or on the GPU (using cufft)

    See the methods `transform` and `inverse transform` for more information
    """

    def __init__(self, Nr, Nz, use_cuda=False, nthreads=4 ):
        """
        Initialize an FFT object

        Parameters
        ----------
        Nr: int
           Number of grid points along the r axis (axis -1)

        Nz: int
           Number of grid points along the z axis (axis 0)

        use_cuda: bool, optional
           Whether to perform the Fourier transform on the z axis

        nthreads : int, optional
            Number of threads for the FFTW transform
        """
        # Check whether to use cuda
        self.use_cuda = use_cuda
        if (self.use_cuda is True) and (cuda_installed is False) :
            self.use_cuda = False
            print('** Cuda not available for Fourier transform.')
            print('** Performing the Fourier transform on the CPU.')

        # Initialize the object for calculation on the GPU
        if self.use_cuda:
            # Initialize the dimension of the grid and blocks
            self.dim_grid, self.dim_block = cuda_tpb_bpg_2d( Nz, Nr)

            # Initialize 1d buffer for cufft
            self.buffer1d_in = cuda.device_array(
                (Nz*Nr,), dtype=np.complex128)
            self.buffer1d_out = cuda.device_array(
                (Nz*Nr,), dtype=np.complex128)
            # Initialize the cuda libraries object
            self.fft = cufft.FFTPlan( shape=(Nz,), itype=np.complex128,
                                      otype=np.complex128, batch=Nr )
            self.blas = cublas.Blas()   # For normalization of the iFFT
            self.inv_Nz = 1./Nz         # For normalization of the iFFT

            # Initialize the spectral buffers
            self.spect_buffer_r = cuda.device_array(
                (Nz, Nr), dtype=np.complex128)
            self.spect_buffer_t = cuda.device_array(
                (Nz, Nr), dtype=np.complex128)

        # Initialize the object for calculation on the CPU
        else:
            # First buffer and FFTW transform
            self.interp_buffer_r = \
                pyfftw.n_byte_align_empty( (Nz,Nr), 16, 'complex128' )
            self.spect_buffer_r = \
                pyfftw.n_byte_align_empty( (Nz,Nr), 16, 'complex128' )
            self.fft_r= pyfftw.FFTW(self.interp_buffer_r, self.spect_buffer_r,
                    axes=(0,), direction='FFTW_FORWARD', threads=nthreads)
            self.ifft_r=pyfftw.FFTW(self.spect_buffer_r, self.interp_buffer_r,
                    axes=(0,), direction='FFTW_BACKWARD', threads=nthreads)

            # Second buffer and FFTW transform
            self.interp_buffer_t = \
                pyfftw.n_byte_align_empty( (Nz,Nr), 16, 'complex128' )
            self.spect_buffer_t = \
                pyfftw.n_byte_align_empty( (Nz,Nr), 16, 'complex128' )
            self.fft_t= pyfftw.FFTW(self.interp_buffer_t, self.spect_buffer_t,
                    axes=(0,), direction='FFTW_FORWARD', threads=nthreads )
            self.ifft_t=pyfftw.FFTW(self.spect_buffer_t, self.interp_buffer_t,
                    axes=(0,), direction='FFTW_BACKWARD', threads=nthreads)

    def get_buffers( self ):
        """
        Return the spectral buffers which are typically used to store
        information inbetween the Hankel transform and the Fourier transform

        Returns
        -------
        A tuple with:
        - Two cuda device arrays when using the GPU
        - Two numpy arrays when using the CPU, which are tied to an FFTW plan
        (Do no modify these arrays to make them point to another array)
        """
        return( self.spect_buffer_r, self.spect_buffer_t )

    def transform( self, array_in, array_out ):
        """
        Perform the Fourier transform of array_in,
        and store the result in array_out

        Parameters
        ----------
        array_in, array_out: cuda device arrays or numpy arrays
            When using the GPU, these should be cuda device array.
            When using the CPU, array_out should be one of the
            two buffers that are returned by `get_buffers`
        """
        if self.use_cuda :
            # Perform the FFT on the GPU
            # (The cuFFT API requires 1D arrays)
            cuda_copy_2d_to_1d[self.dim_grid, self.dim_block](
                array_in, self.buffer1d_in )
            self.fft.forward( self.buffer1d_in, out=self.buffer1d_out )
            cuda_copy_1d_to_2d[self.dim_grid, self.dim_block](
                self.buffer1d_out, array_out )
        else :
            # Perform the FFT on the CPU
            if array_out is self.spect_buffer_r:
                # First copy the input array to the preallocated buffers
                self.interp_buffer_r[:,:] = array_in
                # The following operation transforms from
                # self.interp_buffer_r to self.spect_buffer_r
                self.fft_r()
            elif array_out is self.spect_buffer_t:
                # First copy the input array to the preallocated buffers
                self.interp_buffer_t[:,:] = array_in
                # The following operation transforms from
                # self.interp_buffer_t to self.spect_buffer_t
                self.fft_t()
            else:
                raise ValueError('Invalid output array.The output array '
                'must be either self.spect_buffer_r or self.spect_buffer_t.')

    def inverse_transform( self, array_in, array_out ):
        """
        Perform the inverse Fourier transform of array_in,
        and store the result in array_out

        Parameters
        ----------
        array_in, array_out: cuda device arrays or numpy arrays
            When using the GPU, these should be cuda device array.
            When using the CPU, array_in should be one of the
            two buffers that are returned by `get_buffers`
        """
        if self.use_cuda :
            # Perform the inverse FFT on the GPU
            # (The cuFFT API requires 1D arrays)
            cuda_copy_2d_to_1d[self.dim_grid, self.dim_block](
                array_in, self.buffer1d_in )
            self.fft.inverse( self.buffer1d_in, out=self.buffer1d_out )
            self.blas.scal( self.inv_Nz, self.buffer1d_out ) # Normalization
            cuda_copy_1d_to_2d[self.dim_grid, self.dim_block](
                self.buffer1d_out, array_out )
        else :
            # Perform the inverse FFT on the CPU, using preallocated buffers
            if array_in is self.spect_buffer_r:
                # The following operation transforms from
                # self.spect_buffer_r to self.interp_buffer_r
                self.ifft_r()
                # Copy to the output array
                array_out[:,:] = self.interp_buffer_r[:,:]
            elif array_in is self.spect_buffer_t:
                # The following operation transforms from
                # self.spect_buffer_t to self.interp_buffer_t
                self.ifft_t()
                # Copy to the output array
                array_out[:,:] = self.interp_buffer_t[:,:]
            else:
                raise ValueError('Invalid input array.The input array must'
                ' be either self.spect_buffer_r or self.spect_buffer_t.')
