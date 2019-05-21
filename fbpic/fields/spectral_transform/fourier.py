# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Partile-In-Cell code (FB-PIC)
It defines the FFT object, which performs Fourier transforms along the axis 0,
and is used in spectral_transformer.py
"""
import numpy as np
import numba
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    from fbpic.utils.cuda import cuda, cupy, cuda_tpb_bpg_2d
    from cupy.cuda import cufft
    from .cuda_methods import cuda_copy_2d_to_1d, cuda_copy_1d_to_2d

# Check if the MKL FFT is available
try:
    from .mkl_fft import MKLFFT
    mkl_installed = True
except OSError:
    import pyfftw
    mkl_installed = False

class FFT(object):
    """
    Object that performs Fourier transform of 2D arrays along the z axis,
    (axis 0) either on the CPU (using pyfftw) or on the GPU (using cufft)

    See the methods `transform` and `inverse transform` for more information
    """

    def __init__(self, Nr, Nz, use_cuda=False, nthreads=None ):
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
            Number of threads for the FFTW transform.
            If None, the default number of threads of numba is used
            (environment variable NUMBA_NUM_THREADS)
        """
        # Check whether to use cuda
        self.use_cuda = use_cuda
        if (self.use_cuda is True) and (cuda_installed is False) :
            self.use_cuda = False
            print('** Cuda not available for Fourier transform.')
            print('** Performing the Fourier transform on the CPU.')

        # Check whether to use MKL
        self.use_mkl = mkl_installed

        # Initialize the object for calculation on the GPU
        if self.use_cuda:
            # Initialize the CUDA FFT plan object
            self.fft = cufft.Plan1d(Nz, cufft.CUFFT_Z2Z, Nr)
            self.inv_Nz = 1./Nz         # For normalization of the iFFT

        # Initialize the object for calculation on the CPU
        else:
            # For MKL FFT
            if self.use_mkl:
                # Initialize the MKL plan with dummy array
                spect_buffer = np.zeros( (Nz, Nr), dtype=np.complex128 )
                self.mklfft = MKLFFT( spect_buffer )

            # For FFTW
            else:
                # Determine number of threads
                if nthreads is None:
                    # Get the default number of threads for numba
                    nthreads = numba.config.NUMBA_NUM_THREADS
                # Initialize the FFT plan with dummy arrays
                interp_buffer = np.zeros( (Nz, Nr), dtype=np.complex128 )
                spect_buffer = np.zeros( (Nz, Nr), dtype=np.complex128 )
                self.fft = pyfftw.FFTW( interp_buffer, spect_buffer,
                        axes=(0,), direction='FFTW_FORWARD', threads=nthreads)
                self.ifft = pyfftw.FFTW( spect_buffer, interp_buffer,
                        axes=(0,), direction='FFTW_BACKWARD', threads=nthreads)


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
            # Synchronize automatically managed
            # Numba CUDA stream (if array_in was
            # a numpy array on CPU initially)
            if isinstance(array_in, np.ndarray):
                cuda.synchronize()
            # Transform to cupy arrays
            d_in = cupy.asarray(array_in)
            d_out = cupy.asarray(array_out)
            # reshape to 1D arrays for cuFFT
            d_in_r = d_in.reshape(d_in.size, order='F')
            d_out_r = d_out.reshape(d_out.size, order='F')
            # Perform forward FFT
            self.fft.fft(d_in_r, d_out_r, cufft.CUFFT_FORWARD)
            # Reshape and write to intial array
            d_out[:,:] = d_out_r.reshape(
                d_in.shape[0], d_in.shape[1], order='F')
            # Copy cupy GPU array to host (if array_out 
            # was a numpy array on CPU initially)
            if isinstance(array_out, np.ndarray):
                array_out[:,:] =  d_out.get()
        elif self.use_mkl:
            # Perform the FFT on the CPU using MKL
            self.mklfft.transform( array_in, array_out )
        else :
            # Perform the FFT on the CPU using FFTW
            self.fft.update_arrays( new_input_array=array_in,
                                    new_output_array=array_out )
            self.fft()

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
            # Synchronize automatically managed
            # Numba CUDA stream (if array_in was
            # a numpy array on CPU initially)
            if isinstance(array_in, np.ndarray):
                cuda.synchronize()
            # Transform to cupy arrays
            d_in = cupy.asarray(array_in)
            d_out = cupy.asarray(array_out)
            # reshape to 1D arrays for cuFFT
            d_in_r = d_in.reshape(d_in.size, order='F')
            d_out_r = d_out.reshape(d_out.size, order='F')
            # Perform forward FFT
            self.fft.fft(d_in_r, d_out_r, cufft.CUFFT_INVERSE)
            # Reshape and write back to intial arrays
            # Reshape and write to intial array
            d_out[:,:] = d_out_r.reshape(
                d_in.shape[0], d_in.shape[1], order='F') * self.inv_Nz
            # Copy cupy GPU array to host (if array_out 
            # was a numpy array on CPU initially)
            if isinstance(array_out, np.ndarray):
                array_out[:,:] =  d_out.get()
        elif self.use_mkl:
            # Perform the inverse FFT on the CPU using MKL
            self.mklfft.inverse_transform( array_in, array_out )
        else :
            # Perform the inverse FFT on the CPU using FFTW
            self.ifft.update_arrays( new_input_array=array_in,
                                    new_output_array=array_out )
            self.ifft()