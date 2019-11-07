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
from fbpic.utils.cuda import cuda, cuda_installed, cupy_installed
if cupy_installed:
    import cupy
    from cupy.cuda import cufft
    
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
            self.fft = cufft.PlanNd(shape=(Nz,),
                                    istride=Nr,
                                    ostride=Nr,
                                    inembed=(Nz,),
                                    onembed=(Nz,),
                                    idist=1,
                                    odist=1,
                                    fft_type=cufft.CUFFT_Z2Z,
                                    batch=Nr)
            self.inv_Nz = 1./Nz # For normalization of the iFFT

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
            # If the arrays are on CPU, copy them to GPU
            # Use `synchronize` in order to avoid that the copy
            # happen while a Numba GPU kernel runs
            if isinstance(array_in, np.ndarray) or \
               isinstance(array_out, np.ndarray):
                cuda.synchronize()
                d_in = cupy.asarray(array_in)
                d_out = cupy.asarray(array_out)
            else:
                d_in = array_in
                d_out = array_out
            # Perform forward FFT
            self.fft.fft(d_in, d_out, cufft.CUFFT_FORWARD)
            # Copy back to CPU if needed
            if isinstance(array_out, np.ndarray):
                d_out.get(out=array_out)
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
            # If the arrays are on CPU, copy them to GPU
            # Use `synchronize` in order to avoid that the copy
            # happen while a Numba GPU kernel runs
            if isinstance(array_in, np.ndarray) or \
               isinstance(array_out, np.ndarray):
                cuda.synchronize()
                d_in = cupy.asarray(array_in)
                d_out = cupy.asarray(array_out)
            else:
                d_in = array_in
                d_out = array_out
            # Perform forward FFT
            self.fft.fft(d_in, d_out, cufft.CUFFT_INVERSE)
            # Normalize inverse FFT
            cupy.multiply(d_out, self.inv_Nz, out=d_out)
            # Copy back to CPU if needed
            if isinstance(array_out, np.ndarray):
                d_out.get(out=array_out)
        elif self.use_mkl:
            # Perform the inverse FFT on the CPU using MKL
            self.mklfft.inverse_transform( array_in, array_out )
        else :
            # Perform the inverse FFT on the CPU using FFTW
            self.ifft.update_arrays( new_input_array=array_in,
                                    new_output_array=array_out )
            self.ifft()
