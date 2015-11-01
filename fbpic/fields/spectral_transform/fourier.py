# Add comments for :
#  - this file
#  - the class
#  - the methods
import numpy as np
import pyfftw
try :
    from numbapro.cudalib import cufft, cublas
    from fbpic.cuda_utils import cuda_tpb_bpg_2d
    from .cuda_methods import cuda, cuda_copy_2d_to_1d, cuda_copy_1d_to_2d
    cuda_installed = True
except ImportError :
    cuda_installed = False

class FFT(object):

    def __init__(self, Nr, Nz, use_cuda=False, nthreads=4 ):
        """
        nthreads : int, optional
            Number of threads for the FFTW transform
        """
        
        self.use_cuda = use_cuda
        if (self.use_cuda==True) and (cuda_installed==False) :
            self.use_cuda = False
            print '** Cuda not available for Fourier transform.'
            print '** Performing the Fourier transform on the CPU.'

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
        return( self.spect_buffer_r, self.spect_buffer_t )

    def inverse_transform( self, array_in, array_out ):
        
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
                self.ifft_r()
                # Copy to the output array
                array_out[:,:] = self.interp_buffer_r[:,:]
            elif array_in is self.spect_buffer_t:
                self.ifft_t()
                # Copy to the output array
                array_out[:,:] = self.interp_buffer_t[:,:]                
            else:
                raise ValueError('Invalid input array.The input array must'
                ' be either self.spect_buffer_r or self.spect_buffer_t.')

    def transform( self, array_in, array_out ):
        
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
                # Then perform the FFT
                self.fft_r()
            elif array_out is self.spect_buffer_t:
                # First copy the input array to the preallocated buffers
                self.interp_buffer_t[:,:] = array_in
                # Then perform the FFT
                self.fft_t()
            else:
                raise ValueError('Invalid output array.The output array '
                'must be either self.spect_buffer_r or self.spect_buffer_t.')
