"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the SpectralTransformer class, which handles conversion of
the fields from the interpolation grid to the spectral grid and vice-versa.
"""
import numpy as np
import pyfftw
from .hankel import DHT

# If numbapro is installed, it potentially allows to use the GPU
try :
    from fbpic.cuda_utils import cuda_tpb_bpg_2d
    from numbapro.cudalib import cufft, cublas
    from .cuda_methods import cuda, cuda_copy_2d_to_1d, \
        cuda_copy_1d_to_2d, cuda_rt_to_pm, cuda_pm_to_rt
    cuda_installed = True
except ImportError :
    cuda_installed = False

class SpectralTransformer(object) :
    """
    Object that allows to transform the fields back and forth between the
    spectral and interpolation grid.

    Attributes :
    - dht : the discrete Hankel transform object that operates along r

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

    def __init__(self, Nz, Nr, m, rmax, nthreads=4, use_cuda=False ) :
        """
        Initializes the dht attributes, which contain auxiliary
        matrices allowing to transform the fields quickly

        Parameters
        ----------
        Nz, Nr : int
            Number of points along z and r respectively

        m : int
            Index of the mode (needed for the Hankel transform)

        rmax : float
            The size of the simulation box along r.

        nthreads : int, optional
            Number of threads for the FFTW transform
        """
        # Check whether to use the GPU
        self.use_cuda = use_cuda
        
        # Initialize the DHT (local implementation, see hankel_dt.py)
        if use_cuda :
            print('Preparing the Hankel Transforms for mode %d on the GPU' %m)
        else :
            print('Preparing the Hankel Transforms for mode %d on the CPU' %m)

        self.dht0 = DHT(  m, Nr, Nz, rmax, 'MDHT(m,m)', d=0.5, Fw='inverse',
                           use_cuda=self.use_cuda )
        self.dhtp = DHT(m+1, Nr, Nz, rmax, 'MDHT(m+1,m)', d=0.5, Fw='inverse',
                           use_cuda=self.use_cuda )
        self.dhtm = DHT(m-1, Nr, Nz, rmax, 'MDHT(m-1,m)', d=0.5, Fw='inverse',
                           use_cuda=self.use_cuda )

        if self.use_cuda :
            # Initialize the FFTW on the GPU
            print('Preparing FFTW for mode %d on the GPU' %m)

            # Initialize the dimension of the grid and blocks
            self.dim_grid, self.dim_block = cuda_tpb_bpg_2d( Nz, Nr)
            
            # Initialize the spectral buffers
            self.spect_buffer_r = cuda.device_array((Nz, Nr), 
                                                    dtype=np.complex128)
            self.spect_buffer_t = cuda.device_array((Nz, Nr), 
                                                    dtype=np.complex128)
            # Different names for same object (for economy of memory)
            self.spect_buffer_p = self.spect_buffer_r
            self.spect_buffer_m = self.spect_buffer_t
            # Initialize 1d buffer for cufft
            self.buffer1d_in = cuda.device_array((Nz*Nr,), 
                                                 dtype=np.complex128)
            self.buffer1d_out = cuda.device_array((Nz*Nr,), 
                                                  dtype=np.complex128)

            # Initialize the cuda libraries object
            self.fft = cufft.FFTPlan( shape=(Nz,), itype=np.complex128,
                                      otype=np.complex128, batch=Nr )
            self.blas = cublas.Blas()   # For normalization of the iFFT
            self.inv_Nz = 1./Nz         # For normalization of the iFFT
        else :
            # Initialize the FFTW on the CPU
            print('Preparing FFTW for mode %d on the CPU' %m)
            
            # Two buffers and FFTW objects are initialized, since
            # spect2interp_vect and interp2spect_vect require two separate FFT
            
            # First buffer and FFTW transform
            self.interp_buffer_r = \
                pyfftw.n_byte_align_empty( (Nz,Nr), 16, 'complex128' )
            self.spect_buffer_r = \
                pyfftw.n_byte_align_empty( (Nz,Nr), 16, 'complex128' )
            self.fft_r= pyfftw.FFTW(self.interp_buffer_r, self.spect_buffer_r,
                    axes=(0,), direction='FFTW_FORWARD', threads=nthreads)
            self.ifft_r=pyfftw.FFTW(self.spect_buffer_r, self.interp_buffer_r,
                    axes=(0,), direction='FFTW_BACKWARD', threads=nthreads)
            # Use different name for same object (for economy of memory)
            self.spect_buffer_p = self.spect_buffer_r 
            
            # Second buffer and FFTW transform
            self.interp_buffer_t = \
                pyfftw.n_byte_align_empty( (Nz,Nr), 16, 'complex128' )
            self.spect_buffer_t = \
                pyfftw.n_byte_align_empty( (Nz,Nr), 16, 'complex128' )
            self.fft_t= pyfftw.FFTW(self.interp_buffer_t, self.spect_buffer_t,
                    axes=(0,), direction='FFTW_FORWARD', threads=nthreads )
            self.ifft_t=pyfftw.FFTW(self.spect_buffer_t, self.interp_buffer_t,
                    axes=(0,), direction='FFTW_BACKWARD', threads=nthreads)
            # Use different name for same object (for economy of memory)    
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
        if self.use_cuda :
            # Perform the inverse FFT on the GPU
            # (The cuFFT API requires 1D arrays)            
            cuda_copy_2d_to_1d[self.dim_grid, self.dim_block](
                self.spect_buffer_r, self.buffer1d_in )
            self.fft.inverse( self.buffer1d_in, out=self.buffer1d_out )
            self.blas.scal( self.inv_Nz, self.buffer1d_out ) # Normalization
            cuda_copy_1d_to_2d[self.dim_grid, self.dim_block](
                self.buffer1d_out, interp_array )
        else :
            # Perform the inverse FFT on the CPU, using preallocated buffers
            self.interp_buffer_r = self.ifft_r()
            # Copy to the output array
            interp_array[:,:] = self.interp_buffer_r[:,:]  

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
            # (It is important to write the affectation in the following way,
            # since self.spect_buffer_p and self.spect_buffer_r actually point
            # to the same object, for memory economy)
            self.spect_buffer_r[:,:], self.spect_buffer_t[:,:] = \
                    ( self.spect_buffer_p + self.spect_buffer_m), \
                1.j*( self.spect_buffer_p - self.spect_buffer_m)

        # Finally perform the FFT (along axis 0, which corresponds to z)
        if self.use_cuda :
            # Perform the inverse FFT on spect_buffer_r
            # (The cuFFT API requires 1D arrays)
            cuda_copy_2d_to_1d[self.dim_grid, self.dim_block](
                self.spect_buffer_r, self.buffer1d_in )
            self.fft.inverse( self.buffer1d_in, out=self.buffer1d_out )
            self.blas.scal( self.inv_Nz, self.buffer1d_out ) # Normalization
            cuda_copy_1d_to_2d[self.dim_grid, self.dim_block](
                self.buffer1d_out, interp_array_r )
            
            # Perform the inverse FFT on spect_buffer_t
            # (The cuFFT API requires 1D arrays)
            cuda_copy_2d_to_1d[self.dim_grid, self.dim_block](
                self.spect_buffer_t, self.buffer1d_in )
            self.fft.inverse( self.buffer1d_in, out=self.buffer1d_out )
            self.blas.scal( self.inv_Nz, self.buffer1d_out ) # Normalization
            cuda_copy_1d_to_2d[self.dim_grid, self.dim_block](
                self.buffer1d_out, interp_array_t )
        else :
            # Perform the FFT on the CPU, using the preallocated buffers
            self.interp_buffer_r = self.ifft_r()
            self.interp_buffer_t = self.ifft_t()
            # Copy to the output array
            interp_array_r[:,:] = self.interp_buffer_r[:,:]
            interp_array_t[:,:] = self.interp_buffer_t[:,:]
        

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
        if self.use_cuda :
            # Perform the FFT on the GPU
            # (The cuFFT API requires 1D arrays)
            cuda_copy_2d_to_1d[self.dim_grid, self.dim_block](
                interp_array, self.buffer1d_in )
            self.fft.forward( self.buffer1d_in, out=self.buffer1d_out )
            cuda_copy_1d_to_2d[self.dim_grid, self.dim_block](
                self.buffer1d_out, self.spect_buffer_r )
        else :
            # Perform the FFT on the CPU
            self.interp_buffer_r[:,:] = interp_array #Copy the input array
            self.spect_buffer_r = self.fft_r()
        
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
        if self.use_cuda :
            # Perform the FFT on the GPU for interp_array_r
            # (The cuFFT API requires 1D arrays)
            cuda_copy_2d_to_1d[self.dim_grid, self.dim_block](
                interp_array_r, self.buffer1d_in )
            self.fft.forward( self.buffer1d_in, out=self.buffer1d_out )
            cuda_copy_1d_to_2d[self.dim_grid, self.dim_block](
                self.buffer1d_out, self.spect_buffer_r )

            # Perform the FFT on the GPU for interp_array_t
            # (The cuFFT API requires 1D arrays)
            cuda_copy_2d_to_1d[self.dim_grid, self.dim_block](
                interp_array_t, self.buffer1d_in )
            self.fft.forward( self.buffer1d_in, out=self.buffer1d_out )
            cuda_copy_1d_to_2d[self.dim_grid, self.dim_block](
                self.buffer1d_out, self.spect_buffer_t )
        else :
            # Perform the FFT on the CPU
            
            # First copy the input array to the preallocated buffers
            self.interp_buffer_r[:,:] = interp_array_r
            self.interp_buffer_t[:,:] = interp_array_t
            # Then perform the FFT
            self.spect_buffer_r = self.fft_r()
            self.spect_buffer_t = self.fft_t()

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
            # Combine them on the CPU
            # (It is important to write the affectation in the following way,
            # since self.spect_buffer_p and self.spect_buffer_r actually point
            # to the same object, for memory economy.)
            self.spect_buffer_p[:,:], self.spect_buffer_m[:,:] = \
                0.5*( self.spect_buffer_r - 1.j*self.spect_buffer_t ), \
                0.5*( self.spect_buffer_r + 1.j*self.spect_buffer_t )
        
        # Perform the inverse DHT (along axis -1, which corresponds to r)
        self.dhtp.transform( self.spect_buffer_p, spect_array_p )
        self.dhtm.transform( self.spect_buffer_m, spect_array_m )

