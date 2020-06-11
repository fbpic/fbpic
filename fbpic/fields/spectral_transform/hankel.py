# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of FBPIC (Fourier-Bessel Particle-In-Cell code).
It defines the class that performs the Hankel transform.

Definition of the Hankel forward and backward transform of order p:
g(\nu) = 2 \pi \int_0^\infty f(r) J_p( 2 \pi \nu r) r dr
f( r ) = 2 \pi \int_0^\infty g(\nu) J_p( 2 \pi \nu r) \nu d\nu d
"""
import numpy as np
from scipy.special import jn, jn_zeros

# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
from .numba_methods import numba_copy_2dC_to_2dR, numba_copy_2dR_to_2dC
if cuda_installed:
    from fbpic.utils.cuda import cuda_tpb_bpg_2d, cuda_gpu_model
    from .cuda_methods import cuda_copy_2dC_to_2dR, cuda_copy_2dR_to_2dC
    import cupy
    from cupy.cuda import device, cublas


class DHT(object):
    """
    Class that allows to perform the Discrete Hankel Transform.
    """

    def __init__(self, p, m, Nr, Nz, rmax, use_cuda=False ):
        """
        Calculate the r (position) and nu (frequency) grid
        on which the transform will operate.

        Also store auxiliary data needed for the transform.

        Parameters:
        ------------
        p: int
        Order of the Hankel transform

        m: int
        The azimuthal mode for which the Hankel transform is calculated

        Nr, Nz: float
        Number of points in the r direction and z direction

        rmax: float
        Edge of the box in which the Hankel transform is taken
        (The function is assumed to be zero at that point.)

        use_cuda: bool, optional
        Whether to use the GPU for the Hankel transform
        """
        # Register whether to use the GPU.
        # If yes, initialize the corresponding cuda object
        self.use_cuda = use_cuda
        if (self.use_cuda==True) and (cuda_installed==False):
            self.use_cuda = False
            print('** Cuda not available for Hankel transform.')
            print('** Performing the Hankel transform on the CPU.')

        # Check that m has a valid value
        if (m in [p-1, p, p+1]) == False:
            raise ValueError('m must be either p-1, p or p+1')

        # Register values of the arguments
        self.p = p
        self.m = m
        self.Nr = Nr
        self.rmax = rmax
        self.Nz = Nz

        # Calculate the zeros of the Bessel function
        if m !=0:
            # In this case, 0 is a zero of the Bessel function of order m.
            # It turns out that it is needed to reconstruct the signal for p=0.
            alphas = np.hstack( (np.array([0.]), jn_zeros(m, Nr-1)) )
        else:
            alphas = jn_zeros(m, Nr)

        # Calculate the spectral grid
        self.nu = 1./(2*np.pi*rmax) * alphas

        # Calculate the spatial grid (Uniform grid with an half-cell offset)
        self.r = (rmax*1./Nr) * ( np.arange(Nr) + 0.5 )

        # Calculate and store the inverse matrix invM
        # (imposed by the constraints on the DHT of Bessel modes)
        # NB: When compared with the FBPIC article, all the matrices here
        # are calculated in transposed form. This is done so as to use the
        # `dot` and `gemm` functions, in the `transform` method.
        self.invM = np.empty((Nr, Nr))
        if p == m:
            p_denom = p+1
        else:
            p_denom = p
        denom = np.pi * rmax**2 * jn( p_denom, alphas)**2
        num = jn( p, 2*np.pi* self.r[np.newaxis,:]*self.nu[:,np.newaxis] )
        # Get the inverse matrix
        if m!=0:
            self.invM[1:, :] = num[1:, :] / denom[1:, np.newaxis]
            # In this case, the functions are represented by Bessel functions
            # *and* an additional mode (below) which satisfies the same
            # algebric relations for curl/div/grad as the regular Bessel modes,
            # with the value kperp=0.
            # The normalization of this mode is arbitrary, and is chosen
            # so that the condition number of invM is close to 1
            if p==m-1:
                self.invM[0, :] = self.r**(m-1) * 1./( np.pi * rmax**(m+1) )
            else:
                self.invM[0, :] = 0.
        else :
            self.invM[:, :] = num[:, :] / denom[:, np.newaxis]

        # Calculate the matrix M by inverting invM
        self.M = np.empty((Nr, Nr))
        if m !=0 and p != m-1:
            self.M[:, 1:] = np.linalg.pinv( self.invM[1:,:] )
            self.M[:, 0] = 0.
        else:
            self.M = np.linalg.inv( self.invM )

        # Copy the matrices to the GPU if needed
        if self.use_cuda:
            self.d_M = cupy.asarray( self.M )
            self.d_invM = cupy.asarray( self.invM )

        # Initialize buffer arrays to store the complex Nz x Nr grid
        # as a real 2Nz x Nr grid, before performing the matrix product
        # (This is because a matrix product of reals is faster than a matrix
        # product of complexs, and the real-complex conversion is negligible.)
        if not self.use_cuda:
            # Initialize real buffer arrays on the CPU
            zero_array = np.zeros((2*Nz, Nr), dtype=np.float64)
            self.array_in = zero_array.copy()
            self.array_out = zero_array.copy()
        else:
            # Initialize real buffer arrays on the GPU
            zero_array = np.zeros((2*Nz, Nr), dtype=np.float64)
            self.d_in = cupy.asarray( zero_array )
            self.d_out = cupy.asarray( zero_array )
            # Initialize cuBLAS
            self.blas = device.get_cublas_handle()
            # Set optimal number of CUDA threads per block
            # for copy 2d real/complex (determined empirically)
            copy_tpb = (8,32) if cuda_gpu_model == "V100" else (2,16)
            # Initialize the threads per block and block per grid
            self.dim_grid, self.dim_block = cuda_tpb_bpg_2d(Nz, Nr, *copy_tpb)


    def get_r(self):
        """
        Return the r grid

        Returns:
        ---------
        A real 1darray containing the values of the positions
        """
        return( self.r )


    def get_nu(self):
        """
        Return the natural, non-uniform nu grid

        Returns:
        ---------
        A real 1darray containing the values of the frequencies
        """
        return( self.nu )


    def transform( self, F, G ):
        """
        Perform the Hankel transform of F.

        Parameters:
        ------------
        F: 2darray of complex values
        Array containing the discrete values of the function for which
        the discrete Hankel transform is to be calculated.

        G: 2darray of complex values
        Array where the result will be stored
        """
        # Perform the matrix product with M
        if self.use_cuda:
            # Convert C-order, complex array `F` to F-order, real `d_in`
            cuda_copy_2dC_to_2dR[self.dim_grid, self.dim_block]( F, self.d_in )
            # Call cuBLAS gemm kernel
            cublas.dgemm(self.blas, 0, 0, self.Nr, 2*self.Nz, self.Nr,
                         1, self.d_M.data.ptr, self.Nr,
                            self.d_in.data.ptr, self.Nr,
                         0, self.d_out.data.ptr, self.Nr)
            # Convert F-order, real `d_out` to the C-order, complex `G`
            cuda_copy_2dR_to_2dC[self.dim_grid, self.dim_block]( self.d_out, G )
        else:
            # Convert complex array `F` to real array `array_in`
            numba_copy_2dC_to_2dR( F, self.array_in )
            # Perform real matrix product (faster than complex matrix product)
            np.dot( self.array_in, self.M, out=self.array_out )
            # Convert real array `array_out` to complex array `G`
            numba_copy_2dR_to_2dC( self.array_out, G )


    def inverse_transform( self, G, F ):
        """
        Performs the MDHT of G and stores the result in F
        Reference: see the paper associated with FBPIC

        G: 2darray of real or complex values
        Array containing the values from which to compute the DHT

        F: 2darray of real or complex values
        Array where the result will be stored
        """
        # Perform the matrix product with invM
        if self.use_cuda:
            # Convert C-order, complex array `G` to F-order, real `d_in`
            cuda_copy_2dC_to_2dR[self.dim_grid, self.dim_block](G, self.d_in )
            # Call cuBLAS gemm kernel
            cublas.dgemm(self.blas, 0, 0, self.Nr, 2*self.Nz, self.Nr,
                         1, self.d_invM.data.ptr, self.Nr,
                            self.d_in.data.ptr, self.Nr,
                         0, self.d_out.data.ptr, self.Nr)
            # Convert the F-order d_out array to the C-order F array
            cuda_copy_2dR_to_2dC[self.dim_grid, self.dim_block]( self.d_out, F )
        else:
            # Convert complex array `G` to real array `array_in`
            numba_copy_2dC_to_2dR( G, self.array_in )
            # Perform real matrix product (faster than complex matrix product)
            np.dot( self.array_in, self.invM, out=self.array_out )
            # Convert real array `array_out` to complex array `F`
            numba_copy_2dR_to_2dC( self.array_out, F )
