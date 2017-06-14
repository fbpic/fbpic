# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
Discrete Hankel Transform, with several numerical methods available.

Definition of the Hankel forward and backward transform of order p :
g(\nu) = 2 \pi \int_0^\infty f(r) J_p( 2 \pi \nu r) r dr
f( r ) = 2 \pi \int_0^\infty g(\nu) J_p( 2 \pi \nu r) \nu d\nu

Several method exist to discretize this transform, with usually non-uniform
discretization grids in r and \nu.

Available methods :
-------------------

- FHT (Fast Hankel Transform) :
  In theory, calculates the transform in N log(N) time, but is not appropriate
  for successions of forward transformation and backward transformation
  (accuracy issues).
  The discretization r grid is exponentially spaced, with considerable
  oversampling close to the axis.

- QDHT (Quasi-Discrete Hankel Transform) :
  Calculates the transform in N^2 time. Ensures that the succession
  of a forward and backward transformation retrieves the original function
  (with a very good accuracy).
  The discretization r grid corresponds to the zeros of the Bessel function
  of order p.

- MDHT (Matrix Discrete Hankel Transform) :
  Calculates the transform in N^2 time. Ensures that the succession
  of a forward and backward transformation retrieves the original function
  (to machine precision).
  The discretization r grid is evenly spaced.

See the docstring of the DHT object for usage instructions.
"""
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.optimize import fsolve

# Check if CUDA is available, then import CUDA functions
from fbpic.cuda_utils import cuda_installed
if cuda_installed:
    from accelerate.cuda import blas as cublas
    from fbpic.cuda_utils import cuda, cuda_tpb_bpg_2d
    from .cuda_methods import cuda_copy_2d_to_2d

# The list of available methods
available_methods = [ 'QDHT', 'MDHT(m+1,m)', 'MDHT(m-1,m)', 'MDHT(m,m)']

class DHT(object) :
    """
    Class that allows to perform the Discrete Hankel Transform.

    Usage : (for a callable f for instance)
    >>> trans = DHT(0,10,1,'QDHT')
    >>> r = trans.get_r()  # Array of radial position
    >>> F = f(r)           # Calculate the values of the function
                           # At these positions
    >>> G = trans.transform(F)
    """

    def __init__(self, p, Nr, Nz, rmax, method, use_cuda=False, **kw ) :
        """
        Calculate the r (position) and nu (frequency) grid
        on which the transform will operate.

        Also store auxiliary data needed for the transform.

        Parameters :
        ------------
        p : int
        Order of the Hankel transform

        Nr, Nz : float
        Number of points in the r direction and z direction

        rmax : float
        Edge of the box in which the Hankel transform is taken
        (The function is assumed to be zero at that point.)

        method : string
        The method used to calculate the Hankel transform

        use_cuda : bool, optional
        Whether to use the GPU for the Hankel transform
        (Only available for the MDHT method)

        tpb : int, optional
        Number of threads per block, in the case where cuda is used

        kw : optional arguments to be passed in the case of the MDHT
        """

        # Check that the method is valid
        if ( method in available_methods ) == False :
            raise ValueError('Illegal method string')
        else :
            self.method = method

        # Register whether to use the GPU.
        # If yes, initialize the corresponding cuda stream
        self.use_cuda = use_cuda
        if (self.use_cuda==True) and (cuda_installed==False) :
            self.use_cuda = False
            print('** Cuda not available for Hankel transform.')
            print('** Performing the Hankel transform on the CPU.')
        if self.use_cuda :
            # Initialize a cuda stream (required by cublas)
            self.blas = cublas.Blas()
            # Initialize two buffer arrays on the GPU
            # The cuBlas API requires that these arrays be in Fortran order
            zero_array = np.zeros((Nz, Nr), dtype=np.complex128, order='F')
            self.d_in = cuda.to_device( zero_array )
            self.d_out = cuda.to_device( zero_array )
            # Initialize the threads per block and block per grid
            self.dim_grid, self.dim_block = cuda_tpb_bpg_2d(Nz, Nr)

        # Call the corresponding initialization routine
        if self.method == 'FHT' :
            self.FHT_init(p, Nr, rmax)
        elif self.method == 'QDHT' :
            self.QDHT_init(p, Nr, rmax)
        elif self.method == 'MDHT(m,m)' :
            self.MDHT_init(p, Nr, rmax, m=p, **kw)
        elif self.method == 'MDHT(m-1,m)' :
            self.MDHT_init(p, Nr, rmax, m=p+1, **kw)
        elif self.method == 'MDHT(m+1,m)' :
            self.MDHT_init(p, Nr, rmax, m=p-1, **kw)


    def get_r(self) :
        """
        Return the natural, non-uniform r grid for the chosen method

        Returns :
        ---------
        A real 1darray containing the values of the positions
        """
        return( self.r )


    def get_nu(self) :
        """
        Return the natural, non-uniform nu grid for the chosen method

        Returns :
        ---------
        A real 1darray containing the values of the frequencies
        """
        return( self.nu )


    def transform( self, F, G ) :
        """
        Perform the Hankel transform of F, according to the method
        chosen at initialization.

        Parameters :
        ------------
        F : 2darray of real or complex values
        Array containing the discrete values of the function for which
        the discrete Hankel transform is to be calculated.

        G : 2darray of real or complex values
        Array where the result will be stored

        Returns :
        ---------
        A ndarray of the same shape as F, containing the value of the transform
        """

        # Perform the appropriate transform, depending on the method

        if self.method == 'FHT' :
            G[:,:] = self.FHT_transform(F)
        elif self.method == 'QDHT' :
            G[:,:] = self.QDHT_transform(F)
        elif self.method in [ 'MDHT(m,m)', 'MDHT(m-1,m)', 'MDHT(m+1,m)' ] :
            self.MDHT_transform(F, G)

    def inverse_transform( self, G, F ) :
        """
        Perform the Hankel inverse transform of G, according to the method
        chosen at initialization.

        Parameters :
        ------------
        G : 2darray of real or complex values
        Array containing the values of the function for which
        the discrete inverse Hankel transform is to be calculated.

        F : 2darray of real or complex values
        Array where the result will be stored

        Returns :
        ---------
        A ndarray of the same shape as G, containing the value of the inverse
        transform
        """
        # Perform the appropriate transform, depending on the method

        if self.method == 'FHT' :
            F[:,:] = self.FHT_inverse_transform(G)
        elif self.method == 'QDHT' :
            F[:,:] = self.QDHT_inverse_transform(G)
        elif self.method in [ 'MDHT(m,m)', 'MDHT(m-1,m)', 'MDHT(m+1,m)' ] :
            self.MDHT_inverse_transform(G, F)


    def MDHT_init(self, p, N, rmax, m, d=0.5, Fw='inverse') :
        """
        Initializes the matrix DHT
        (custom Hankel transform, many different options for testing )
        Reference : see the paper associated with FBPIC

        Grid :
        r_n = (n+d)*rmax/N        (if d is not None)
        r_n = alpha_{m,n}/S   (if d is None)
        nu_n = alpha_{m,n}/(2*pi*rmax)
        where alpha_{m,n} is the n^th zero of the m^th Bessel function

        m : int
           Index of the nu grid on which to evaluate the Hankel
           transform. This can only be p-1, p or p+1 for the
           algorithm to work.

        d : float, optional
           Offset of the evenly-spaced radial grid, within one cell
           If None, this uses the zeros of the Bessel function.

        Fw : string, optional
           Method to calculate the forward transformation
           If 'symmetric', uses a symmetric formula, similar to the backward
           transformation.
           If 'inverse', inverses the matrix of the backward transformation
           to find that of the forward transformation.

        """
        # Check that m has a valid value
        if (m in [p-1, p, p+1]) == False :
            raise ValueError('m must be either p-1, p or p+1')

        # Register values of the arguments
        self.d = d
        self.Fw =Fw
        self.p = p
        self.m = m
        self.N = N
        self.rmax = rmax

        # Calculate the zeros of the Bessel function
        if m !=0 :
            # In this case, 0 is a zero of the Bessel function of order m.
            # It turns out that it is needed to reconstruct the signal for p=0.
            zeros = np.hstack( (np.array([0.]), jn_zeros(m, N)) )
        else :
            zeros = jn_zeros(m, N+1)
        last_alpha = zeros[-1] # The N+1^{th} zero
        alphas = zeros[:-1]    # The N first zeros

        # Calculate the spectral grid
        self.nu = 1./(2*np.pi*rmax) * alphas

        # Calculate the spatial grid
        if d is not None :
            # Uniform grid with offset d
            self.r = (rmax*1./N) * ( np.arange(N) + d )
            S = last_alpha  # product of the spatial and spectral bandwidth
        else :
            # Bessel-like grid
            if m == p :
                # S from Guizar-Sicairos et al., JOSA A 21 (2004)
                S = last_alpha
            else :
                # S from Kai-Ming et al., Chinese Physics B, 18 (2009)
                k = int(N/4)
                A = alphas[k]
                J = jn_zeros(m,N)[-1]
                S = abs(2./jn( m-1, alphas[k]))*np.sqrt(
                1 + ( jn( m-1, A*alphas[1:]/J )**2 / \
                    jn( m-1, alphas[1:] )**2 ).sum() )
            self.r = rmax*alphas/S

        # Calculate and store the inverse matrix invM
        # (imposed by the constraints on the DHT of Bessel modes)
        # NB : When compared with the article, all the matrices here
        # are calculated in transposed form. This is done so as to use the
        # `dot` and `gemm` functions, in the `transform` method.
        self.invM = np.empty((N, N))
        if p == m :
            p_denom = p+1
        else :
            p_denom = p
        denom = np.pi * rmax**2 * jn( p_denom, alphas)**2
        num = jn( p, 2*np.pi* self.r[np.newaxis,:]*self.nu[:,np.newaxis] )
        # Get the inverse matrix
        if m!=0 and p!=0 :
            self.invM[1:, :] = num[1:, :] / denom[1:, np.newaxis]
            self.invM[0, :] = 0.
        else :
            self.invM[:, :] = num[:, :] / denom[:, np.newaxis]

        # Calculate the matrix M
        self.M = np.empty((N, N))
        if Fw == 'inverse' :
            if m !=0 and p != 0 :
                self.M[:, 1:] = np.linalg.pinv( self.invM[1:, :] )
                self.M[:, 0] = 0.
            else :
                self.M = np.linalg.inv( self.invM )

        if Fw == 'symmetric' :
            self.M = (2*np.pi*rmax**2/S)**2 * self.invM.T

        # Copy the arrays to the GPU if needed
        if self.use_cuda :
            # Conversion to complex and Fortran order
            # is needed for the cuBlas API
            self.d_M = cuda.to_device(
                np.asfortranarray( self.M, dtype=np.complex128 ) )
            self.d_invM = cuda.to_device(
                np.asfortranarray( self.invM, dtype=np.complex128 ) )


    def MDHT_transform( self, F, G ) :
        """
        Performs the MDHT of F and stores the result in G
        Reference: see the paper associated with FBPIC

        F : 2darray of real or complex values
        Array containing the values from which to compute the DHT

        G : 2darray of real or complex values
        Array where the result will be stored
        """

        # Perform the matrix product with M
        if self.use_cuda :
            # Check that the shapes agree
            if (F.shape!=self.d_in.shape) or (G.shape!=self.d_out.shape) :
                raise ValueError('The shape of F or G is different from '
                                 'the shape chosen at initialization.')
            # Convert the C-order F array to the Fortran-order d_in array
            cuda_copy_2d_to_2d[self.dim_grid, self.dim_block]( F, self.d_in )
            # Perform the matrix product using cuBlas
            self.blas.gemm( 'N', 'N', F.shape[0], F.shape[1],
                   F.shape[1], 1.0, self.d_in, self.d_M, 0., self.d_out )
            # Convert the Fortran-order d_out array to the C-order G array
            cuda_copy_2d_to_2d[self.dim_grid, self.dim_block]( self.d_out, G )

        else :
            np.dot( F, self.M, out=G )


    def MDHT_inverse_transform( self, G, F ) :
        """
        Performs the MDHT of G and stores the result in F
        Reference: see the paper associated with FBPIC

        G : 2darray of real or complex values
        Array containing the values from which to compute the DHT

        F : 2darray of real or complex values
        Array where the result will be stored
        """

        # Perform the matrix product with invM
        if self.use_cuda :
            # Check that the shapes agree
            if (G.shape!=self.d_in.shape) or (F.shape!=self.d_out.shape) :
                raise ValueError('The shape of F or G is different from '
                                 'the shape chosen at initialization.')
            # Convert the C-order G array to the Fortran-order d_in array
            cuda_copy_2d_to_2d[self.dim_grid, self.dim_block](G, self.d_in )
            # Perform the matrix product using cuBlas
            self.blas.gemm( 'N', 'N', G.shape[0], G.shape[1],
                   G.shape[1], 1.0, self.d_in, self.d_invM, 0., self.d_out )
            # Convert the Fortran-order d_out array to the C-order G array
            cuda_copy_2d_to_2d[self.dim_grid, self.dim_block]( self.d_out, F )

        else :
            np.dot( G, self.invM, out=F )


    def QDHT_init(self,p,N,rmax) :
        """
        Calculate r and nu for the QDHT.
        Reference : Guizar-Sicairos et al., J. Opt. Soc. Am. A 21 (2004)

        Also store the auxilary matrix T and vectors J and J_inv required for
        the transform.

        Grid : r_n = alpha_{p,n}*rmax/alpha_{p,N+1}
        where alpha_{p,n} is the n^th zero of the p^th Bessel function
        """

        # Calculate the zeros of the Bessel function
        zeros = jn_zeros(p,N+1)

        # Calculate the grid
        last_alpha = zeros[-1] # The N+1^{th} zero
        alphas = zeros[:-1]    # The N first zeros
        numax = last_alpha/(2*np.pi*rmax)
        self.N = N
        self.rmax = rmax
        self.numax = numax
        self.r = rmax*alphas/last_alpha
        self.nu = numax*alphas/last_alpha

        # Calculate and store the vector J
        J = abs( jn(p+1,alphas) )
        self.J = J
        self.J_inv = 1./J

        # Calculate and store the matrix T
        denom = J[:,np.newaxis]*J[np.newaxis,:]*last_alpha
        num = 2*jn( p, alphas[:,np.newaxis]*alphas[np.newaxis,:]/last_alpha )
        self.T = num/denom

    def QDHT_transform( self, F ) :
        """
        Performs the QDHT of F and returns the results.
        Reference : Guizar-Sicairos et al., J. Opt. Soc. Am. A 21 (2004)

        F : ndarray of real or complex values
        Array containing the values from which to compute the FHT.
        """

        # Multiply the input function by the vector J_inv
        F = array_multiply( F, self.J_inv*self.rmax, -1 )

        # Perform the matrix product with T
        G = np.dot( F, self.T )

        # Multiply the result by the vector J
        G = array_multiply( G, self.J / self.numax, -1 )

        return( G )

    def QDHT_inverse_transform( self, G ) :
        """
        Performs the QDHT of G and returns the results.
        Reference : Guizar-Sicairos et al., J. Opt. Soc. Am. A 21 (2004)

        G : ndarray of real or complex values
        Array containing the values from which to compute the DHT.
        """

        # Multiply the input function by the vector J_inv
        G =  array_multiply( G, self.J_inv*self.numax, -1 )

        # Perform the matrix product with T
        F = np.dot( G, self.T )

        # Multiply the result by the vector J
        F = array_multiply( F, self.J / self.rmax, -1 )

        return( F )

    def FHT_init(self, p, N, rmax) :
        """
        Calculate r and nu for the FHT.
        Reference : A. Siegman, Optics Letters 1 (1977)

        Also store the auxilary vector fft_j_convol needed for the
        transformation.

        Grid : r = dr*exp( alpha*n )
          with rmax = dr*exp( alpha*N )
          and exp( alpha*N )*(1-exp(-alpha))
         """

        # Minimal number of points of the r grid, within one
        # oscillation of the highest frequency of the nu grid
        # (Corresponds to K1 and K2, in Siegman's article, with
        # K1 = K2 = K here.)
        K = 4.

        # Find the alpha corresponding to N
        alpha = fsolve( lambda x : np.exp(x*N)*(1-np.exp(-x)) - 1,
                        x0 = 1. )[0]
        # Corresponding dr
        dr = rmax/np.exp( alpha*N )
        # The r and nu grid.
        self.N = N
        self.r = dr*np.exp( alpha*np.arange(N) )
        self.nu = 1./(K*rmax)*np.exp( alpha*np.arange(N) )

        # Store vector containing the convolutional filter
        r_nu = dr/(K*rmax) * np.exp( alpha*np.arange(2*N) )
        j_convol = 2*np.pi* alpha * r_nu * jn( p, 2*np.pi * r_nu )
        self.fft_j_convol = np.fft.ifft( j_convol )


    def FHT_transform( self, F ) :
        """
        Performs the FHT of F and returns the results.
        Reference : A. Siegman, Optics Letters 1 (1977)

        F : ndarray of real or complex values
        Array containing the values from which to compute the FHT.
        """
        # This function calculates the convolution of j_convol and F
        # by multiplying their fourier transform

        # Multiply F by self.r
        rF = array_multiply( F, self.r, -1 )
        # Perform the FFT of rF with 0 padding from N to 2N along axis
        fft_rF = np.fft.fft( rF, axis=-1, n=2*self.N )

        # Mutliply fft_rF and fft_j_convol, along axis
        fft_nuG = array_multiply( fft_rF, self.fft_j_convol, -1 )

        # Perform the FFT again
        nuG_large = np.fft.fft( fft_nuG, axis=-1 )
        # Discard the N last values along axis, and divide by nu
        nuG = np.split( nuG_large, 2, axis=-1 )[0]
        G  = array_multiply( nuG, 1./self.nu, -1 )

        return( G )

    def FHT_inverse_transform( self, G ) :
        """
        Performs the inverse FHT of G and returns the results.
        Reference : A. Siegman, Optics Letters 1 (1977)

        G : ndarray of real or complex values
        Array containing the values from which to compute the inverse FHT.
        """
        # This function calculates the convolution of j_convol and G
        # by multiplying their fourier transform

        # Multiply G by self.nu, along axis
        nuG = array_multiply( G, self.nu, -1 )
        # Perform the FFT of nuG with 0 padding from N to 2N along axis
        fft_nuG = np.fft.fft( nuG, axis=-1, n=2*self.N )

        # Mutliply fft_nuG and fft_j_convol, along axis
        fft_rF = array_multiply( fft_nuG, self.fft_j_convol, -1 )

        # Perform the FFT again
        rF_large = np.fft.fft( fft_rF, axis=-1 )
        # Discard the N last values along axis, and divide by r
        rF = np.split( rF_large, 2, axis=-1 )[0]
        F  = array_multiply( rF, 1./self.r, -1 )

        return( F )

# ------------------
# Utility functions
# ------------------

def array_multiply( a, v, axis ) :
    """
    Mutliply the array `a` (of any shape) with the vector
    `v` along the axis `axis`
    The axis `axis` of `a` should have the same length as `v`

    Parameters :
    ------------
    a : ndarray (real or complex values)
    The array to be multiplied by v

    v : 1darray
    The 1d vector used for the multiplication

    axis : int
    The axis of `a` along which the multiplication is carried out

    Returns :
    --------
    A matrix of the same shape as a
    """

    if a.ndim > 1 and axis!=-1 :
        # Carry out the product by moving the axis `axis`
        # (This is done because the shape of the array a is
        # unknown, and thus the syntax for multiplying is unclear.
        # The axis -1 is chosen because it is the fastest index
        # for multiplying.)
        r = a.swapaxes(-1,axis)
    else :
        r = a
    # Calculate the product
    r = r*v
    # If needed swap the axes back
    if a.ndim > 1 and axis!=-1 :
        r = r.swapaxes(-1,axis)

    return(r)
