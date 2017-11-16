# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file tests the Discrete Hankel Transform which is implemented
in hankel_dt, by performing the forward and backward Hankel Transform
of a set of known Hankel couples.

Usage :
from the top-level directory of FBPIC run
$ python tests/test_hankel_dt.py
"""

import numpy as np
import matplotlib.pyplot as plt
from fbpic.fields.spectral_transform.hankel import DHT
from scipy.special import jn, jn_zeros
from scipy.special import eval_genlaguerre
import time

available_methods = [ 'QDHT', 'MDHT(m+1,m)', 'MDHT(m-1,m)', 'MDHT(m,m)']

# Define a class for calculating the Hankel transform with the QDHT method
# This method is never used in FBPIC, but is useful for comparison
class QDHT(object):

    def __init__(self,p,N,rmax):
        """
        Calculate r and nu for the QDHT.
        Reference: Guizar-Sicairos et al., J. Opt. Soc. Am. A 21 (2004)

        Also store the auxilary matrix T and vectors J and J_inv required for
        the transform.

        Grid: r_n = alpha_{p,n}*rmax/alpha_{p,N+1}
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

    def transform( self, F, G ):
        """
        Performs the QDHT of F and returns the results.
        Reference: Guizar-Sicairos et al., J. Opt. Soc. Am. A 21 (2004)

        F: ndarray of real or complex values
        Array containing the values from which to compute the FHT.
        """
        # Multiply the input function by the vector J_inv
        F = F * (self.J_inv*self.rmax)[np.newaxis, :]

        # Perform the matrix product with T
        G[:,:] = np.dot( F, self.T )

        # Multiply the result by the vector J
        G[:,:] = G * (self.J / self.numax)[np.newaxis,:]

        return( G )

    def inverse_transform( self, G, F ):
        """
        Performs the QDHT of G and returns the results.
        Reference: Guizar-Sicairos et al., J. Opt. Soc. Am. A 21 (2004)

        G: ndarray of real or complex values
        Array containing the values from which to compute the DHT.
        """

        # Multiply the input function by the vector J_inv
        G = G * (self.J_inv*self.numax)[np.newaxis,:]

        # Perform the matrix product with T
        F[:,:] = np.dot( G, self.T )

        # Multiply the result by the vector J
        F[:,:] = F * (self.J / self.rmax)[np.newaxis,:]

        return( F )


def compare_Hankel_methods( f_analytic, g_analytic, p, Nz, Nr,
                            rmax, npts=1000, methods=available_methods ) :
    """
    Compare the discrete Hankel transform of f_analytic to the
    analytical result g_analytic.

    This function allows to see the effect of the Hankel transform on a
    *2d array*, where the Hankel transform is taken only along the last axis,
    with length Nr. Nothing happens along the other axis, which
    has length Nz.
    """

    # Plot the analytic function and its Hankel transform
    plt.figure()
    plt.subplot(121)
    r = np.linspace( rmax/npts, rmax, npts )
    plt.plot( r, f_analytic(r) )
    plt.subplot(122)
    nu = np.linspace( 0.5*Nr/rmax/npts, 0.5*Nr/rmax, npts )
    plt.plot( nu, g_analytic(nu) )

    # Test the different methods
    for method in methods :

        # Initialize transform
        if method == 'QDHT':
            dht = QDHT( p, Nr, rmax)
        elif method == 'MDHT(m,m)':
            dht = DHT( p, p, Nr, Nz, rmax )
        elif method == 'MDHT(m-1,m)':
            dht = DHT( p, p+1, Nr, Nz, rmax )
        elif method == 'MDHT(m+1,m)':
            dht = DHT( p, p-1, Nr, Nz, rmax )

        # Calculate f and g on the natural grid
        f = np.empty((Nz,Nr), dtype=np.complex128)
        f[:,:] = f_analytic( dht.r )[np.newaxis,:]
        g = np.empty((Nz,Nr), dtype=np.complex128)
        g[:,:] = g_analytic( dht.nu )[np.newaxis,:]

        # Initialize empty matrices
        f_dht = np.empty((Nz, Nr), dtype=np.complex128)
        g_dht = np.empty((Nz, Nr), dtype=np.complex128)

        # Apply the forward and backward transform
        t1 = time.time()
        dht.transform( f, g_dht )
        dht.inverse_transform( g_dht, f_dht )
        t2 = time.time()

        # Plot the results
        plt.subplot(121)
        plt.plot( dht.r, f_dht[int(Nz/2)].real, 'o', label=method )
        plt.subplot(122)
        plt.plot( dht.nu, g_dht[int(Nz/2)].real, 'o', label=method )

        # Calculate the maximum error
        error_f = np.max( abs(f_dht-f) )
        error_g = np.max( abs(g_dht-g) )
        Dt_ms = (t2-t1)/(2)*1.e3

        # Finalization of the plots
        plt.legend(loc=0)
        plt.suptitle('Hankel transform of order %d' %p)

        # Diagnostic
        print('')
        print(method + ' with %d points' % Nr )
        print('--------------------')
        print('  - Time per transform : %.3f ms' % Dt_ms )
        print('  - Max error on the Hankel transform : %.3e' % error_g )
        print("""  - Max error on the orginal function
        after back and forth transform : %.3e""" % error_f)

    plt.show()

def compare_power_p( p, rcut, N, rmax, Nz=1) :
    """
    Test the Hankel transforms for the test function :
    x -> (x/rcut)^p for x < rcut
    x -> 0   for x > rcut
    The analytical transform is
    x -> rcut J_{p+1}(2\pi rcut x) / x

    """
    print('')
    print('-------------------------------------------')
    print('Testing with power function, of exponent %d' %p )
    print('-------------------------------------------')

    def power_p( x ) :
        return( np.where( x<rcut, (x/rcut)**p, 0. ) )

    def power_p_trans(x) :
        if x[0] == 0 :
            ans = np.hstack( (
                np.array([ 2*np.pi*rcut**2*jn(p,0)/(p+2) ]),
                rcut*jn( p+1, 2*np.pi*rcut*x[1:]) /x[1:] ) )
        else :
           ans =  rcut*jn( p+1, 2*np.pi*rcut*x) / x
        return( ans )

    compare_Hankel_methods( power_p, power_p_trans, p, Nz, N, rmax)

def compare_laguerre_gauss( p, n, N, rmax, Nz=1) :
    """
    Test the Hankel transforms for the test function :
    x -> x^p L_n^p(x^2) exp(-x^2/2)
    where L_n^p is a Gauss-Laguerre polynomial
    x -> (-1)^n (2\pi) (2\pi x)^p L_n^p((2\pi x)^2) exp(-(2\pi x)^2/2)

    See Cavanagh et al., IEEE Transactions on Acoustics, Speech,
    and Signal Processing, vol. assp-27, no. 4, August 1979
    """

    print('')
    print('-----------------------------------------------------')
    print('Testing with Laguerre-Gauss polynomial of order %d %d' %(n,p) )
    print('-----------------------------------------------------')

    def laguerre_n_p( x ) :
        return( x**p * eval_genlaguerre( n, p, x**2 ) * np.exp(-x**2/2) )

    def laguerre_n_p_trans( x ) :
        return( (-1)**n * (2*np.pi) * (2*np.pi*x)**p * \
                eval_genlaguerre( n, p, (2*np.pi*x)**2 ) * \
                np.exp(-(2*np.pi*x)**2/2) )

    compare_Hankel_methods( laguerre_n_p, laguerre_n_p_trans, p,
                            Nz, N, rmax)

def compare_bessel( p, m, n, N, rmax, Nz=1) :
    """
    Test the Hankel transforms for the test function :
    x -> J_p( k_n^{m} x )
    """

    print('')
    print('-----------------------------------------------')
    print('Testing with Bessel mode order %d, with %d-th k' %(p,n) )
    print('-----------------------------------------------')

    k = jn_zeros( m, n+1 )[-1]/rmax

    def bessel_n_p( x ) :
        return( jn(p, k*x) )

    def delta( x ) :
        if p == m :
            return( np.where( abs(x - k/(2*np.pi)) < 0.1,
                          np.pi*rmax**2*jn(m+1, k*rmax)**2 , 0.) )
        if p != m :
            return( np.where( abs(x - k/(2*np.pi)) < 0.1,
                          np.pi*rmax**2*jn(p, k*rmax)**2 , 0.) )

    compare_Hankel_methods( bessel_n_p, delta, p, Nz, N, rmax)

if __name__ == '__main__' :

    Nr = 200
    Nz = 1000
    pmax = 2
    rmax = 4

    # Get the available methods for each value of p
    # For p==0, the method MDHT(m+1,m) corresponds to m=-1; never used in FBPIC
    methods = [ available_methods for p in range(pmax+1) ]
    methods[0] = \
      [ method for method in available_methods if method != 'MDHT(m+1,m)']

    for p in range(pmax+1) :
        compare_power_p( p, 1, Nr, rmax, Nz=Nz, methods=methods[p])

    for p in range(pmax+1) :
        for n in range(2) :
            compare_laguerre_gauss( p, n, Nr, rmax, methods=methods[p], Nz=Nz)

    for p in range(pmax+1) :
        compare_bessel( p, p, int(Nr*0.3), Nr, rmax,
                        methods=methods[p], Nz=Nz)
        compare_bessel( p, p, int(Nr*0.9), Nr, rmax,
                        methods=methods[p], Nz=Nz)

    for p in range(pmax+1) :
        compare_bessel( p, p+1, int(Nr*0.3), Nr, rmax,
                        methods=methods[p], Nz=Nz)
        compare_bessel( p, p+1, int(Nr*0.9), Nr, rmax,
                        methods=methods[p], Nz=Nz)
