"""
This file tests the Discrete Hankel Transform which is implemented in hankel_dt
"""

import numpy as np
import matplotlib.pyplot as plt
from fbpic.hankel_dt import DHT, available_methods
from scipy.special import jn
import time

def compare_Hankel_methods( f_analytic, g_analytic, p, N, rmax, npts=1000 ) :
    """
    Compare the discrete Hankel transform of f_analytic to the
    analytical result g_analytic. 
    """

    # Plot the analytic function and its Hankel transform
    plt.figure()
    plt.subplot(121)
    r = np.linspace( rmax/npts, rmax, npts )
    plt.plot( r, f_analytic(r) )
    plt.subplot(122)
    nu = np.linspace( 0.5*N/rmax/npts, 0.5*N/rmax, npts )
    plt.plot( nu, g_analytic(nu) )
    
    # Test the different methods
    for method in available_methods :

        # Initialize transform
        dht = DHT( p, N, rmax, method )

        # Calculate f on the natural grid
        f = f_analytic( dht.get_r() )
        # For comparison, calculate g on the natural grid
        g = g_analytic( dht.get_nu() )

        # Apply the forward and backward transform
        t1 = time.time()
        g_dht = dht.transform( f )        
        f_dht = dht.inverse_transform( g_dht )
        t2 = time.time()
        
        # Plot the results
        # - Hankel transform
        plt.subplot(121)
        plt.plot( dht.r, f_dht.real, 'o', label=method )
        plt.subplot(122)
        plt.plot( dht.nu, g_dht.real, 'o', label=method )
        
        # Calculate the RMS error
        error_f = np.sqrt( ((f_dht.real-f)**2).mean() )
        error_g = np.sqrt( ((g_dht.real-g)**2).mean() )
        Dt_ms = (t2-t1)/2*1.e3
        
        # Diagnostic
        print('')
        print(method)
        print('----')
        print('  - Time per transform : %.3f ms' % Dt_ms )
        print('  - RMS error on the Hankel transform : %f' % error_g )
        print("""  - RMS error on the orginal function
        after back and forth transform : %f""" % error_f)

    plt.show()
        
def compare_flattop( N, rmax ) :
    """
    Test with the transform pair flattop / Airy.
    """
    def flattop(x) :
        return( np.where( x<1, 1., 0. ) )

    def airy(x) :
        return( jn(1,2*np.pi*x)/x )  # Check formula
    
    compare_Hankel_methods( flattop, airy, 0, N, rmax )
        
def compare_airy( N, rmax ) :
    """
    Test with the transform pair flattop / Airy.
    """
    def flattop(x) :
        return( np.where( x<1, 1., 0. ) )

    def airy(x) :
        return( jn(1,2*np.pi*x)/x )
    
    compare_Hankel_methods( airy , flattop, 0, N, rmax )
    
def compare_laguerre_gauss( p, N, rmax ) :
    """
    Test with the Laguerre-Gauss functions, which are invariant under the
    Hankel transform.
    """
    pass
    

def test_airy() :
    " Tests the Airy function "
    compare_airy( 100, 5.223)

def test_flattop() :
    " Tests the flattop function "
    compare_flattop( 100, 5.223)

if __name__ == '__main__' :
    compare_airy( 100, 5.223)
    compare_flattop( 100, 5.223)
