"""
This file tests the Discrete Hankel Transform which is implemented in hankel_dt
"""

import numpy as np
import matplotlib.pyplot as plt
from fbpic.hankel_dt import DHT, available_methods
from scipy.special import jn, jn_zeros
from scipy.special import eval_genlaguerre
import time

def compare_Hankel_methods( f_analytic, g_analytic, p, Nz, Nr,
                            axis, rmax, npts=1000, **kw ) :
    """
    Compare the discrete Hankel transform of f_analytic to the
    analytical result g_analytic.

    This function allows to see the effect of the Hankel transform on a
    *2d array*, where the Hankel transform is taken only along the axis
    `axis`, with length Nr. Nothing happens along the other axis, which
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
    for method in available_methods :

        # Initialize transform
        dht = DHT( p, Nr, rmax, method, **kw )

        # Calculate f and g on the natural grid
        if axis != 0 :
            f = np.empty((Nz,Nr))
            f[:,:] = f_analytic( dht.get_r() )[np.newaxis,:]
            g = np.empty((Nz,Nr))
            g = g_analytic( dht.get_nu() )[np.newaxis,:]
        else :
            f = np.empty((Nr,Nz))
            f[:,:] = f_analytic( dht.get_r() )[:,np.newaxis]
            g = np.empty((Nr,Nz))
            g = g_analytic( dht.get_nu() )[:,np.newaxis]
            
        # Apply the forward and backward transform
        t1 = time.time()
        g_dht = dht.transform( f, axis=axis )        
        f_dht = dht.inverse_transform( g_dht, axis=axis )
        t2 = time.time()
        
        # Plot the results
        for iz in range(Nz) :
            if axis !=0 :
                plt.subplot(121)
                plt.plot( dht.r, f_dht[iz], 'o', label=method )
                plt.subplot(122)
                plt.plot( dht.nu, g_dht[iz], 'o', label=method )
            else :
                plt.subplot(121)
                plt.plot( dht.r, f_dht[:,iz], 'o', label=method )
                plt.subplot(122)
                plt.plot( dht.nu, g_dht[:,iz], 'o', label=method )
        
        # Calculate the maximum error
        error_f = np.max( abs(f_dht-f) )
        error_g = np.max( abs(g_dht-g) )
        Dt_ms = (t2-t1)/(2*Nz)*1.e3

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
        
def compare_power_p( p, rcut, N, rmax, **kw ) :
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
    
    compare_Hankel_methods( power_p, power_p_trans, p, 1, N, -1, rmax, **kw )

def compare_laguerre_gauss( p, n, N, rmax ) :
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
                            1, N, -1, rmax, **kw )

def compare_bessel( p, m, n, N, rmax, **kw ) :
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
    
    compare_Hankel_methods( bessel_n_p, delta, p, 1, N, -1, rmax, **kw )
    
if __name__ == '__main__' :

    N = 200
    pmax = 1
    rmax = 4
    kw = { 'd' : 0.5, 'Fw' : 'inverse' }
    
    for p in range(pmax+1) :
        compare_power_p( p, 1, N, rmax, **kw )

    for p in range(pmax+1) :
        for n in range(2) :
            compare_laguerre_gauss( p, n, N, rmax )

    for p in range(pmax+1) :
        compare_bessel( p, p, int(N*0.3), N, rmax, **kw )
        compare_bessel( p, p, int(N*0.9), N, rmax, **kw )

    for p in range(pmax+1) :
        compare_bessel( p, p+1, int(N*0.3), N, rmax, **kw )
        compare_bessel( p, p+1, int(N*0.9), N, rmax, **kw )
