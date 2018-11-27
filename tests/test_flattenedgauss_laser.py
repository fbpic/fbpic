# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It tests the flattened Gaussian laser profile by initializing the laser
before the focal plane, and letting it propagate for many Rayleigh lengths.
This test then automatically checks that the transverse profile at this
point corresponds to a flattened Gaussian.

Usage :
-------
In order to show the transverse profile out of focus:
$ python tests/test_flattenedgauss_laser.py
(except when setting show to False in the parameters below)

In order to let Python check the agreement between the curve without
having to look at the plots
$ py.test -q tests/test_flattenedgauss_laser.py
or
$ python setup.py test
"""
import numpy as np
from scipy.special import factorial
from scipy.constants import c
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser_pulse, FlattenedGaussianLaser

# Parameters
# ----------

show = True  # Whether to show the plots, and check them manually

use_cuda = True

# Simulation box
Nz = 1600
zmin = -40.e-6
zmax = 40.e-6
Nr = 600
rmax = 300.e-6
Nm = 2
n_order = -1
# Laser pulse
w0 = 4.e-6
N = 6
ctau = 10.e-6
k0 = 2*np.pi/0.8e-6
a0 = 1.
# Propagation
zfoc = 400.e-6
Lprop = 2800.e-6 # Corresponds to many Rayleigh lengths

# Checking the results
rtol = 1.5e-2

def flat_gauss(x, N):
    """
    Function that calculates the theoretical profile out of focus
    """
    u = np.zeros_like(x)
    for n in range(N+1):
        u += 1./factorial(n) * ((N+1)*x**2)**n
    u *= np.exp(-(N+1)*x**2)
    return u


def test_laser_periodic(show=False):
    """
    Function that is run by py.test, when doing `python setup.py test`
    Test the propagation of a laser in a periodic box.
    """
    # Propagate the pulse in a single step
    dt = Lprop*1./c

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
                n_order=n_order, zmin=zmin, boundaries='periodic' )

    # Initialize the laser fields
    profile = FlattenedGaussianLaser(a0=a0, w0=w0, N=N, 
                                     tau=ctau/c, z0=0, zf=zfoc)
    add_laser_pulse( sim, profile )

    # Propagate the pulse
    sim.step(1)

    # Check the validity of the transverse field profile
    # (Take the RMS field in order to suppress the laser oscillations)
    trans_profile = np.sqrt( np.average(sim.fld.interp[1].Er.real**2, axis=0) )
    # Calculate the theortical profile out-of-focus
    Zr = k0*w0**2/2
    w_th = w0*(Lprop-zfoc)/Zr
    r = sim.fld.interp[1].r
    th_profile = trans_profile[0]*flat_gauss( r/w_th, N )
    # Plot the profile, if requested by the user
    if show:
        import matplotlib.pyplot as plt
        plt.plot( 1.e6*r, trans_profile, label='Simulated' )
        plt.plot( 1.e6*r, th_profile, label='Theoretical' )
        plt.legend(loc=0)
        plt.xlabel('r (microns)')
        plt.title('Transverse profile out-of-focus')
        plt.tight_layout()
        plt.show()
    # Check the validity
    assert np.allclose( th_profile, trans_profile, atol=rtol*th_profile[0] )


if __name__ == '__main__' :

    # Run the testing function
    test_laser_periodic(show=show)
