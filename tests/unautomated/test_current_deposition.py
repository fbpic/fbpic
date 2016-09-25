# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file tests the current deposition by initializing the electron
momenta as if they where in a laser field, and projecting the currents.

The resulting currents can be calculated analytically.

Usage :
from the top-level directory of FBPIC run
$ python tests/test_current_deposition.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e
from fbpic.main import Simulation

def a(z, x, y, w0, a0, k0, ctau, z0 ) :
    """
    Returns the amplitude of the normalized vector potential
    in a Gaussian pulse, at the position of the macroparticles, or on a
    grid

    NB : the pulse envelope is centered on z = 0
    
    Parameters
    ----------
    z, x, y : 1darrays of floats (one element per macroparticles)
              or 2darrays of floats ( positions on a grid )

    a0, w0, ctau, k0, z0 : floats
        Amplitude (dimensionless), waist (in meters), length (in meters),
        wavevector (in meters^-1) and centroid position (meters) of the
        laser pulse

    Returns
    -------
    A 1d array of floats, containing the amplitude of the normalized
    vector potential
    """
    a_array = a0*np.exp( - (x**2+y**2)/w0**2 - (z-z0)**2/ctau**2 )*np.cos(k0*(z-z0))
    return(a_array)

def Jr(z, r, w0, a0, k0, ctau, z0, n, q, pol ) :
    """
    Returns the analytical expression of the current along r in the mode 1
    on a given grid.
    
    Parameters
    ----------
    z, r : 1darrays of floats (meters)
        Contain the position of the gridpoints in the longitudinal
        and transverse direction

    a0, w0, ctau, k0, z0 : floats
        Amplitude (dimensionless), waist (in meters), length (in meters),
        wavevector (in meters^-1) and centroid position (meters) of the
        laser pulse

    n, q : floats
        Density (meters^-3), charge (Coulomb) and mass (kg) of the particles
    
    pol : string
        Polarization of the laser. Either 'x' or 'y'    
    """
    # Build 2d arrays from the 1d arrays
    r, z = np.meshgrid( r, z, copy=True )
    
    # Get the amlitude of the vector potential,
    # in the azimuthal Fourier decomposition
    a_on_grid = a( z, r, 0, w0, a0, k0, ctau, z0 )
    # Get the right complex phase ; the factor 0.5
    # comes from the normalization of the modes in FBPIC
    if pol == 'x' :
        ar_on_circ_grid = 0.5*a_on_grid*(1.+0.j)
    elif pol == 'y' :
        ar_on_circ_grid = 0.5*a_on_grid*(0.+1.j)
    else :
        raise ValueError('Illegal polarization : %s' %pol)

    # Velocity of the particles
    Vr_on_circ_grid = c * ar_on_circ_grid / np.sqrt(1+a_on_grid**2)
    Jr_on_circ_grid = n * q * Vr_on_circ_grid

    return( Jr_on_circ_grid )

def impart_momenta( ptcl, a0, w0, ctau, k0, z0, pol ) :
    """
    Impart momenta to the particles, as if they where in a laser pulse,
    (by assuming momentum conservation : u = a)

    Parameters
    ----------
    ptcl : a Particles object

    a0, w0, ctau, k0, z0 : floats
        Amplitude (dimensionless), waist (in meters), length (in meters),
        wavevector (in meters^-1) and centroid position (meters) of the
        laser pulse

    pol : string
        Polarization of the laser. Either 'x' or 'y'
    """
    # Impart the transverse momenta
    if pol == 'x' :
        ptcl.ux = a( ptcl.z, ptcl.x, ptcl.y, w0, a0, k0, ctau, z0  )
    elif pol == 'y' :
        ptcl.uy = a( ptcl.z, ptcl.x, ptcl.y, w0, a0, k0, ctau, z0  )
    else :
        raise ValueError('Illegal polarization : %s' %pol)

    # Calculate the corresponding Lorentz factor
    ptcl.inv_gamma = 1./np.sqrt( 1 + ptcl.ux**2 + ptcl.uy**2 + ptcl.uz**2 )

    
def deposit_current( ptcl, fld ) :
    """
    Deposit the current from the particles `ptcl` onto the fields `fld`

    Parameters
    ----------
    ptcl : a Particles object
    fld : a Fields object
    """
    fld.erase('J')
    ptcl.deposit( fld, 'J')
    fld.divide_by_volume('J') 

def compare( Jr_analytic, Jr_simulation ) :
    """
    Draws a series of plots to compare the analytical and theoretical results
    """
    plt.figure(figsize=(15,15))

    plt.subplot(321)
    plt.imshow( Jr_analytic.T[::-1].real, aspect='auto', interpolation='nearest' )
    plt.colorbar()
    plt.title('Analytical Jr (real part)')
    
    plt.subplot(322)
    plt.imshow( Jr_analytic.T[::-1].imag, aspect='auto', interpolation='nearest' )
    plt.colorbar()
    plt.title('Analytical Jr (imaginary part)')

    plt.subplot(323)
    plt.imshow( Jr_simulation.T[::-1].real, aspect='auto', interpolation='nearest' )
    plt.colorbar()
    plt.title('Deposited Jr (real part)')
    
    plt.subplot(324)
    plt.imshow( Jr_simulation.T[::-1].imag, aspect='auto', interpolation='nearest' )
    plt.colorbar()
    plt.title('Deposited Jr (imaginary part)')

    plt.subplot(325)
    plt.plot( Jr_analytic[:,0].real, label='Analytical' )
    plt.plot( Jr_simulation[:,0].real, label='Deposited' )
    plt.title('On-axis Jr (real part)')
    plt.legend(loc=0)

    plt.subplot(326)
    plt.plot( Jr_analytic[:,0].imag, label='Analytical' )
    plt.plot( Jr_simulation[:,0].imag, label='Deposited' )
    plt.title('On-axis Jr (imaginary part)')
    plt.legend(loc=0)
    
    plt.show()
    
    
    
if __name__ == '__main__' :

    # Dimensions of the box on which the laser field will
    # be interpolated.
    Nz = 250
    zmax = 20.e-6
    Nr = 150
    rmax= 20.e-6
    Nm = 2
    # Laser pulse
    w0 = 5.e-6
    ctau = 10.e-6
    k0 = 2*np.pi/0.8e-6
    a0 = 1.
    z0 = 10.e-6
    pol = 'x'
    # Particles
    p_nr = 2
    p_nz = 2
    p_nt = 2
    n = 9.e24
    q = -e
    
    # Initialize the different structures
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, zmax/Nz/c,
        0, zmax, 0, rmax, p_nz, p_nr, p_nt, n )

    # Impart the momenta
    impart_momenta( sim.ptcl[0], a0, w0, ctau, k0, z0, pol )

    # Deposit the currents
    deposit_current( sim.ptcl[0], sim.fld )

    # Get the analyical result
    Jr_analytical = Jr( sim.fld.interp[0].z, sim.fld.interp[0].r,
                        w0, a0, k0, ctau, z0, n, q, pol )

    # Compare the results
    compare( Jr_analytical, sim.fld.interp[1].Jr )
