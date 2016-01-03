"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It tests the structures implemented in fields.py,
by studying the propagation of a laser in vacuum:
- The mode 0 is tested by using an annular, radial beam,
   which propagates to the right.
- The mode 1 is tested by using a linearly polarized beam,
   which propagates to the left.

In both cases, the evolution of the a0 and w0 are compared
with theoretical results from diffraction theory.

NB: The box is periodic. The timesteps are extremely large, compared
to their typical value in a PIC simulation. This uses the fact that
the PSATD scheme has no Courant limit, and therefore this test
propagates the laser over long distances using only a few timesteps.

Usage :
-------
In order to show the images of the laser, and manually check the
agreement between the simulation and the theory:
$ python tests/test_fields.py
(except when setting show to False in the parameters below)

In order to let Python check the agreement between the curve without
having to look at the plots
$ py.test -q tests/test_fields.py
or 
$ python setup.py test
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, m_e, e
from scipy.optimize import curve_fit
from fbpic.fields import Fields
from fbpic.lpa_utils import add_laser

# Parameters
# ----------
# (See the documentation of the function propagate_pulse 
# below for their definition)

# Simulation box
Nz = 300
zmin = -15.e-6
zmax = 15.e-6
Nr = 100
Lr = 40.e-6
Nm = 2
n_order = -1
# Laser pulse
w0 = 4.e-6
ctau = 5.e-6
k0 = 2*np.pi/0.8e-6
E0 = 1.
# Propagation
L_prop = 120.e-6
zf = -120.e-6
N_step = 10
# Checking the results
show = True  # Whether to show the plots, and check them manually
N_show = 1
rtol = 1.e-3

def test_laser(show=False):
    "Function that is run by py.test, when doing `python setup.py test`"
    print('')
    print('Testing mode m=0 with an annular beam')
    plt.figure()
    propagate_pulse( Nz, Nr, Nm, zmin, zmax, Lr, L_prop, zf,
                      N_step, w0, ctau, k0, E0, 0, N_show, n_order,
                      rtol, show=show )

    print('')
    print('Testing mode m=1 with an gaussian beam')
    plt.figure()
    propagate_pulse(Nz, Nr, Nm, zmin, zmax, Lr, L_prop, zf,
                     N_step, w0, ctau, k0, E0, 1, N_show,
                     n_order, rtol, show=show )
    
    print('')

def propagate_pulse( Nz, Nr, Nm, zmin, zmax, Lr, L_prop, zf, Nt, w0, ctau,
                k0, E0, m, N_show, n_order, rtol, show=False ) :
    """
    Propagate the beam over a distance L_prop in Nt steps,
    and extracts the waist and a0 at each step.

    Parameters
    ----------
    show : bool
       Wether to show the fields, so that the user can manually check
       the agreement with the theory.
       If True, this will periodically show the map of the fields (with
       a period N_show), as well as (eventually) the evoluation of a0 and w0.
       If False, this 

    N_show : int
       Number of timesteps between two consecutive plots of the fields
    
    Nz, Nr : int
       The number of points on the grid in z and r respectively

    Nm : int
        The number of modes in the azimuthal direction

    zmin, zmax : float
        The limits of the box in z
           
    Lr : float
       The size of the box in the r direction
       (In the case of Lr, this is the distance from the *axis*
       to the outer boundary)

    L_prop : float
       The total propagation distance (in meters)

    zf : float
       The position of the focal plane of the laser (only works for m=1)

    Nt : int
       The number of timesteps to take, to reach that distance

    w0 : float
       The initial waist of the laser (in meters)

    ctau : float
       The initial temporal waist of the laser (in meters)

    k0 : flat
       The central wavevector of the laser (in meters^-1)

    E0 : float
       The initial E0 of the pulse

    m : int
       Index of the mode to be tested
       For m = 1 : test with a gaussian, linearly polarized beam
       For m = 0 : test with an annular beam, polarized in E_theta

    n_order : int
       Order of the stencil

    rtol : float
       Relative 
                  
    Returns
    -------
    A dictionary containing :
    - 'E' : 1d array containing the values of the amplitude
    - 'w' : 1d array containing the values of waist
    - 'fld' : the Fields object at the end of the simulation.
    """

    # Initialize the fields object
    dt = L_prop/c * 1./Nt
    fld = Fields( Nz, zmax, Nr, Lr, Nm, dt, n_order=n_order, zmin=zmin )
    z0 = (zmax+zmin)/2
    init_fields( fld, w0, ctau, k0, z0, zf, E0, m )

    # Create the arrays to get the waist and amplitude
    w = np.zeros(Nt)
    E = np.zeros(Nt)
        
    # Get the fields in spectral space
    fld.interp2spect('E')
    fld.interp2spect('B')

    # Loop over the iterations
    print('Running the simulation...')
    for it in range(Nt) :
        # Advance the Maxwell equations
        fld.push()
        # Bring the fields back onto the interpolation grid
        fld.spect2interp('E')
        fld.spect2interp('B')
        # Fit the fields to find the waist and a0
        w[it], E[it] = fit_fields( fld, m )
        # Since the fit returns the RMS of E, renormalize it
        E[it] = E[it]*2**(3./4)/np.pi**(1./4)*np.sqrt((zmax-zmin)/ctau)
        # Plot the fields during the simulation
        if show==True and it%N_show == 0 :
            plt.clf()
            fld.interp[m].show('Et')
            plt.show()
        # Bring the fields back again onto the spectral grid
        # (This is not needed in principle, as the fields
        # were not modified in the real space, but it allows
        # additional checking on the reversibility of the transform)
        fld.interp2spect('E')
        fld.interp2spect('B')
                            
    # Get the analytical solution
    z_prop = c*dt*np.arange(1, Nt+1) 
    ZR = 0.5*k0*w0**2
    if m == 0 : # zf is not implemented for m = 0
        w_analytic = w0*np.sqrt( 1 + z_prop**2/ZR**2 )
        E_analytic = E0/( 1 + z_prop**2/ZR**2 )**(1./2)
    else : # + zf because the pulse is backward propagating
        w_analytic = w0*np.sqrt( 1 + (z_prop+zf)**2/ZR**2 )
        E_analytic = E0/( 1 + (z_prop+zf)**2/ZR**2 )**(1./2)
        
    # Either plot the results and check them manually
    if show is True:
        plt.suptitle('Diffraction of a pulse in the mode %d' %m)
        plt.subplot(121)
        plt.plot( 1.e6*z_prop, 1.e6*w, 'o', label='Simulation' )
        plt.plot( 1.e6*z_prop, 1.e6*w_analytic, '--', label='Theory' )
        plt.xlabel('z (microns)')
        plt.ylabel('w (microns)')
        plt.title('Waist')
        plt.legend(loc=0)
        plt.subplot(122)
        plt.plot( 1.e6*z_prop, E, 'o', label='Simulation' )
        plt.plot( 1.e6*z_prop, E_analytic, '--', label='Theory' )
        plt.xlabel('z (microns)')
        plt.ylabel('E')
        plt.legend(loc=0)
        plt.title('Amplitude')
        plt.show()
    # or automatically check that the theoretical and simulated curves
    # of a0 and E are close 
    else:
        assert np.allclose( w, w_analytic, rtol=rtol )
        print('The simulation results agree with the theory to %e.' %rtol)
        
    # Return a dictionary of the results
    return( { 'E' : E, 'w' : w, 'fld' : fld } )

    
def init_fields( fld, w, ctau, k0, z0, zf, E0, m=1 ) :
    """
    Imprints the appropriate profile on the fields of the simulation.

    Parameters
    ----------
    fld: Fields object from fbpic

    w : float
       The initial waist of the laser (in microns)

    ctau : float
       The initial temporal waist of the laser (in microns)

    k0 : flat
       The central wavevector of the laser (in microns^-1)

    z0 : float
       The position of the centroid on the z axis

    zf : float 
       The position of the focal plane

    E0 : float
       The initial E0 of the pulse

    m: int, optional
        The mode on which to imprint the profile
        For m = 1 : gaussian profile, linearly polarized beam
        For m = 0 : annular profile, polarized in E_theta
    """
    
    # Initialize the fields with the right value and phase
    if m == 1 :
        add_laser( fld, E0*e/(m_e*c**2*k0), w, ctau, z0, zf=zf, 
                   lambda0 = 2*np.pi/k0, fw_propagating=False )
    if m == 0 :
        z = fld.interp[m].z
        r = fld.interp[m].r
        profile = annular_pulse( z, r, w, ctau, k0, z0, E0 ) 
        fld.interp[m].Et[:,:] = profile
        fld.interp[m].Br[:,:] = -1./c*profile

        
def gaussian_transverse_profile( r, w, E ) :
    """
    Calculte the Gaussian transverse profile.

    This is used for the fit of the fields
    
    Parameters
    ----------
    r: 1darray
       Represents the positions of the grid in r

    w : float
       The initial waist of the laser (in microns)

    E : float
       The a0 of the pulse
    """
    return( E*np.exp( -r**2/w**2 ) )

def annular_transverse_profile( r, w, E ) :
    """
    Calculte the annular transverse profile.

    This is used both for the initialization and
    for the fit of the fields
    
    Parameters
    ----------
    r: 1darray
       Represents the positions of the grid in r

    w : float
       The initial waist of the laser (in microns)

    E : float
       The E0 of the pulse
    """
    return( E*(r/w)*np.exp( -r**2/w**2 ) )
    
def annular_pulse( z, r, w0, ctau, k0, z0, E0 ) :
    """
    Calculate the profile of an annular beam.
    This is used to initialize the beam
    
    Parameters
    ----------
    z: 1darray
       Represents the positions of the grid in z
       
    r: 1darray
       Represents the positions of the grid in r

    w0 : float
       The initial waist of the laser (in microns)

    ctau : float
       The initial temporal waist of the laser (in microns)

    k0 : flat
       The central wavevector of the laser (in microns^-1)

    z0 : float
       The position of the centroid on the z axis

    E0 : float
       The initial E0 of the pulse
       
    Return
    ------
       A 2d array with z as the first axis and r as the second axis,
       which contains the values of the 
    
    """
    longitudinal = np.exp( -(z-z0)**2/ctau**2 )*np.cos(k0*(z-z0))
    transverse = annular_transverse_profile( r, w0, E0 )
    profile = longitudinal[:,np.newaxis]*transverse[np.newaxis,:]
    
    return(profile)
    
    
def fit_fields( fld, m ) :
    """
    Extracts the waist and a0 of the pulse through a transverse Gaussian fit.

    The laser oscillations are first averaged longitudinally.

    Parameters
    ----------
    fld : Fields object from fbpic

    m : int, optional
       The index of the mode to be fitted
    """
    # Average the laser oscillations longitudinally
    laser_profile = np.sqrt( (abs( fld.interp[m].Et )**2).mean(axis=0) )

    # Do the fit
    r = fld.interp[m].r
    if m==1 :  # Gaussian profile
        fit_result = curve_fit(gaussian_transverse_profile, r,
                            laser_profile, p0=np.array([w0,E0]) )
        # Factor 2 on the amplitude, related to the factor 2
        # in the particle gather for the modes m > 0
        fit_result[0][1] = 2*fit_result[0][1]
    elif m==0 : # Annular profile
        fit_result = curve_fit(annular_transverse_profile, r,
                            laser_profile, p0=np.array([w0,E0]) )
    
    return( fit_result[0] )
    
if __name__ == '__main__' :
    
    # Run the testing function
    test_laser(show=True)

