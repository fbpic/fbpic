"""
This file tests the structures implemented in fields.py,
by studying the propagation of a Gaussian beam in vacuum.

Usage :
from the top-level directory of FBPIC run
$ python tests/test_fields.py
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from scipy.optimize import curve_fit
from fbpic.fields import Fields

def test_pulse( Nz, Nr, Nm, Lz, Lr, L_prop, Nt, w0, ctau, k0, a0, m ) :
    """
    Propagate the beam over a distance L_prop in N_step,
    and extracts the waist and a0 at each step

    Parameters
    ----------
    Nz, Nr : int
       The number of points on the grid in z and r respectively

    Nm : int
        The number of modes in the azimuthal direction
       
    Lz, Lr : float
       The size of the box in z and r respectively (in microns)
       (In the case of Lr, this is the distance from the *axis*
       to the outer boundary)

    L_prop : float
       The total propagation distance (in microns)

    Nt : int
       The number of timesteps to take, to reach that distance

    w0 : float
       The initial waist of the laser (in microns)

    ctau : float
       The initial temporal waist of the laser (in microns)

    k0 : flat
       The central wavevector of the laser (in microns^-1)

    a0 : float
       The initial a0 of the pulse

    m : int
       Index of the mode to be tested
       For m = 1 : test with a gaussian, linearly polarized beam
       For m = 0 : test with an annular beam, polarized in E_theta
           
    Returns
    -------
    A dictionary containing :
    - 'a' : 1d array containing the values of a0
    - 'w' : 1d array containing the values of w0
    - 'fld' : the Fields object at the end of the simulation.
    """

    # Initialize the fields object
    dt = L_prop/c * 1./Nt
    fld = Fields( Nz, Lz, Nr, Lr, Nm, dt )
    z0 = Lz/2
    init_fields( fld, w0, ctau, k0, z0, a0, m )

    # Create the arrays to get the waist and a0
    w = np.zeros(Nt)
    a = np.zeros(Nt)
        
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
        w[it], a[it] = fit_fields( fld, m )
        # Since the fit returns the RMS of a, renormalize it
        a[it] = a[it]*2**(3./4)/np.pi**(1./4)*np.sqrt(Lz/ctau)
        # Show the progression bar
        progression_bar(it, Nt-1)
        # Plot the fields during the simulation
      #  plt.clf()
      #  fld.interp[m].show('Et')
      #  plt.show()

    # Get the analytical solution
    z_prop = c*dt*np.arange(1, Nt+1)
    ZR = 0.5*k0*w0**2
    w_analytic = w0*np.sqrt( 1 + z_prop**2/ZR**2 )
    a_analytic = a0/( 1 + z_prop**2/ZR**2 )**(1./2)
        
    # Plot the results
    plt.subplot(121)
    plt.plot( z_prop, w, 'o' )
    plt.plot( z_prop, w_analytic, '--' )
    plt.xlabel('z (microns)')
    plt.ylabel('w (microns)')
    plt.title('Waist')
    plt.subplot(122)
    plt.plot( z_prop, a, 'o' )
    plt.plot( z_prop, a_analytic, '--' )
    plt.xlabel('z (microns)')
    plt.ylabel('a')
    plt.title('Amplitude')
    
    # Return a dictionary of the results
    return( { 'a' : a, 'w' : w, 'fld' : fld } )

    
def init_fields( fld, w, ctau, k0, z0, a0, m=1 ) :
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

    a0 : float
       The initial a0 of the pulse

    m: int, optional
        The mode on which to imprint the profile
        For m = 1 : gaussian profile, linearly polarized beam
        For m = 0 : annular profile, polarized in E_theta
    """
    # Extract the coordinates of the grid
    z = fld.interp[m].z
    r = fld.interp[m].r

    # Initialize the fields with the right value and phase
    if m == 1 :
        profile = gaussian_pulse( z, r, w, ctau, k0, z0, a0 ) 
        fld.interp[m].Er[:,:] = profile
        fld.interp[m].Et[:,:] = -1.j*profile
        fld.interp[m].Br[:,:] = 1.j*1./c*profile
        fld.interp[m].Bt[:,:] = 1./c*profile
    if m == 0 :
        profile = annular_pulse( z, r, w, ctau, k0, z0, a0 ) 
        fld.interp[m].Et[:,:] = profile
        fld.interp[m].Br[:,:] = -1./c*profile

        
def gaussian_transverse_profile( r, w, a ) :
    """
    Calculte the Gaussian transverse profile.

    This is used both for the initialization and
    for the fit of the fields
    
    Parameters
    ----------
    r: 1darray
       Represents the positions of the grid in r

    w : float
       The initial waist of the laser (in microns)

    a : float
       The a0 of the pulse
    """
    return( a*np.exp( -r**2/w**2 ) )
    
def gaussian_pulse( z, r, w0, ctau, k0, z0, a0 ) :
    """
    Calculate the profile of a Gaussian beam.
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

    a0 : float
       The initial a0 of the pulse
       
    Return
    ------
       A 2d array with z as the first axis and r as the second axis,
       which contains the values of the 
    
    """
    longitudinal = np.exp( -(z-z0)**2/ctau**2 )*np.cos(k0*(z-z0))
    transverse = gaussian_transverse_profile( r, w0, a0 )
    profile = longitudinal[:,np.newaxis]*transverse[np.newaxis,:]
    
    return(profile)

def annular_transverse_profile( r, w, a ) :
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

    a : float
       The a0 of the pulse
    """
    return( a*(r/w)*np.exp( -r**2/w**2 ) )
    
def annular_pulse( z, r, w0, ctau, k0, z0, a0 ) :
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

    a0 : float
       The initial a0 of the pulse
       
    Return
    ------
       A 2d array with z as the first axis and r as the second axis,
       which contains the values of the 
    
    """
    longitudinal = np.exp( -(z-z0)**2/ctau**2 )*np.cos(k0*(z-z0))
    transverse = annular_transverse_profile( r, w0, a0 )
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
                            laser_profile, p0=np.array([1,1]) )
    elif m==0 : # Annular profile
        fit_result = curve_fit(annular_transverse_profile, r,
                            laser_profile, p0=np.array([1,1]) )
    
    return( fit_result[0] )

def progression_bar(i, Ntot, Nbars=60, char='-') :
    "Shows a progression bar with Nbars"
    nbars = int( i*1./Ntot*Nbars )
    sys.stdout.write('\r[' + nbars*char )
    sys.stdout.write((Nbars-nbars)*' ' + ']')
    sys.stdout.flush()
    
if __name__ == '__main__' :
    
    # Simulation box
    Nz = 300
    Lz = 30.
    Nr = 300
    Lr = 40.
    Nm = 2
    # Laser pulse
    w0 = 2.
    ctau = 10.
    k0 = 2*np.pi/0.8
    a0 = 1.
    # Propagation
    L_prop = 200.
    N_step = 20

    print('')
    print('Testing mode m=0 with an annular beam')
    plt.figure()
    res = test_pulse( Nz, Nr, Nm, Lz, Lr, L_prop, N_step, w0, ctau, k0, a0, 0 )
    plt.show()
    res['fld'].interp[0].show('Et')
    plt.show()
    
    print('')
    print('Testing mode m=1 with an gaussian beam')
    plt.figure()
    res = test_pulse(Nz, Nr, Nm, Lz, Lr, L_prop, N_step, w0, ctau, k0, a0, 1 )
    plt.show()
    res['fld'].interp[1].show('Et')
    plt.show()

    print('')
