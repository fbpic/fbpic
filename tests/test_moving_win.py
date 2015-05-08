"""
This file tests the effect of the moving window on the fields, by
letting a laser propagate (with the moving window on and no macroparticles).

Usage :
from the top-level directory of FBPIC run
$ python tests/test_moving_win.py
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, m_e, e
from scipy.optimize import curve_fit
from fbpic.fields import Fields
from fbpic.lpa_utils import add_laser
from fbpic.moving_window import MovingWindow

def test_pulse( Nz, Nr, Nm, Lz, Lr, Nt, w0, ctau,
                k0, E0, m, N_show, show=False ) :
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
       
    Nt : int
       The number of timesteps to take

    w0 : float
       The initial waist of the laser (in microns)

    ctau : float
       The initial temporal waist of the laser (in microns)

    k0 : flat
       The central wavevector of the laser (in microns^-1)

    E0 : float
       The initial E0 of the pulse

    m : int
       Index of the mode to be tested
       For m = 1 : test with a gaussian, linearly polarized beam
       For m = 0 : test with an annular beam, polarized in E_theta
       
    show : bool
       Wether to show the fields

    N_show : int
       Number of timesteps between two consecutive plots of the fields
           
    Returns
    -------
    A dictionary containing :
    - 'E' : 1d array containing the values of the amplitude
    - 'w' : 1d array containing the values of waist
    - 'fld' : the Fields object at the end of the simulation.
    - 'z_centroid' : 1d array containing the values of the centroid position
    """

    # Choose the timestep, so that the simulation advances by
    # one cell at every timestep
    dt = Lz*1./Nz * 1./c
    
    # Initialize the fields object
    fld = Fields( Nz, Lz, Nr, Lr, Nm, dt )
    z0 = Lz/2
    init_fields( fld, w0, ctau, k0, z0, E0, m )

    # Create the arrays to get the waist and amplitude
    w = np.zeros(Nt)
    E = np.zeros(Nt)
    z_center = np.zeros(Nt)
        
    # Get the fields in spectral space
    fld.interp2spect('E')
    fld.interp2spect('B')
    
    #Create moving window object
    mov_win = MovingWindow()
    
    # Loop over the iterations
    print('Running the simulation...')
    for it in range(Nt) :
        # Shift the fields using the moving window
        mov_win.shift_fields(fld)
        # Advance the Maxwell equations
        fld.push()
        # Bring the fields back onto the interpolation grid
        fld.spect2interp('E')
        fld.spect2interp('B')
        # Fit the fields to find the waist and a0
        w[it], E[it] = fit_fields( fld, m )
        # Since the fit returns the RMS of E, renormalize it
        E[it] = E[it]*2**(3./4)/np.pi**(1./4)*np.sqrt(Lz/ctau)
        # Get the average position of the laser
        z_center[it] = average_position( fld, m )
        # Show the progression bar
        progression_bar(it, Nt-1)
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
    w_analytic = w0*np.sqrt( 1 + z_prop**2/ZR**2 )
    E_analytic = E0/( 1 + z_prop**2/ZR**2 )**(1./2)
        
    # Plot the results
    plt.suptitle('Diffraction of a pulse in the mode %d' %m)
    plt.subplot(131)
    plt.plot( z_prop, w, 'o', label='Simulation' )
    plt.plot( z_prop, w_analytic, '--', label='Theory' )
    plt.xlabel('z (microns)')
    plt.ylabel('w (microns)')
    plt.title('Waist')
    plt.legend(loc=0)
    plt.subplot(132)
    plt.plot( z_prop, E, 'o', label='Simulation' )
    plt.plot( z_prop, E_analytic, '--', label='Theory' )
    plt.xlabel('z (microns)')
    plt.ylabel('E')
    plt.legend(loc=0)
    plt.title('Amplitude')
    plt.subplot(133)
    plt.plot( z_prop, z_center, 'o', label='Simulation' )
    plt.xlabel('z (microns)')
    plt.ylabel('z_center')
    plt.legend(loc=0)
    plt.title('Centroid position')
    
    # Return a dictionary of the results
    return( { 'E' : E, 'w' : w, 'fld' : fld, 'z_center' : z_center } )

    
def init_fields( fld, w, ctau, k0, z0, E0, m=1 ) :
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

    E0 : float
       The initial E0 of the pulse

    m: int, optional
        The mode on which to imprint the profile
        For m = 1 : gaussian profile, linearly polarized beam
        For m = 0 : annular profile, polarized in E_theta
    """
    
    # Initialize the fields with the right value and phase
    if m == 1 :
        add_laser( fld, E0*e/(m_e*c**2*k0), w, ctau, z0,
                   lambda0 = 2*np.pi/k0, fw_propagating=True )
    if m == 0 :
        z = fld.interp[m].z
        r = fld.interp[m].r
        profile = annular_pulse( z, r, w, ctau, k0, z0, E0 ) 
        fld.interp[m].Et[:,:] = profile
        fld.interp[m].Br[:,:] = -1./c*profile

def average_position( fld, m ) :
    """
    Calculates the position of the centroid of the laser.
    
    If the moving window does not work, this wraps around between 0 and Lz.

    Parameters
    ----------
    fld : a Fields object
        Contains the field data of the simulation

    m : int
        The index of the azimuthal mode considered

    Returns
    -------
    A float indicating the position of the centroid
    """
    # Average the positions with the amplitude of the laser *off-axis*
    # (since the amplitude may be 0 on axis for m=0)
    z_center = np.average( fld.interp[m].z,
                           weights = abs(fld.interp[m].Et[:,4]) )

    return(z_center)    
    
        
def gaussian_transverse_profile( r, w, E ) :
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

    E : float
       The a0 of the pulse
    """
    return( E*np.exp( -r**2/w**2 ) )
    
def gaussian_pulse( z, r, w0, ctau, k0, z0, E0 ) :
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

    E0 : float
       The initial E0 of the pulse
       
    Return
    ------
       A 2d array with z as the first axis and r as the second axis,
       which contains the values of the 
    
    """
    longitudinal = np.exp( -(z-z0)**2/ctau**2 )*np.cos(k0*(z-z0))
    transverse = gaussian_transverse_profile( r, w0, E0 )
    profile = longitudinal[:,np.newaxis]*transverse[np.newaxis,:]
   
    return( profile )

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
                            laser_profile, p0=np.array([1,1]) )
        # Factor 2 on the amplitude, related to the factor 2
        # in the particle gather for the modes m > 0
        fit_result[0][1] = 2*fit_result[0][1]
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
    Nz = 150
    Lz = 15.
    Nr = 100
    Lr = 10.
    Nm = 2
    # Laser pulse
    w0 = 2.
    ctau = 5.
    k0 = 2*np.pi/0.8
    E0 = 1.
    # Propagation
    N_step = 200
    N_show = 20 # interval between two plots (in number of timestep)

    show=False

    print('')
    print('Testing mode m=0 with an annular beam')
    plt.figure()
    res = test_pulse( Nz, Nr, Nm, Lz, Lr, N_step, w0, ctau,
                      k0, E0, 0, N_show, show=show )
    plt.show()
    
    print('')
    print('Testing mode m=1 with an gaussian beam')
    plt.figure()
    res = test_pulse(Nz, Nr, Nm, Lz, Lr, N_step, w0, ctau,
                     k0, E0, 1, N_show, show=show )
    plt.show()

    print('')
