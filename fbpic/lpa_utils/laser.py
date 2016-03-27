"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of utilities for laser initialization
"""
import numpy as np
from scipy.constants import m_e, c, e
from .boosted_frame import BoostConverter

def add_laser( sim, a0, w0, ctau, z0, zf=None, lambda0=0.8e-6,
               theta_pol=0., fw_propagating=True,
               update_spectral=True, gamma_boost=None ) :
    """
    Add a linearly-polarized, Gaussian laser pulse in the Fields object

    The laser profile is added to the interpolation grid, and then
    transformed into spectral space if update_spectral is True
    
    Parameters
    ----------
    fld : a Simulation object
       The structure that contains the fields of the simulation
    
    a0 : float (unitless)
       The a0 of the pulse at focus
    
    w0 : float (in meters)
       The waist of the pulse at focus

    ctau : float (in meters)
       The "longitudinal waist" (or pulse length) of the pulse

    z0 : float (in meters)
       The position of the laser centroid relative to z=0.

    zf : float (in meters), optional
       The position of the laser focus relative to z=0.
       If not provided, then the laser focus is at z0
    
    lambda0 : float (in meters), optional
       The central wavelength of the laser
       Default : 0.8 microns (Ti:Sapph laser)

    theta_pol : float (in radians), optional
       The angle of polarization with respect to the x axis
       Default : 0 rad

    fw_propagating : bool, optional
       Wether the laser is propagating in the forward or backward direction
       Default : True (forward propagation)

    update_spectral : bool, optional
       Wether to update the fields in spectral space after modifying the
       fields on the interpolation grid.
       Default : True

    gamma_boost : float, optional
        (only works when fw_propagating=1)
        When initializing the laser in a boosted frame, set the value of
        `gamma_boost` to the corresponding Lorentz factor. All the other
        quantities (ctau, zf, etc.) are to be given in the lab frame.
    """
    # Set a number of parameters for the laser
    k0 = 2*np.pi/lambda0
    E0 = a0*m_e*c**2*k0/e      # Amplitude at focus
    zr = np.pi*w0**2/lambda0   # Rayleigh length
    # Get the polarization component
    # (Due to the Fourier transform along theta, the
    # polarization angle becomes a complex phase in the fields)
    exptheta = np.exp(1.j*theta_pol)
    # Sign for the propagation (1 for forward propagation, and -1 otherwise)
    prop = 2*int(fw_propagating) - 1.
    # Set default focusing position
    if zf is None : zf = z0

    # When running a simulation in boosted frame, convert these parameters
    if (gamma_boost is not None) and (fw_propagating==True):
        boost = BoostConverter( gamma_boost )
        zr, zf = boost.static_length([ zr, zf])
        ctau, z0, lambda0 = boost.copropag_length([ ctau, z0, lambda0 ])
        k0, E0 = boost.wavenumber([ k0, E0 ])

    # Get the 2D mesh for z and r
    # (When running a simulation in boosted frame, then z is the coordinate
    # in the boosted frame -- if the Fields object was correctly initialized.)
    r, z = np.meshgrid( fld.interp[1].r, fld.interp[1].z )
    
    # Calculate the laser waist and curvature in the pulse (2d arrays)
    waist = w0*np.sqrt( 1+( (z-zf) /zr)**2 )
    R = (z-zf)*( 1 + (zr/(z-zf))**2 )
    # Convert the curvature, when running a simulation in the boosted frame
    if (gamma_boost is not None) and (fw_propagating==True):
        R, = boost.curvature([ R ])

    # Longitudinal and transverse profile
    long_profile = np.exp( -(z-z0)**2/ctau**2 )
    trans_profile = w0/waist * np.exp( -(r/waist)**2 )
    # Curvature and laser oscillations (cos part)
    propag_phase = np.arctan((z-zf)/zr) - k0*r**2/(2*R) - k0*(z-zf)
    curvature_oscillations_cos = np.cos( propag_phase )
    # Combine profiles
    profile = long_profile * trans_profile * curvature_oscillations_cos

    # Add the Er and Et fields to the mode m=1 (linearly polarized laser)
    # (The factor 0.5 is related to the fact that there is a factor 2
    # in the gathering function, for the particles)
    fld.interp[1].Er +=  0.5  * E0 * exptheta * profile
    fld.interp[1].Et += -0.5j * E0 * exptheta * profile
    fld.interp[1].Br +=  0.5j * prop * E0/c * exptheta * profile
    fld.interp[1].Bt +=  0.5  * prop * E0/c * exptheta * profile

    # Get the profile for the Ez fields (to ensure div(E) = 0)
    # (This uses the approximation lambda0 << ctau for long_profile )
    # Transverse profile
    trans_profile = - r/zr * ( w0/waist )**3 * np.exp( -(r/waist)**2 )
    # Curvature and laser oscillations (sin part)
    curvature_oscillations_sin = np.sin( propag_phase )
    # Combine profiles
    profile = long_profile * trans_profile * \
    ( curvature_oscillations_sin + (z-zf)/zr*curvature_oscillations_cos )

    # Add the Ez fields to the mode m=1 (linearly polarized laser)
    fld.interp[1].Ez +=  0.5  * E0 * exptheta * profile
    fld.interp[1].Bz +=  0.5j * prop * E0/c * exptheta * profile
        
    # Up to now, only the spectral grid was modified.
    # Now convert the fields to spectral space.
    if update_spectral :
        fld.interp2spect('E')
        fld.interp2spect('B')

