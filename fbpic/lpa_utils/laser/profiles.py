"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of common laser profiles
"""
import numpy as np
from scipy.constants import c

def gaussian_profile( z, r, t, w0, ctau, z0, zf, k0,
                      prop=1., boost=None, output_Ez_profile=False ):
    """
    Calculate the profile of a Gaussian pulse
    (normalized to a maximum amplitude of 1)

    If output_Ez_profile is True, then both the profile for Eperp and
    the profile for Ez are given.
    
    Parameters
    ----------
    ## TO BE COMPLETED
    """
    # Calculate the Rayleigh length
    zr = k0*w0**2 / 2.
    
    # When running in a boosted frame, convert the different laser quantities
    if boost is not None:
        zr, zf = boost.static_length([ zr, zf])
        ctau, z0 = boost.copropag_length([ ctau, z0 ])
        k0, = boost.wavenumber([ k0 ])

    # Calculate the laser waist and curvature in the pulse (2d arrays)
    waist = w0*np.sqrt( 1+( (z-zf) /zr)**2 )
    R = (z-zf)*( 1 + (zr/(z-zf))**2 )
    # Convert the curvature, when running a simulation in the boosted frame
    if boost is not None:
        R, = boost.curvature([ R ])

    # Longitudinal and transverse profile
    long_profile = np.exp( -(z-c*prop*t-z0)**2/ctau**2 )
    trans_profile_Eperp = w0/waist * np.exp( -(r/waist)**2 )
    # Curvature and laser oscillations (cos part)
    propag_phase = np.arctan((z-zf)/zr) - k0*r**2/(2*R) - k0*(z-c*prop*t-zf)
    curvature_oscillations_cos = np.cos( propag_phase )
    # Get the profile for the Ez fields (to ensure div(E) = 0)
    # (This uses the approximation lambda0 << ctau for long_profile )
    if output_Ez_profile:
        trans_profile_Ez = - r/zr * ( w0/waist )**3 * np.exp( -(r/waist)**2 )
        curvature_oscillations_sin = np.sin( propag_phase )
    
    # Combine profiles to create the Eperp and Ez profiles
    profile_Eperp = long_profile * trans_profile_Eperp \
                      * curvature_oscillations_cos
    if output_Ez_profile:
        profile_Ez = long_profile * trans_profile_Ez * \
        ( curvature_oscillations_sin + (z-zf)/zr*curvature_oscillations_cos )

    # Return the profiles
    if output_Ez_profile is False:
        return( profile_Eperp )
    else:
        return( profile_Eperp, profile_Ez )
    
