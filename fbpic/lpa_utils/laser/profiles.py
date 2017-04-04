# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of common laser profiles
"""
import numpy as np
from scipy.constants import c

def gaussian_profile( z, r, t, w0, ctau, z0, zf, k0, phi2=0.,
                      prop=1., boost=None, output_Ez_profile=False ):
    """
    Calculate the profile of a Gaussian pulse
    (normalized to a maximum amplitude of 1)

    If output_Ez_profile is True, then both the profile for Eperp and
    the profile for Ez are given. (The field Ez is never 0 when there are
    transverse variations of the intensity, due to div(E)=0 )

    Parameters
    ----------
    z, r: 1darrays or 2darrays (meters)
        The positions at which to calculate the profile
        *either in the lab frame or in the boosted frame*
        (if these positions are boosted-frame positions,
        a boost object needs to be passed)
        Both arrays should have the same shape

    t: float (seconds)
        The time at which to calculate the profile

    w0: float (m)
        The waist at focus

    ctau: float (m)
        The length of the pulse *in the lab frame*

    z0: float (m)
        The initial position of the centroid *in the lab frame*

    zf: float (m)
        The initial position of the focal plane *in the lab frame*

    k0: float (m)
        The wavenumber *in the lab frame*

    phi2: float (in second^2)
        The amount of temporal chirp, at focus *in the lab frame*
        Namely, a wave packet centered on the frequency (w0 + dw) will
        reach its peak intensity at a time z(dw) = z0 - c*phi2*dw.
        Thus, a positive phi2 corresponds to positive chirp, i.e. red part
        of the spectrum in the front of the pulse and blue part of the
        spectrum in the back.

    prop: float (either +1 or -1)
        Whether the laser is forward or backward propagating

    boost: a BoostConverter object or None
        Contains the information about the boosted frame

    output_Ez_profile: bool
        Whether to also output the Ez profile

    Returns
    -------
    If output_Ez_profile is False:
       an array of reals of the same shape as z and r, containing Eperp
    If output_Ez_profile is True:
       a tuple with 2 array of reals of the same shape as z and r
    """
    # Calculate the Rayleigh length
    zr = 0.5*k0*w0**2
    inv_zr = 1./zr
    inv_ctau2 = 1./ctau**2

    # When running in a boosted frame, convert the position and time at
    # which to find the laser amplitude.
    if boost is not None:
        zlab_source = self.boost.gamma0*( z + self.boost.beta0*c*t )
        tlab_source = self.boost.gamma0*( t + self.boost.beta0*z/c )
        # Overwrite boosted frame values, within the scope of this function
        z = zlab_source
        t = tlab_source

    # Lab-frame formula for the laser:
    # Diffraction and stretch_factor
    diffract_factor = 1. - 1j*(z-zf)*inv_zr
    stretch_factor = 1 + 2j * phi2 * c**2 * inv_ctau2

    # Calculate the argument of the complex exponential
    exp_argument = 1j*k0*(c*t + z0 - z) - r**2 / (w0**2 * diffract_factor) \
      - 1./stretch_factor * inv_ctau2 * ( c*t  + z0 - z )**2

    # Get the transverse profile
    profile_Eperp = np.exp(exp_argument) \
        / ( diffract_factor * stretch_factor**0.5 )

    # Get the profile for the Ez fields (to ensure div(E) = 0)
    # (This uses the approximation lambda0 << ctau for long_profile )
    if output_Ez_profile:
        profile_Ez = 1.j * r * inv_zr / diffract_factor * profile_Eperp

    # Return the profiles
    if output_Ez_profile is False:
        return( profile_Eperp.real )
    else:
        return( profile_Eperp.real, profile_Ez.real )
