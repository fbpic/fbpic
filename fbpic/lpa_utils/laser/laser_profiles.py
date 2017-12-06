# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of common laser profiles.
"""
import numpy as np
from scipy.constants import c, m_e, e

class GaussianLaser( object ):
    """Class that calculates a Gaussian laser pulse."""

    def __init__( self, a0, waist, tau, z0, zf=None, theta_pol=0.,
                    lambda0=0.8e-6, cep_phase=0., phi2_chirp=0. ):
        """
        Define a linearly-polarized Gaussian laser profile.

        Parameters:
        -----------
        a0: float (dimensionless)
            The peak normalized vector potential, in the focal plane

        waist: float (in meters)
            Laser waist in the focal plane, defined as :math:`w_0` below:

            ..math:

                E(\boldsymbol{x},t) \propto \exp\left( -\frac{\boldsymbol{x}_\perp^2}{w_0^2} \right)

        tau: float (in meters^-1)
            The duration of the laser, defined as :math:`\tau` below:

            .. math::

                E(\boldsymbol{x},t) \propto \exp\left( -\frac{(t-z_0/c)^2}{\tau^2} \right)

        z0: float (m)
            The initial position of the centroid of the laser (in the lab frame)

        zf: float (m), optional
            The position of the focal plane (in the lab frame)

        theta_pol: float (in radians), optional
           The angle of polarization with respect to the x axis.
           Default: 0 rad.

        lambda0: float (m)
            The wavelength of the laser (in the lab frame)

        cep_phase: float (rad)
            The Carrier Enveloppe Phase (CEP), i.e. the phase of the laser
            oscillation, at the position where the laser enveloppe is maximum.

        phi2_chirp: float (in second^2)
            The amount of temporal chirp, at focus (in the lab frame)
            Namely, a wave packet centered on the frequency (w0 + dw) will
            reach its peak intensity at a time z(dw) = z0 - c*phi2*dw.
            Thus, a positive phi2 corresponds to positive chirp, i.e. red part
            of the spectrum in the front of the pulse and blue part of the
            spectrum in the back.
        """
        # Set a number of parameters for the laser
        k0 = 2*np.pi/lambda0
        E0 = a0*m_e*c**2*k0/e
        zr = 0.5*k0*waist**2

        # If no focal plane position is given, use z0
        if zf is None:
            zf = z0

        # Store the parameters
        self.k0 = k0
        self.inv_zr = 1./zr
        self.zf = zf
        self.z0 = z0
        self.E0x = E0 * np.cos(theta_pol)
        self.E0y = E0 * np.sin(theta_pol)
        self.w0 = waist
        self.cep_phase = cep_phase
        self.phi2_chirp = phi2_chirp
        self.inv_ctau2 = 1./(c*tau)**2

    def E_field( self, x, y, z, t ):
        """
        Return the electric field of the laser

        Parameters:
        -----------
        x, y, z: ndarrays (meters)
            The positions at which to calculate the profile (in the lab frame)

        t: ndarray or float (seconds)
            The time at which to calculate the profile (in the lab frame)

        Returns:
        --------
        Ex, Ey: ndarrays of the same shape as x, y, z
        """
        # Note: this formula is expressed with complex numbers for compactness
        # and simplicity, but only the real part is used in the end
        # (see final return statement)
        # The formula for the laser (in complex numbers) is obtained by
        # multiplying the Fourier transform of the laser at focus
        # E(k_x,k_y,\omega) = exp( -(\omega-\omega_0)^2(\tau^2/4 + \phi^(2)/2)
        # - (k_x^2 + k_y^2)w_0^2/4 ) by the paraxial propagator
        # e^(i(\omega/c - (k_x^2 +k_y^2)/2k0)(z-z_foc))
        # and then by taking the inverse Fourier transform in x, y, and t

        # Diffraction and stretch_factor
        diffract_factor = 1. - 1j * ( z - self.zf ) * self.inv_zr
        stretch_factor = 1 + 2j * self.phi2_chirp * c**2 * self.inv_ctau2
        # Calculate the argument of the complex exponential
        exp_argument = 1j*self.cep_phase + 1j*self.k0*( c*t + self.z0 - z ) \
            - (x**2 + y**2) / (self.w0**2 * diffract_factor) \
            - 1./stretch_factor * self.inv_ctau2 * ( c*t  + self.z0 - z )**2
        # Get the transverse profile
        profile = np.exp(exp_argument) / ( diffract_factor * stretch_factor**0.5 )

        # Get the projection along x and y, with the correct polarization
        Ex = self.E0x * profile
        Ey = self.E0y * profile

        return( Ex.real, Ey.real )
