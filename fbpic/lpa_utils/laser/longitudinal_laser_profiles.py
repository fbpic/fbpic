# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of common longitudinal laser profiles.
"""
import numpy as np
from scipy.constants import c
from scipy.interpolate import interp1d
from scipy.special import factorial

# Generic classes
# ---------------

class LaserLongitudinalProfile(object):
    """
    Base class for all 1D longitudinal laser profiles.
    Such a profile can be combined with a 2D transverse laser profile to
    define a 3D laser profile that is valid under the paraxial approximation.

    Any new longitudinal laser profile should inherit from this class,
    and define its own `evaluate(z,t)` method, using the same signature
    as the method below.
    """

    def __init__(self, propagation_direction, gpu_capable=False):
        """
        Initialize the propagation direction of the laser.
        (Each subclass should call this method at initialization.)

        Parameter
        ---------
        propagation_direction: int
            Indicates in which direction the laser propagates.
            This should be either 1 (laser propagates towards positive z)
            or -1 (laser propagates towards negative z).
        gpu_capable: boolean
            Indicates whether this laser profile works with cupy arrays on
            GPU. This is usually the case if it only uses standard arithmetic
            and numpy operations. Default: False.
        """
        assert propagation_direction in [-1, 1]
        self.propag_direction = float(propagation_direction)
        self.gpu_capable = gpu_capable

    def evaluate(self, z, t):
        """
        Return the complex longitudinal laser profile.

        This profile should be valid for any z and t. Under the paraxial
        approximation, this is true if this function is a simple translation
        at c*t along the z axis. The other propagation effects, namely the
        diffraction effects, are taken into account by the transverse profile.

        Parameters
        -----------
        z: ndarray (meters)
            The longitudinal position at which to calculate the profile
            (in the lab frame)
        t: ndarray or float (seconds)
            The time at which to calculate the profile (in the lab frame)

        Returns:
        --------
        profile: ndarray
            Arrays of the same shape as z, containing the complex
            longitudinal profile
        """
        # The base class only defines dummy fields
        # (This should be replaced by any class that inherits from this one.)
        return np.zeros_like(z, dtype='complex')

    def squared_profile_integral(self):
        """
        Return the integral of the square of the absolute value of
        of the (complex) laser profile along the `z` axis:

        .. math::

            \\int_{-\\infty}^\\infty \,dz|f(z)|^2

        Returns:
        --------
        integral: float
        """
        # The base class only defines a dummy implementation
        # (This should be replaced by any class that inherits from this one.)
        return 0

# Particular classes for each longitudinal laser profile
# ------------------------------------------------------

class GaussianChirpedLongitudinalProfile(LaserLongitudinalProfile):
    """Class that calculates a Gaussian chirped longitudinal laser profile."""

    def __init__(self, tau, z0, lambda0=0.8e-6, cep_phase=0.,
                 phi2_chirp=0., propagation_direction=1):
        """
        Define the complex longitudinal profile of a Gaussian laser pulse.

        At the focus and for zero chirp, this translates to a laser with an
        axial electric field:

        .. math::

            E(z,t) \propto \exp\left( \\frac{(z-z_0-ct)^2}{c^2\\tau^2} \\right)
            \cos[ k_0( z - z_0 - ct ) - \phi_{cep} ]

        where :math:`k_0 = 2\pi/\\lambda_0` is the wavevector, :math:`\\tau`
        is the laser duration, :math:`\phi_{cep}` is the CEP phase.

        Note that, for a transform-limited pulse, the peak field amplitude of
        the profile is unity. For a non-zero chirp, the peak amplitude is
        reduced while keeping the pulse energy constant.

        Parameters
        ----------
        tau: float (in second)
            The duration of the laser (in the lab frame),
            defined as :math:`\\tau` in the above formula.

        z0: float (in meter)
            The initial position of the centroid of the laser
            (in the lab frame), defined as :math:`z_0` in the above formula.

        lambda0: float (in meter), optional
            The wavelength of the laser (in the lab frame), defined as
            :math:`\\lambda_0` in the above formula.
            Default: 0.8 microns (Ti:Sapph laser).

        cep_phase: float (in radian), optional
            The Carrier Enveloppe Phase (CEP), defined as :math:`\phi_{cep}`
            in the above formula (i.e. the phase of the laser
            oscillation, at the position where the laser enveloppe is maximum)

        phi2_chirp: float (in second^2)
            The amount of temporal chirp, at focus (in the lab frame)
            Namely, a wave packet centered on the frequency
            :math:`(\omega_0 + \delta \omega)` will reach its peak intensity
            at :math:`z(\delta \omega) = z_0 - c \phi^{(2)} \, \delta \omega`.
            Thus, a positive :math:`\phi^{(2)}` corresponds to positive chirp,
            i.e. red part of the spectrum in the front of the pulse and blue
            part of the spectrum in the back.

        propagation_direction: int, optional
            Indicates in which direction the laser propagates.
            This should be either 1 (laser propagates towards positive z)
            or -1 (laser propagates towards negative z).
        """
        # Initialize propagation direction and mark the profile as GPU capable
        LaserLongitudinalProfile.__init__(self,propagation_direction,
                                          gpu_capable=True)

        # Set and store the parameters
        self.k0 = 2*np.pi/lambda0
        self.z0 = z0
        self.cep_phase = cep_phase
        self.phi2_chirp = phi2_chirp
        self.inv_ctau2 = 1. / (c * tau) ** 2

    def evaluate(self, z, t):
        """
        See the docstring of LaserLongitudinalProfile.evaluate
        """
        # The formula for the longitudinal laser profile (in complex numbers)
        # is obtained by defining the Fourier transform of the laser at focus
        # E(\omega) = exp( -(\omega-\omega_0)^2(\tau^2/4 + i \phi^(2)/2) )
        # and then by taking the inverse Fourier transform in t.
        prop_dir = self.propag_direction
        # Stretch factor due to chirp
        stretch_factor = 1 - 2j * self.phi2_chirp * c ** 2 * self.inv_ctau2
        # Calculate the argument of the complex exponential
        exp_argument = - 1j * self.cep_phase \
                       + 1j * self.k0 * (prop_dir * (z - self.z0) - c * t) \
                       - 1. / stretch_factor * self.inv_ctau2 * \
                       (prop_dir * (z - self.z0) - c * t) ** 2
        # Get the longitudinal profile
        profile = np.exp(exp_argument) / stretch_factor ** 0.5

        return profile

    def squared_profile_integral(self):
        """
        See the docstring of LaserLongitudinalProfile.squared_profile_integral
        """
        return (0.5 * np.pi * 1./self.inv_ctau2)**.5


class CustomSpectrumLongitudinalProfile(LaserLongitudinalProfile):
    """Class that calculates a longitudinal profile with a user defined
    spectral amplitude and phase."""

    def __init__(self, z0, spectrum_file,
                 phi2_chirp=0., phi3_chirp=0., phi4_chirp=0.,
                 subtract_linear_phase=False,
                 propagation_direction=1):
        """
        Define the complex longitudinal profile of the laser pulse,
        from the spectrum provided in `spectrum_file`.

        More specifically, the temporal characteristics of the pulse are
        calculated numerically via the spectral intensity and phase which
        are provided to the class as a path to a csv file containing the data.

        More specifically, the electric field computed according to:
        .. math::

            E(z,t) \propto \int_{-\infty}^{\infty}\!\! dk \;k
                \sqrt{I(2\pi/k)}
                \,e^{i\phi(2\pi/k)}\,e^{i k (z-z0 -ct)}

        where :math:`I` and :math:`\phi` are the
        spectral intensity provided in the csv file (as a function of the
        wavelength :math:`\lambda=2\pi/k`). (The fact that the integrand
        is multiplied by :math:`k` in the above formula is because
        the csv file provides the intensity distribution over wavelengths 
        :math:`\lambda`, instead of over wavevectors :math:`k`.)

        Parameters:
        -----------
        z0: float (in meter)
            The initial position of the centroid of the laser
            (in the lab frame), defined as :math:`z_0` in the above formula.

        spectrum_file: file path
            The path to a csv file containing 3 columns (no headers).
            The three columns should represent wavelength (in m), spectral
            intensity (in the same dimension as J/m ; the exact unit/scaling
            coefficient does not matter, since the overall amplitude will
            be rescaled anyway by FBPIC when this class is used) and
            spectral phase (in radians). Use a "\t" tab as the deliminator
            in the file. The spectral phase can be omitted, in which case it
            will be assumed to be 0.

        phi2_chirp: float (in second^2), optional
            The amount of additional temporal chirp, at focus (in the
            lab frame). This spectral phase is added to that read from
            spectrum_file. A positive :math:`\phi^{(2)}` corresponds to
            a positive chirp, i.e. red part of the spectrum in the
            front of the pulse and blue part of the spectrum in the back.

        phi3_chirp: float (in second^3), optional
            The amount of additional third order dispersion,
            at focus (in the lab frame). This spectral phase is added to
            that read from spectrum_file. This parameter leads to an
            asymmetrical temporal shape of the laser pulse envelope.
            A positive :math:`\phi^{(3)}` leads to a slow rise
            of the pulse envelope and fast fall, along with appearance
            of post-pulses for higher values of :math:`\phi^{(3)}`.

        phi4_chirp: float (in second^4), optional
            Additional fourth order dispersion at focus (in the lab frame).
            This spectral phase is added to that read from spectrum_file.

        subtract_linear_phase: bool, optional
            If True, a linear slope in the spectral phase read
            from spectrum_file will be subtracted from it. A non-zero
            linear slope leads to pulse being shifted in time.

        propagation_direction: int, optional
            Indicates in which direction the laser propagates.
            This should be either 1 (laser propagates towards positive z)
            or -1 (laser propagates towards negative z).
        """
        # Initialize propagation direction
        LaserLongitudinalProfile.__init__(self, propagation_direction,
                                          gpu_capable=False)

        # Load the spectrum from the text file
        spectrum_data = np.loadtxt( spectrum_file, delimiter='\t' )
        wavelength = spectrum_data[:, 0]
        intensity = spectrum_data[:, 1]
        if spectrum_data.shape[1] < 3:
            spectral_phase = np.zeros_like(wavelength)
        else:
            spectral_phase = spectrum_data[:, 2]
        omega_axis = 2 * np.pi * c / wavelength

        # Subtract linear slope, if asked to
        if subtract_linear_phase:
            # Fit a 4 order polynomial and subtract the linear term
            poly_coeffs = np.polyfit(omega_axis, spectral_phase, 4)
            spectral_phase -= omega_axis * poly_coeffs[-2]

        # Compute central wavelength from the text file data
        self.lambda0 = np.trapz(wavelength*intensity, wavelength) * \
                        1. / np.trapz(intensity, wavelength)
        self.k0 = 2 * np.pi / self.lambda0

        # Apply spectral phase expansion terms
        spectral_phase += 1 / factorial(2) * phi2_chirp * \
                          (omega_axis - self.k0 * c) ** 2
        spectral_phase += 1 / factorial(3) * phi3_chirp * \
                          (omega_axis - self.k0 * c) ** 3
        spectral_phase += 1 / factorial(4) * phi4_chirp * \
                          (omega_axis - self.k0 * c) ** 4

        # Create functions that interpolate the spectral phase and intensity
        # from the text file data
        spectral_inten_fn = interp1d( omega_axis,
                                      intensity*wavelength**2,
                                      fill_value=0, bounds_error=False)
        spectral_phase_fn = interp1d( omega_axis, spectral_phase,
                                      fill_value=0, bounds_error=False)

        # Computation Parameters
        lambda_resolution = self.lambda0 / 1000  # spectral resolution
        dt = lambda_resolution / c
        time_window = self.lambda0 * self.lambda0 / c / lambda_resolution
        Nt = np.round(time_window/dt).astype(int)

        # Define the time array and its corresponding frequency array
        time_arr = -0.5*time_window + dt*np.arange(Nt)
        omega_arr = 2*np.pi * np.fft.fftfreq( Nt, dt )

        # Calculate the normalised temporal profile of the electric
        # field from user defined spectrum
        spectral_Efield = np.sqrt( spectral_inten_fn(omega_arr) ) * \
                            np.exp( 1j*spectral_phase_fn(omega_arr) )
        temporal_Efield = np.fft.fftshift(np.fft.fft(spectral_Efield))

        temporal_Efield = temporal_Efield/abs(temporal_Efield).max()

        # Note: this part could be potentially ported to GPU with cupy.interp
        self.interp_Efield_function = interp1d(
                           self.propag_direction*z0 - c*time_arr,
                           temporal_Efield, fill_value=0, bounds_error=False )

        # Compute integral of squared field
        self.squared_field_integral = np.trapz( abs(temporal_Efield)**2,
                                                c*time_arr )

    def get_mean_wavelength(self):
        """
        Extract the mean wavelength.
        """
        return self.lambda0

    def squared_profile_integral(self):
        """
        See the docstring of LaserLongitudinalProfile.squared_profile_integral
        """
        return self.squared_field_integral

    def evaluate(self, z, t):
        """
        See the docstring of LaserLongitudinalProfile.evaluate
        """
        # Interpolate the temporal profile of the pulse.
        # We center the pulse temporally around the pulse starting point
        profile = self.interp_Efield_function( self.propag_direction*z - c*t )

        return profile
