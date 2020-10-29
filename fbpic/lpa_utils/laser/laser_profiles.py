# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of common laser profiles.
"""
import numpy as np
from scipy.constants import c, m_e, e
from scipy.special import factorial, genlaguerre, binom

# Generic classes
# ---------------

class LaserProfile( object ):
    """
    Base class for all laser profiles.

    Any new laser profile should inherit from this class, and define its own
    `E_field` method, using the same signature as the method below.

    Profiles that inherit from this base class can be summed,
    using the overloaded + operator.
    """
    def __init__( self, propagation_direction, gpu_capable=False ):
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

    def E_field( self, x, y, z, t ):
        """
        Return the electric field of the laser

        Parameters
        -----------
        x, y, z: ndarrays (meters)
            The positions at which to calculate the profile (in the lab frame)
        t: ndarray or float (seconds)
            The time at which to calculate the profile (in the lab frame)

        Returns:
        --------
        Ex, Ey: ndarrays (V/m)
            Arrays of the same shape as x, y, z, containing the fields
        """
        # The base class only defines dummy fields
        # (This should be replaced by any class that inherits from this one.)
        return( np.zeros_like(x), np.zeros_like(x) )

    def __add__( self, other ):
        """
        Overload the + operations for laser profiles
        """
        return( SummedLaserProfile( self, other ) )


class SummedLaserProfile( LaserProfile ):
    """
    Class that represents the sum of two instances of LaserProfile
    """
    def __init__( self, profile1, profile2 ):
        """
        Initialize the sum of two instances of LaserProfile

        Parameters
        -----------
        profile1, profile2: instances of LaserProfile
        """
        # Check that both profiles propagate in the same direction
        assert profile1.propag_direction == profile2.propag_direction
        LaserProfile.__init__(self, profile1.propag_direction)

        # Register the profiles from which the sum should be calculated
        self.profile1 = profile1
        self.profile2 = profile2

    def E_field( self, x, y, z, t ):
        """
        See the docstring of LaserProfile.E_field
        """
        Ex1, Ey1 = self.profile1.E_field( x, y, z, t )
        Ex2, Ey2 = self.profile2.E_field( x, y, z, t )
        return( Ex1+Ex2, Ey1+Ey2 )


# Particular classes for each laser profile
# -----------------------------------------

class GaussianLaser( LaserProfile ):
    """Class that calculates a Gaussian laser pulse."""

    def __init__( self, a0, waist, tau, z0, zf=None, theta_pol=0.,
                    lambda0=0.8e-6, cep_phase=0., phi2_chirp=0.,
                    propagation_direction=1 ):
        """
        Define a linearly-polarized Gaussian laser profile.

        More precisely, the electric field **near the focal plane**
        is given by:

        .. math::

            E(\\boldsymbol{x},t) = a_0\\times E_0\,
            \exp\left( -\\frac{r^2}{w_0^2} - \\frac{(z-z_0-ct)^2}{c^2\\tau^2} \\right)
            \cos[ k_0( z - z_0 - ct ) - \phi_{cep} ]

        where :math:`k_0 = 2\pi/\\lambda_0` is the wavevector and where
        :math:`E_0 = m_e c^2 k_0 / q_e` is the field amplitude for :math:`a_0=1`.

        .. note::

            The additional terms that arise **far from the focal plane**
            (Gouy phase, wavefront curvature, ...) are not included in the above
            formula for simplicity, but are of course taken into account by
            the code, when initializing the laser pulse away from the focal plane.

        Parameters
        ----------

        a0: float (dimensionless)
            The peak normalized vector potential at the focal plane, defined
            as :math:`a_0` in the above formula.

        waist: float (in meter)
            Laser waist at the focal plane, defined as :math:`w_0` in the
            above formula.

        tau: float (in second)
            The duration of the laser (in the lab frame),
            defined as :math:`\\tau` in the above formula.

        z0: float (in meter)
            The initial position of the centroid of the laser
            (in the lab frame), defined as :math:`z_0` in the above formula.

        zf: float (in meter), optional
            The position of the focal plane (in the lab frame).
            If ``zf`` is not provided, the code assumes that ``zf=z0``, i.e.
            that the laser pulse is at the focal plane initially.

        theta_pol: float (in radian), optional
           The angle of polarization with respect to the x axis.

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
        LaserProfile.__init__(self, propagation_direction, gpu_capable=True)

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
        See the docstring of LaserProfile.E_field
        """
        # Note: this formula is expressed with complex numbers for compactness
        # and simplicity, but only the real part is used in the end
        # (see final return statement)
        # The formula for the laser (in complex numbers) is obtained by
        # multiplying the Fourier transform of the laser at focus
        # E(k_x,k_y,\omega) = exp( -(\omega-\omega_0)^2(\tau^2/4 + \phi^(2)/2)
        # - (k_x^2 + k_y^2)w_0^2/4 ) by the paraxial propagator
        # e^(-i(\omega/c - (k_x^2 +k_y^2)/2k0)(z-z_foc))
        # and then by taking the inverse Fourier transform in x, y, and t

        # Diffraction and stretch_factor
        prop_dir = self.propag_direction
        diffract_factor = 1. + 1j * prop_dir*(z - self.zf) * self.inv_zr
        stretch_factor = 1 - 2j * self.phi2_chirp * c**2 * self.inv_ctau2
        # Calculate the argument of the complex exponential
        exp_argument = - 1j*self.cep_phase \
            + 1j*self.k0*( prop_dir*(z - self.z0) - c*t ) \
            - (x**2 + y**2) / (self.w0**2 * diffract_factor) \
            - 1./stretch_factor*self.inv_ctau2 * \
                                    ( prop_dir*(z - self.z0) - c*t )**2
        # Get the transverse profile
        profile = np.exp(exp_argument) /(diffract_factor * stretch_factor**0.5)

        # Get the projection along x and y, with the correct polarization
        Ex = self.E0x * profile
        Ey = self.E0y * profile

        return( Ex.real, Ey.real )


class LaguerreGaussLaser( LaserProfile ):
    """Class that calculates a Laguerre-Gauss pulse."""

    def __init__( self, p, m, a0, waist, tau, z0, zf=None, theta_pol=0.,
                    lambda0=0.8e-6, cep_phase=0., theta0=0.,
                    propagation_direction=1 ):
        """
        Define a linearly-polarized Laguerre-Gauss laser profile.

        Unlike the :any:`DonutLikeLaguerreGaussLaser` profile, this
        profile has a phase which is independent of the azimuthal angle
        :math:`theta`, and an intensity profile which does depend on
        :math:`theta`.

        More precisely, the electric field **near the focal plane**
        is given by:

        .. math::

            E(\\boldsymbol{x},t) = a_0\\times E_0 \, f(r, \\theta) \,
            \exp\left( -\\frac{r^2}{w_0^2} - \\frac{(z-z_0-ct)^2}{c^2\\tau^2}
            \\right) \cos[ k_0( z - z_0 - ct ) - \phi_{cep} ]

            \mathrm{with} \qquad f(r, \\theta) =
            \sqrt{\\frac{p!(2-\delta_{m,0})}{(m+p)!}}
            \\left( \\frac{\sqrt{2}r}{w_0} \\right)^m
            L^m_p\\left( \\frac{2 r^2}{w_0^2} \\right)
            \cos[ m(\\theta - \\theta_0)]

        where :math:`L^m_p` is a Laguerre polynomial,
        :math:`k_0 = 2\pi/\\lambda_0` is the wavevector and where
        :math:`E_0 = m_e c^2 k_0 / q_e`.

        (For more info, see
        `Siegman, Lasers (1986) <https://www.osapublishing.org/books/bookshelf/lasers.cfm>`_,
        Chapter 16: Wave optics and Gaussian beams)

        .. note::

            The additional terms that arise **far from the focal plane**
            (Gouy phase, wavefront curvature, ...) are not included in the above
            formula for simplicity, but are of course taken into account by
            the code, when initializing the laser pulse away from the focal plane.

        .. warning::
            The above formula depends on a parameter :math:`m`
            (see documentation below). In order to be properly resolved by
            the simulation, a Laguerre-Gauss profile with a given :math:`m`
            requires the azimuthal modes from :math:`0` to :math:`m+1`.
            (i.e. the number of required azimuthal modes is ``Nm=m+2``)

            The non-linear plasma response for this profile (e.g.
            wakefield driven by the ponderomotive force) may require
            even more azimuthal modes.

        Parameters
        ----------

        p: int (positive)
            The order of the Laguerre polynomial. (Increasing ``p`` increases
            the number of "rings" in the radial intensity profile of the laser.)

        m: int (positive)
            The azimuthal order of the pulse.
            (In the transverse plane, the field of the pulse varies as
            :math:`\cos[m(\\theta-\\theta_0)]`.)

        a0: float (dimensionless)
            The amplitude of the pulse, defined so that the total
            energy of the pulse is the same as that of a Gaussian pulse
            with the same :math:`a_0`, :math:`w_0` and :math:`\\tau`.
            (i.e. The energy of the pulse is independent of ``p`` and ``m``.)

        waist: float (in meter)
            Laser waist at the focal plane, defined as :math:`w_0` in the
            above formula.

        tau: float (in second)
            The duration of the laser (in the lab frame),
            defined as :math:`\\tau` in the above formula.

        z0: float (in meter)
            The initial position of the centroid of the laser
            (in the lab frame), defined as :math:`z_0` in the above formula.

        zf: float (in meter), optional
            The position of the focal plane (in the lab frame).
            If ``zf`` is not provided, the code assumes that ``zf=z0``, i.e.
            that the laser pulse is at the focal plane initially.

        theta_pol: float (in radian), optional
           The angle of polarization with respect to the x axis.

        lambda0: float (in meter), optional
            The wavelength of the laser (in the lab frame), defined as
            :math:`\\lambda_0` in the above formula.
            Default: 0.8 microns (Ti:Sapph laser).

        cep_phase: float (in radian), optional
            The Carrier Enveloppe Phase (CEP), defined as :math:`\phi_{cep}`
            in the above formula (i.e. the phase of the laser
            oscillation, at the position where the laser enveloppe is maximum)

        theta0: float (in radian), optional
            The azimuthal position of (one of) the maxima of intensity, in the
            transverse plane.
            (In the transverse plane, the field of the pulse varies as
            :math:`\cos[m(\\theta-\\theta_0)]`.)

        propagation_direction: int, optional
            Indicates in which direction the laser propagates.
            This should be either 1 (laser propagates towards positive z)
            or -1 (laser propagates towards negative z).
        """
        # Initialize propagation direction
        LaserProfile.__init__(self, propagation_direction)

        # Set a number of parameters for the laser
        k0 = 2*np.pi/lambda0
        zr = 0.5*k0*waist**2
        # Scaling factor, so that the pulse energy is independent of p and m.
        if m ==0:
            scaled_amplitude = 1.
        else:
            scaled_amplitude = np.sqrt( factorial(p)/factorial(m+p) )
        if m != 0:
            scaled_amplitude *= 2**.5
        E0 = scaled_amplitude * a0 * m_e*c**2 * k0/e

        # If no focal plane position is given, use z0
        if zf is None:
            zf = z0

        # Store the parameters
        if m < 0 or type(m) is not int:
            raise ValueError("m should be an integer positive number.")
        self.p = p
        self.m = m
        self.laguerre_pm = genlaguerre(self.p, self.m) # Laguerre polynomial
        self.theta0 = theta0
        self.k0 = k0
        self.inv_zr = 1./zr
        self.zf = zf
        self.z0 = z0
        self.E0x = E0 * np.cos(theta_pol)
        self.E0y = E0 * np.sin(theta_pol)
        self.w0 = waist
        self.cep_phase = cep_phase
        self.inv_ctau2 = 1./(c*tau)**2

    def E_field( self, x, y, z, t ):
        """
        See the docstring of LaserProfile.E_field
        """
        # Diffraction factor, waist and Gouy phase
        prop_dir = self.propag_direction
        diffract_factor = 1. + 1j * prop_dir * (z - self.zf) * self.inv_zr
        w = self.w0 * abs( diffract_factor )
        psi = np.angle( diffract_factor )
        # Calculate the scaled radius and azimuthal angle
        scaled_radius_squared = 2*( x**2 + y**2 ) / w**2
        scaled_radius = np.sqrt( scaled_radius_squared )
        theta = np.angle( x + 1.j*y )
        # Calculate the argument of the complex exponential
        exp_argument = - 1j*self.cep_phase \
            + 1j*self.k0*( prop_dir*(z - self.z0) - c*t ) \
            - (x**2 + y**2) / (self.w0**2 * diffract_factor) \
            - self.inv_ctau2 * ( prop_dir*(z - self.z0) - c*t )**2 \
            - 1.j*(2*self.p + self.m)*psi # *Additional* Gouy phase
        # Get the transverse profile
        profile = np.exp(exp_argument) / diffract_factor \
            * scaled_radius**self.m * self.laguerre_pm(scaled_radius_squared) \
            * np.cos( self.m*(theta-self.theta0) )

        # Get the projection along x and y, with the correct polarization
        Ex = self.E0x * profile
        Ey = self.E0y * profile

        return( Ex.real, Ey.real )


class DonutLikeLaguerreGaussLaser( LaserProfile ):
    """Class that calculates a donut-like Laguerre-Gauss pulse."""

    def __init__( self, p, m, a0, waist, tau, z0, zf=None, theta_pol=0.,
                    lambda0=0.8e-6, cep_phase=0., propagation_direction=1 ):
        """
        Define a linearly-polarized donut-like Laguerre-Gauss laser profile.

        Unlike the :any:`LaguerreGaussLaser` profile, this
        profile has a phase which depends on the azimuthal angle
        :math:`\\theta` (cork-screw pattern), and an intensity profile which
        is independent on :math:`\\theta` (donut-like).

        More precisely, the electric field **near the focal plane**
        is given by:

        .. math::

            E(\\boldsymbol{x},t) = a_0\\times E_0 \, f(r) \,
            \exp\left( -\\frac{r^2}{w_0^2} - \\frac{(z-z_0-ct)^2}{c^2\\tau^2}
            \\right) \cos[ k_0( z - z_0 - ct ) - m\\theta - \phi_{cep} ]

            \mathrm{with} \qquad f(r) =
            \sqrt{\\frac{p!}{(|m|+p)!}}
            \\left( \\frac{\sqrt{2}r}{w_0} \\right)^{|m|}
            L^{|m|}_p\\left( \\frac{2 r^2}{w_0^2} \\right)

        where :math:`L^m_p` is a Laguerre polynomial,
        :math:`k_0 = 2\pi/\\lambda_0` is the wavevector and where
        :math:`E_0 = m_e c^2 k_0 / q_e`.

        (For more info, see
        `Siegman, Lasers (1986) <https://www.osapublishing.org/books/bookshelf/lasers.cfm>`_,
        Chapter 16: Wave optics and Gaussian beams)

        .. note::

            The additional terms that arise **far from the focal plane**
            (Gouy phase, wavefront curvature, ...) are not included in the above
            formula for simplicity, but are of course taken into account by
            the code, when initializing the laser pulse away from the focal plane.

        .. warning::
            The above formula depends on a parameter :math:`m`
            (see documentation below). In order to be properly resolved by
            the simulation, a Laguerre-Gauss profile with a given :math:`m`
            requires the azimuthal modes from :math:`0` to :math:`|m|+1`.
            (i.e. the number of required azimuthal modes is ``Nm=|m|+2``)

        Parameters
        ----------

        p: int
            The order of the Laguerre polynomial. (Increasing ``p`` increases
            the number of "rings" in the radial intensity profile of the laser.)

        m: int (positive or negative)
            The azimuthal order of the pulse. The laser phase in a given
            transverse plane varies as :math:`m \\theta`.

        a0: float (dimensionless)
            The amplitude of the pulse, defined so that the total
            energy of the pulse is the same as that of a Gaussian pulse
            with the same :math:`a_0`, :math:`w_0` and :math:`\\tau`.
            (i.e. The energy of the pulse is independent of ``p`` and ``m``.)

        waist: float (in meter)
            Laser waist at the focal plane, defined as :math:`w_0` in the
            above formula.

        tau: float (in second)
            The duration of the laser (in the lab frame),
            defined as :math:`\\tau` in the above formula.

        z0: float (in meter)
            The initial position of the centroid of the laser
            (in the lab frame), defined as :math:`z_0` in the above formula.

        zf: float (in meter), optional
            The position of the focal plane (in the lab frame).
            If ``zf`` is not provided, the code assumes that ``zf=z0``, i.e.
            that the laser pulse is at the focal plane initially.

        theta_pol: float (in radian), optional
           The angle of polarization with respect to the x axis.

        lambda0: float (in meter), optional
            The wavelength of the laser (in the lab frame), defined as
            :math:`\\lambda_0` in the above formula.
            Default: 0.8 microns (Ti:Sapph laser).

        cep_phase: float (in radian), optional
            The Carrier Enveloppe Phase (CEP), defined as :math:`\phi_{cep}`
            in the above formula (i.e. the phase of the laser
            oscillation, at the position where the laser enveloppe is maximum)

        propagation_direction: int, optional
            Indicates in which direction the laser propagates.
            This should be either 1 (laser propagates towards positive z)
            or -1 (laser propagates towards negative z).
        """
        # Initialize propagation direction
        LaserProfile.__init__(self, propagation_direction)

        # Set a number of parameters for the laser
        k0 = 2*np.pi/lambda0
        zr = 0.5*k0*waist**2
        # Scaling factor, so that the pulse energy is independent of p and m.
        scaled_amplitude = np.sqrt( factorial(p)/factorial(abs(m)+p) )
        E0 = scaled_amplitude * a0 * m_e*c**2 * k0/e

        # If no focal plane position is given, use z0
        if zf is None:
            zf = z0

        # Store the parameters
        self.p = p
        self.m = m
        self.laguerre_pm = genlaguerre(self.p, abs(m)) # Laguerre polynomial
        self.k0 = k0
        self.inv_zr = 1./zr
        self.zf = zf
        self.z0 = z0
        self.E0x = E0 * np.cos(theta_pol)
        self.E0y = E0 * np.sin(theta_pol)
        self.w0 = waist
        self.cep_phase = cep_phase
        self.inv_ctau2 = 1./(c*tau)**2

    def E_field( self, x, y, z, t ):
        """
        See the docstring of LaserProfile.E_field
        """
        # Diffraction factor, waist and Gouy phase
        prop_dir = self.propag_direction
        diffract_factor = 1. + 1j * prop_dir * ( z - self.zf ) * self.inv_zr
        w = self.w0 * abs( diffract_factor )
        psi = np.angle( diffract_factor )
        # Calculate the scaled radius and azimuthal angle
        scaled_radius_squared = 2*( x**2 + y**2 ) / w**2
        scaled_radius = np.sqrt( scaled_radius_squared )
        theta = np.angle( x + 1.j*y )
        # Calculate the argument of the complex exponential
        exp_argument = 1j*self.k0*( prop_dir*(z - self.z0) - c*t ) \
            - 1j*self.cep_phase - 1.j*self.m*theta \
            - (x**2 + y**2) / (self.w0**2 * diffract_factor) \
            - self.inv_ctau2 * ( prop_dir*(z - self.z0) - c*t )**2 \
            + 1.j*(2*self.p + abs(self.m))*psi # *Additional* Gouy phase
        # Get the transverse profile
        profile = np.exp(exp_argument) / diffract_factor \
            * scaled_radius**abs(self.m) \
            * self.laguerre_pm(scaled_radius_squared)

        # Get the projection along x and y, with the correct polarization
        Ex = self.E0x * profile
        Ey = self.E0y * profile

        return( Ex.real, Ey.real )


class FlattenedGaussianLaser( LaserProfile ):
    """Class that calculates a focused flattened Gaussian"""

    def __init__( self, a0, w0, tau, z0, N=6, zf=None, theta_pol=0.,
                    lambda0=0.8e-6, cep_phase=0., propagation_direction=1 ):
        """
        Define a linearly-polarized laser such that the transverse intensity
        profile is a flattened Gaussian **far from focus**, and a distribution
        with rings **in the focal plane**. (See `Santarsiero et al., J.
        Modern Optics, 1997 <http://doi.org/10.1080/09500349708232927>`_)

        Increasing the parameter ``N`` increases the
        flatness of the transverse profile **far from focus**,
        and increases the number of rings **in the focal plane**.

        More precisely, the expression **in the focal plane** uses the
        Laguerre polynomials :math:`L^0_n`, and reads:

        .. math::

            E(\\boldsymbol{x},t)\propto
            \exp\\left(-\\frac{r^2}{(N+1)w_0^2}\\right)
            \sum_{n=0}^N c'_n L^0_n\\left(\\frac{2\,r^2}{(N+1)w_0^2}\\right)

            \mathrm{with} \qquad c'_n = \sum_{m=n}^{N}\\frac{1}{2^m}\\binom{m}{n}

        - For :math:`N=0`, this is a Gaussian profile: :math:`E\propto\exp\\left(-\\frac{r^2}{w_0^2}\\right)`.

        - For :math:`N\\rightarrow\infty`, this is a Jinc profile: :math:`E\propto \\frac{J_1(r/w_0)}{r/w_0}`.

        The expression **far from focus** is

        .. math::

            E(\\boldsymbol{x},t)\propto
            \exp\\left(-\\frac{(N+1)r^2}{w(z)^2}\\right)
            \sum_{n=0}^N \\frac{1}{n!}\left(\\frac{(N+1)\,r^2}{w(z)^2}\\right)^n

            \mathrm{with} \qquad w(z) = \\frac{\lambda_0}{\pi w_0}|z-z_{foc}|

        - For :math:`N=0`, this is a Gaussian profile: :math:`E\propto\exp\\left(-\\frac{r^2}{w_(z)^2}\\right)`.

        - For :math:`N\\rightarrow\infty`, this is a flat profile: :math:`E\propto \\Theta(w(z)-r)`.

        Parameters
        ----------
        a0: float (dimensionless)
            The peak normalized vector potential at the focal plane.

        w0: float (in meter)
            Laser spot size in the focal plane, defined as :math:`w_0` in the
            above formula.

        tau: float (in second)
            The duration of the laser (in the lab frame)

        z0: float (in meter)
            The initial position of the centroid of the laser
            (in the lab frame)

        N: int
            Determines the "flatness" of the transverse profile, far from
            focus (see the above formula).
            Default: ``N=6`` ; somewhat close to an 8th order supergaussian.

        zf: float (in meter), optional
            The position of the focal plane (in the lab frame).
            If ``zf`` is not provided, the code assumes that ``zf=z0``, i.e.
            that the laser pulse is at the focal plane initially.

        theta_pol: float (in radian), optional
           The angle of polarization with respect to the x axis.

        lambda0: float (in meter), optional
            The wavelength of the laser (in the lab frame)
            Default: 0.8 microns (Ti:Sapph laser).

        cep_phase: float (in radian), optional
            The Carrier Enveloppe Phase (CEP, i.e. the phase of the laser
            oscillation, at the position where the laser enveloppe is maximum)

        propagation_direction: int, optional
            Indicates in which direction the laser propagates.
            This should be either 1 (laser propagates towards positive z)
            or -1 (laser propagates towards negative z).
        """
        # Initialize propagation direction
        LaserProfile.__init__(self, propagation_direction)

        # Ensure that N is an integer
        self.N = int(round(N))
        # Calculate effective waist of the Laguerre-Gauss modes, at focus
        self.w_foc = w0*(self.N+1)**.5

        k0 = 2* np.pi / lambda0
        # Rayleigh length
        zr = 0.5 * k0 * self.w_foc**2

        # Peak field
        E0 = a0 * m_e * c**2 * k0 / e

        self.E0x = E0 * np.cos(theta_pol)
        self.E0y = E0 * np.sin(theta_pol)

        if zf is None:
            zf = z0

        self.k0 = k0
        self.inv_zr = 1./zr
        self.zf = zf
        self.z0 = z0
        self.cep_phase = cep_phase
        self.inv_ctau2 = 1./(c*tau)**2

        # Calculate the coefficients for the Laguerre-Gaussian modes
        self.cn = np.empty(self.N+1)
        for n in range(self.N+1):
            m_values = np.arange(n, self.N+1)
            self.cn[n] = np.sum((1./2)**m_values * binom(m_values,n)) / (self.N+1)


    def E_field( self, x, y, z, t ):
        """
        See the docstring of LaserProfile.E_field
        """
        # Diffraction factor, waist and Gouy phase
        prop_dir = self.propag_direction
        diffract_factor = 1. + 1j * prop_dir * (z - self.zf) * self.inv_zr
        w = self.w_foc * np.abs( diffract_factor )
        psi = np.angle( diffract_factor )

        # Argument for the Laguerre polynomials
        scaled_radius_squared = 2*( x**2 + y**2 ) / w**2

        # Sum recursively over the Laguerre polynomials
        laguerre_sum = np.zeros_like( x, dtype=np.complex128 )
        for n in range(0, self.N+1):

            # Recursive calculation of the Laguerre polynomial
            # - `L` represents $L_n$
            # - `L1` represents $L_{n-1}$
            # - `L2` represents $L_{n-2}$
            if n==0:
                L = 1.
            elif n==1:
                L1 = L
                L = 1. - scaled_radius_squared
            else:
                L2 = L1
                L1 = L
                L = (((2*n -1) - scaled_radius_squared) * L1 - (n - 1) * L2) / n

            # Add to the sum, including the term for the additional Gouy phase
            laguerre_sum += self.cn[n] * np.exp( - (2j* n) * psi ) * L

        # Final profile: multiply by n-independent propagation factors
        exp_argument = - 1j*self.cep_phase \
            + 1j*self.k0*( prop_dir*(z - self.z0) - c*t ) \
            - (x**2 + y**2) / (self.w_foc**2 * diffract_factor) \
            - self.inv_ctau2 * ( prop_dir*(z - self.z0) - c*t )**2
        profile = laguerre_sum * np.exp( exp_argument ) / diffract_factor

        # Get the projection along x and y, with the correct polarization
        Ex = self.E0x * profile
        Ey = self.E0y * profile

        return( Ex.real, Ey.real )


class FewCycleLaser( LaserProfile ):
    """Class that calculates an ultra-short, tightly focussed laser"""

    def __init__( self, a0, waist, tau_fwhm, z0, zf=None, theta_pol=0.,
                    lambda0=0.8e-6, cep_phase=0., propagation_direction=1 ):
        """
        When a laser pulse is so short that it contains **only a few laser cycles**,
        the standard Gaussian profile :any:`GaussianLaser` is not well-adapted.
        This is because :any:`GaussianLaser` neglects the fact that different
        frequencies focus in different ways. In particular, when initializing
        a :any:`GaussianLaser` (with a short duration :math:`\\tau`) out of
        focus, the profile at focus will not be the expected one.

        Instead, the :any:`FewCycleLaser` profile overcomes this limitation.
        The electric field for this profile is given by (see
        `Caron & Potvilege, Journal of Modern Optics 46, 1881 (1999)
        <https://www.tandfonline.com/doi/abs/10.1080/09500349908231378>`__):

        .. math::

            E(\\boldsymbol{x},t) = Re\\left[ a_0\\times E_0\,
            e^{i\phi_{cep}} \\frac{i Z_R}{q(z)}
            \\left( 1 + \\frac{ik_0}{s}\\left(z-z_0-ct+
            \\frac{r^2}{2q(z)}\\right)\\right)^{-(s+1)} \\right]

        where :math:`k_0 = 2\pi/\\lambda_0` is the wavevector,
        :math:`E_0 = m_e c^2 k_0 / q_e` is the field amplitude for :math:`a_0=1`,
        :math:`Z_R = k_0 w_0^2/2` is the Rayleigh length,
        :math:`q(z) = z-z_f + iZ_R`, and where :math:`s`
        controls the duration of the pulse and is given by:

        .. math::

            \omega_0 \\tau_{FWHM} = s\\sqrt{2(4^{1/(s+1)}-1)}

        .. note::

            In the case of :math:`\omega_0 \\tau_{FWHM} \gg 1` (i.e. many
            laser cycles within the envelope), the above expression approaches
            that of a standard Gaussian laser pulse, and thus the :any:`FewCycleLaser`
            profile becomes equivalent to the :any:`GaussianLaser` profile
            (with :math:`\\tau_{FWHM} = \\sqrt{2\\log(2)}\\tau`).

        Parameters
        ----------

        a0: float (dimensionless)
            The peak normalized vector potential at the focal plane, defined
            as :math:`a_0` in the above formula.

        waist: float (in meter)
            Laser waist at the focal plane, defined as :math:`w_0` in the
            above formula.

        tau_FWHM: float (in second)
            The full-width half-maximum duration of the **envelope intensity** (in
            the lab frame), defined as :math:`\\tau_{FWHM}` in the above formula.

        z0: float (in meter)
            The initial position of the centroid of the laser
            (in the lab frame), defined as :math:`z_0` in the above formula.

        zf: float (in meter), optional
            The position of the focal plane (in the lab frame).
            If ``zf`` is not provided, the code assumes that ``zf=z0``, i.e.
            that the laser pulse is at the focal plane initially.

        theta_pol: float (in radian), optional
           The angle of polarization with respect to the x axis.

        lambda0: float (in meter), optional
            The wavelength of the laser (in the lab frame), defined as
            :math:`\\lambda_0` in the above formula.
            Default: 0.8 microns (Ti:Sapph laser).

        cep_phase: float (in radian), optional
            The Carrier Enveloppe Phase (CEP), defined as :math:`\phi_{cep}`
            in the above formula (i.e. the phase of the laser
            oscillation, at the position where the laser enveloppe is maximum)

        propagation_direction: int, optional
            Indicates in which direction the laser propagates.
            This should be either 1 (laser propagates towards positive z)
            or -1 (laser propagates towards negative z).
        """
        # Initialize propagation direction and mark as GPU capable
        LaserProfile.__init__(self, propagation_direction, gpu_capable=True)

        # Set a number of parameters for the laser
        k0 = 2*np.pi/lambda0
        E0 = a0*m_e*c**2*k0/e
        zr = 0.5*k0*waist**2

        # If no focal plane position is given, use z0
        if zf is None:
            zf = z0

        # Store the parameters
        self.k0 = k0
        self.zr = zr
        self.zf = zf
        self.z0 = z0
        self.E0x = E0 * np.cos(theta_pol)
        self.E0y = E0 * np.sin(theta_pol)
        self.w0 = waist
        self.cep_phase = cep_phase

        # Find the Poisson parameter s, by solving the non-linear equation
        from scipy.optimize import fsolve
        w_tau = c*k0*tau_fwhm
        sol = fsolve(lambda s: s*(2*(4**(1/(s+1))-1))**.5 - w_tau, 1.)
        self.s = sol[0]

    def E_field( self, x, y, z, t ):
        """
        See the docstring of LaserProfile.E_field
        """
        prop_dir = self.propag_direction
        inv_q = 1./( prop_dir * (z - self.zf) + 1.j*self.zr )
        # Calculate the argument inside the power function
        argument = 1. + 1.j*self.k0/self.s*(
            prop_dir*(z - self.z0) - c*t + 0.5*(x**2 + y**2)*inv_q )

        # Get the transverse profile
        profile = np.exp(1.j*self.cep_phase) * 1.j*self.zr*inv_q * \
                    argument**(-self.s-1)

        # Get the projection along x and y, with the correct polarization
        Ex = self.E0x * profile
        Ey = self.E0y * profile

        return( Ex.real, Ey.real )
