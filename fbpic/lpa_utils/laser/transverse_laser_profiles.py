# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of common transverse laser profiles.
"""
import numpy as np
from scipy.special import factorial, genlaguerre, binom

# Generic classes
# ---------------

class LaserTransverseProfile(object):
    """
    Base class for all 2D transverse laser profiles.
    Such a profile can be combined with a 1D longitudinal laser profile to
    define a 3D laser profile that is valid under the paraxial approximation.

    Any new transverse laser profile should inherit from this class,
    and define its own `evaluate(x,y,z)` method, using the same signature
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

    def evaluate(self, x, y, z):
        """
        Return the complex longitudinal laser profile.

        This profile should be valid for any propagation distance z.
        In particular, it should include diffraction effects (e.g. change in
        effective waist, and effective amplitude, Gouy phase, etc. for a
        Gaussian pulse). It should not include the longitudinal laser envelope,
        since this is instead included in the longitudinal profile.

        Parameters
        -----------
        x, y, z: ndarrays (meters)
            The position at which to calculate the profile (in the lab frame)

        Returns:
        --------
        profile: ndarray
            Arrays of the same shape as x, containing the complex
            transverse profile
        """
        # The base class only defines dummy fields
        # (This should be replaced by any class that inherits from this one.)
        return np.zeros_like(x, dtype='complex')

    def squared_profile_integral(self):
        """
        Return the integral of the square of the absolute value of
        of the (complex) laser profile in the transverse plane:

        .. math::

            \\int_0^{2\\pi} d\\theta \\int_0^\\infty r \,dr|f(r, \\theta)|^2

        Returns:
        --------
        integral: float
        """
        # The base class only defines a dummy implementation
        # (This should be replaced by any class that inherits from this one.)
        return 0


# Particular classes for each transverse laser profile
# ------------------------------------------------------

class GaussianTransverseProfile(LaserTransverseProfile):
    """Class that calculates a Gaussian transverse laser profile."""

    def __init__(self, waist, zf=0., lambda0=0.8e-6, propagation_direction=1):
        """
        Define the complex transverse profile of a Gaussian laser.

        **In the focal plane** (:math:`z=z_f`), the profile translates to a
        laser with a transverse electric field:

        .. math::

            E(x,y,z=z_f) \propto \exp\left( -\\frac{r^2}{w_0^2} \\right)

        where :math:`w_0` is the laser waist and :math:`r = \sqrt{x^2 + y^2}`.

        Note that the peak amplitude of the profile is unity at the focus.

        Parameters
        ----------

        waist: float (in meter)
            Laser waist at the focal plane, defined as :math:`w_0` in the
            above formula.

        zf: float (in meter), optional
            The position of the focal plane (in the lab frame).
            If ``zf`` is not provided, the code assumes that ``zf=0.``.

        lambda0: float (in meter), optional
            The wavelength of the laser (in the lab frame), defined as
            :math:`\\lambda_0` in the above formula.
            Default: 0.8 microns (Ti:Sapph laser).

        propagation_direction: int, optional
            Indicates in which direction the laser propagates.
            This should be either 1 (laser propagates towards positive z)
            or -1 (laser propagates towards negative z).
        """
        # Initialize propagation direction and mark the profile as GPU capable
        LaserTransverseProfile.__init__(self, propagation_direction,
                                        gpu_capable=True)

        # Wavevector and Rayleigh length
        k0 = 2 * np.pi / lambda0
        zr = 0.5 * k0 * waist ** 2
        # Store the parameters
        self.k0 = k0
        self.inv_zr = 1. / zr
        self.zf = zf
        self.w0 = waist

    def evaluate(self, x, y, z):
        """
        See the docstring of LaserTransverseProfile.evaluate
        """
        # The formula for the transverse laser profile (in complex numbers) is
        # obtained by multiplying the Fourier transform of the laser at focus
        # E(k_x, k_y) = exp(-(k_x^2 + k_y^2)w_0^2/4)
        # by the paraxial propagator e^(-i ((k_x^2 +k_y^2)/2k0)(z-z_foc) )
        # and then by taking the inverse Fourier transform in x, y.
        prop_dir = self.propag_direction
        # Diffraction factor
        diffract_factor = 1. + 1j * prop_dir * (z - self.zf) * self.inv_zr
        # Calculate the argument of the complex exponential
        exp_argument = - (x ** 2 + y ** 2) / (self.w0 ** 2 * diffract_factor)
        # Get the transverse profile
        profile = np.exp(exp_argument) / diffract_factor

        return profile

    def squared_profile_integral(self):
        """
        See the docstring of LaserTransverseProfile.squared_profile_integral
        """
        return 0.5 * np.pi * self.w0**2


class LaguerreGaussTransverseProfile( LaserTransverseProfile ):
    """Class that calculates a Laguerre-Gauss transverse laser profile."""

    def __init__( self, p, m, waist, zf=0., lambda0=0.8e-6, theta0=0.,
                  propagation_direction=1 ):
        """
        Define the complex transverse profile of a Laguerre-Gauss laser.

        Unlike the :any:`DonutLikeLaguerreGaussLaser` profile, this
        profile has a phase which is independent of the azimuthal angle
        :math:`theta`, and an intensity profile which does depend on
        :math:`theta`.

        **In the focal plane** (:math:`z=z_f`), the profile translates to a
        laser with a transverse electric field:

        .. math::

            E(x,y,z=zf) \propto \, f(r, \\theta) \,
            \exp\left( -\\frac{r^2}{w_0^2} )

            \mathrm{with} \qquad f(r, \\theta) =
            \sqrt{\\frac{p!(2-\delta_{m,0})}{(m+p)!}}
            \\left( \\frac{\sqrt{2}r}{w_0} \\right)^m
            L^m_p\\left( \\frac{2 r^2}{w_0^2} \\right)
            \cos[ m(\\theta - \\theta_0)]

        where :math:`L^m_p` is a Laguerre polynomial and :math:`w_0` is the
        laser waist.

        Note that, for :math:`p=m=0`, the profile reduces to a Gaussian and the
        peak field amplitude is unity. For :math:`m \\neq 0`, the peak
        amplitude is reduced, such that the energy of the pulse is independent
        of ``p`` and ``m``.

        (For more info, see
        `Siegman, Lasers (1986) <https://www.osapublishing.org/books/bookshelf/lasers.cfm>`_,
        Chapter 16: Wave optics and Gaussian beams)

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

        waist: float (in meter)
            Laser waist at the focal plane, defined as :math:`w_0` in the
            above formula.

        zf: float (in meter), optional
            The position of the focal plane (in the lab frame).
            If ``zf`` is not provided, the code assumes ``zf=0.``.

        lambda0: float (in meter), optional
            The wavelength of the laser (in the lab frame), defined as
            :math:`\\lambda_0` in the above formula.
            Default: 0.8 microns (Ti:Sapph laser).

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
        LaserTransverseProfile.__init__(self, propagation_direction)

        # Wavevector and Rayleigh length
        k0 = 2*np.pi/lambda0
        zr = 0.5*k0*waist**2
        # Scaling factor, so that the pulse energy is independent of p and m.
        if m ==0:
            scaled_amplitude = 1.
        else:
            scaled_amplitude = np.sqrt( factorial(p)/factorial(m+p) )
        if m != 0:
            scaled_amplitude *= 2**.5
        # Store the parameters
        if m < 0 or type(m) is not int:
            raise ValueError("m should be an integer positive number.")
        self.p = p
        self.m = m
        self.scaled_amplitude = scaled_amplitude
        self.laguerre_pm = genlaguerre(self.p, self.m) # Laguerre polynomial
        self.theta0 = theta0
        self.k0 = k0
        self.inv_zr = 1./zr
        self.zf = zf
        self.w0 = waist

    def evaluate( self, x, y, z ):
        """
        See the docstring of LaserTransverseProfile.evaluate
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
        exp_argument = - (x**2 + y**2) / (self.w0**2 * diffract_factor) \
            - 1.j*(2*self.p + self.m)*psi # *Additional* Gouy phase
        # Get the transverse profile
        profile = np.exp(exp_argument) / diffract_factor \
            * scaled_radius**self.m * self.laguerre_pm(scaled_radius_squared) \
            * np.cos( self.m*(theta-self.theta0) )
        # Scale the amplitude, so that the pulse energy is independent of m and p
        profile *= self.scaled_amplitude

        return profile

    def squared_profile_integral(self):
        """
        See the docstring of LaserTransverseProfile.squared_profile_integral
        """
        return 0.5 * np.pi * self.w0**2

class DonutLikeLaguerreGaussTransverseProfile( LaserTransverseProfile ):
    """Define the complex transverse profile of a donut-like Laguerre-Gauss
    laser."""

    def __init__( self, p, m, waist, zf=0., lambda0=0.8e-6,
                  propagation_direction=1 ):
        """
        Define the complex transverse profile of a donut-like Laguerre-Gauss
        laser.

        Unlike the :any:`LaguerreGaussLaser` profile, this
        profile has a phase which depends on the azimuthal angle
        :math:`\\theta` (cork-screw pattern), and an intensity profile which
        is independent on :math:`\\theta` (donut-like).

        **In the focal plane** (:math:`z=z_f`), the profile translates to a
        laser with a transverse electric field:

        .. math::

            E(x,y,z=zf) \propto \, \, f(r) \,
            \exp\left( -\\frac{r^2}{w_0^2} \\right) \,
            \exp\left( -i m\\theta \\right)

            \mathrm{with} \qquad f(r) =
            \sqrt{\\frac{p!}{(|m|+p)!}}
            \\left( \\frac{\sqrt{2}r}{w_0} \\right)^{|m|}
            L^{|m|}_p\\left( \\frac{2 r^2}{w_0^2} \\right)

        where :math:`L^m_p` is a Laguerre polynomial and :math:`w_0` is the
        laser waist.

        (For more info, see
        `Siegman, Lasers (1986) <https://www.osapublishing.org/books/bookshelf/lasers.cfm>`_,
        Chapter 16: Wave optics and Gaussian beams)

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

        waist: float (in meter)
            Laser waist at the focal plane, defined as :math:`w_0` in the
            above formula.

        zf: float (in meter), optional
            The position of the focal plane (in the lab frame).
            If ``zf`` is not provided, the code assumes that ``zf=0.``.

        lambda0: float (in meter), optional
            The wavelength of the laser (in the lab frame), defined as
            :math:`\\lambda_0` in the above formula.
            Default: 0.8 microns (Ti:Sapph laser).

        propagation_direction: int, optional
            Indicates in which direction the laser propagates.
            This should be either 1 (laser propagates towards positive z)
            or -1 (laser propagates towards negative z).
        """
        # Initialize propagation direction
        LaserTransverseProfile.__init__(self, propagation_direction)

        # Wavevector and Rayleigh length
        k0 = 2*np.pi/lambda0
        zr = 0.5*k0*waist**2
        # Scaling factor, so that the pulse energy is independent of p and m.
        scaled_amplitude = np.sqrt( factorial(p)/factorial(abs(m)+p) )
        # Store the parameters
        self.p = p
        self.m = m
        self.scaled_amplitude = scaled_amplitude
        self.laguerre_pm = genlaguerre(self.p, abs(m)) # Laguerre polynomial
        self.k0 = k0
        self.inv_zr = 1./zr
        self.zf = zf
        self.w0 = waist

    def evaluate( self, x, y, z ):
        """
        See the docstring of LaserTransverseProfile.evaluate
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
        exp_argument = - 1.j*self.m*theta \
            - (x**2 + y**2) / (self.w0**2 * diffract_factor) \
            + 1.j*(2*self.p + abs(self.m))*psi # *Additional* Gouy phase
        # Get the transverse profile
        profile = np.exp(exp_argument) / diffract_factor \
            * scaled_radius**abs(self.m) \
            * self.laguerre_pm(scaled_radius_squared)
        # Scale the amplitude, so that the pulse energy is independent of m and p
        profile *= self.scaled_amplitude

        return profile

    def squared_profile_integral(self):
        """
        See the docstring of LaserTransverseProfile.squared_profile_integral
        """
        return 0.5 * np.pi * self.w0**2

class FlattenedGaussianTransverseProfile( LaserTransverseProfile ):
    """Define the complex transverse profile of a focused flattened Gaussian
    laser."""

    def __init__( self, w0, N=6, zf=0., lambda0=0.8e-6,
                  propagation_direction=1 ):
        """
        Define a complex transverse profile with a flattened Gaussian intensity
        distribution **far from focus** that transform into a distribution
        with rings **in the focal plane**. (See `Santarsiero et al., J.
        Modern Optics, 1997 <http://doi.org/10.1080/09500349708232927>`_)

        Increasing the parameter ``N`` increases the
        flatness of the transverse profile **far from focus**,
        and increases the number of rings **in the focal plane**.

        **In the focal plane** (:math:`z=z_f`), the profile translates to a
        laser with a transverse electric field:

        .. math::

            E(x,y,z=zf) \propto
            \exp\\left(-\\frac{r^2}{(N+1)w_0^2}\\right)
            \sum_{n=0}^N c'_n L^0_n\\left(\\frac{2\,r^2}{(N+1)w_0^2}\\right)

            \mathrm{with} Laguerre polynomials :math:`L^0_n` and
            \qquad c'_n = \sum_{m=n}^{N}\\frac{1}{2^m}\\binom{m}{n}

        - For :math:`N=0`, this is a Gaussian profile: :math:`E\propto\exp\\left(-\\frac{r^2}{w_0^2}\\right)`.

        - For :math:`N\\rightarrow\infty`, this is a Jinc profile: :math:`E\propto \\frac{J_1(r/w_0)}{r/w_0}`.

        The equivalent expression **far from focus** is

        .. math::

            E(x,y,z=\infty) \propto
            \exp\\left(-\\frac{(N+1)r^2}{w(z)^2}\\right)
            \sum_{n=0}^N \\frac{1}{n!}\left(\\frac{(N+1)\,r^2}{w(z)^2}\\right)^n

            \mathrm{with} \qquad w(z) = \\frac{\lambda_0}{\pi w_0}|z-z_{foc}|

        - For :math:`N=0`, this is a Gaussian profile: :math:`E\propto\exp\\left(-\\frac{r^2}{w_(z)^2}\\right)`.

        - For :math:`N\\rightarrow\infty`, this is a flat profile: :math:`E\propto \\Theta(w(z)-r)`.

        Parameters
        ----------
        w0: float (in meter)
            Laser spot size in the focal plane, defined as :math:`w_0` in the
            above formula.

        N: int
            Determines the "flatness" of the transverse profile, far from
            focus (see the above formula).
            Default: ``N=6`` ; somewhat close to an 8th order supergaussian.

        zf: float (in meter), optional
            The position of the focal plane (in the lab frame).
            If ``zf`` is not provided, the code assumes that ``zf=.``.

        lambda0: float (in meter), optional
            The wavelength of the laser (in the lab frame)
            Default: 0.8 microns (Ti:Sapph laser).

        propagation_direction: int, optional
            Indicates in which direction the laser propagates.
            This should be either 1 (laser propagates towards positive z)
            or -1 (laser propagates towards negative z).
        """
        # Initialize propagation direction
        LaserTransverseProfile.__init__(self, propagation_direction)

        # Ensure that N is an integer
        self.N = int(round(N))
        # Calculate effective waist of the Laguerre-Gauss modes, at focus
        self.w_foc = w0*(self.N+1)**.5
        # Wavevector and Rayleigh length
        k0 = 2 * np.pi / lambda0
        zr = 0.5 * k0 * self.w_foc**2
        # Store laser parameters
        self.k0 = k0
        self.inv_zr = 1./zr
        self.zf = zf
        # Calculate the coefficients for the Laguerre-Gaussian modes
        self.cn = np.empty(self.N+1)
        for n in range(self.N+1):
            m_values = np.arange(n, self.N+1)
            self.cn[n] = np.sum((1./2)**m_values * binom(m_values,n)) / (self.N+1)

    def evaluate( self, x, y, z ):
        """
        See the docstring of LaserTransverseProfile.evaluate
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
        exp_argument = - (x**2 + y**2) / (self.w_foc**2 * diffract_factor)
        profile = laguerre_sum * np.exp( exp_argument ) / diffract_factor

        return profile

    def squared_profile_integral(self):
        """
        See the docstring of LaserTransverseProfile.squared_profile_integral
        """
        return 0.5 * np.pi * self.w_foc**2 * sum( self.cn**2 )
