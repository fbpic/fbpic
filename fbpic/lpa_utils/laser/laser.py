# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of utilities for laser initialization
"""
from scipy.constants import c
from fbpic.lpa_utils.boosted_frame import BoostConverter
from .laser_profiles import GaussianLaser
from .direct_injection import add_laser_direct
from .antenna_injection import LaserAntenna

def add_laser_pulse( sim, laser_profile, gamma_boost=None,
                method='direct', z0_antenna=None, v_antenna=0.):
    """
    Introduce a laser pulse in the simulation.

    The laser is either added directly to the interpolation grid initially
    (method= ``direct``) or it is progressively emitted by an antenna
    (method= ``antenna``).

    Parameters
    ----------
    sim: a Simulation object
       The structure that contains the simulation.

    laser_profile: a valid laser profile object
        Laser profiles can be imported from ``fbpic.lpa_utils.laser``

    gamma_boost: float, optional
        When initializing the laser in a boosted frame, set the value of
        ``gamma_boost`` to the corresponding Lorentz factor.
        In this case, ``laser_profile`` should be initialized with all its
        physical quantities (wavelength, etc...) given in the lab frame,
        and the function ``add_laser_pulse`` will automatically convert
        these properties to their boosted-frame value.

    method: string, optional
        Whether to initialize the laser directly in the box
        (method= ``direct``) or through a laser antenna (method= ``antenna``).
        Default: ``direct``

    z0_antenna: float, optional (meters)
       Required for the ``antenna`` method: initial position
       (in the lab frame) of the antenna.

    v_antenna: float, optional (meters per second)
       Only used for the ``antenna`` method: velocity of the antenna
       (in the lab frame)

    Example
    -------
    In order to initialize a Laguerre-Gauss profile with a waist of
    5 microns and a duration of 30 femtoseconds, centered at :math:`z=3`
    microns initially:

    ::

        from fbpic.lpa_utils.laser import add_laser_pulse, LaguerreGaussLaser

        profile = LaguerreGaussLaser(a0=0.5, waist=5.e-6, tau=30.e-15, z0=3.e-6, p=1, m=0)

        add_laser_pulse( sim, profile )
    """
    # Prepare the boosted frame converter
    if (gamma_boost is not None) and (gamma_boost != 1.):
        if laser_profile.propag_direction == 1:
            boost = BoostConverter( gamma_boost )
        else:
            raise ValueError('For now, backward-propagating lasers '
                         'cannot be used in the boosted-frame.')
    else:
        boost = None

    # Handle the introduction method of the laser
    if method == 'direct':
        # Directly add the laser to the interpolation object
        add_laser_direct( sim, laser_profile, boost )

    elif method == 'antenna':
        # Add a laser antenna to the simulation object
        if z0_antenna is None:
            raise ValueError('You need to provide `z0_antenna`.')
        dr = sim.fld.interp[0].dr
        Nr = sim.fld.interp[0].Nr
        Nm = sim.fld.Nm
        sim.laser_antennas.append(
            LaserAntenna( laser_profile, z0_antenna, v_antenna,
                            dr, Nr, Nm, boost, use_cuda=sim.use_cuda ) )

    else:
        raise ValueError('Unknown laser method: %s' %method)


# The function below is kept for backward-compatibility

def add_laser( sim, a0, w0, ctau, z0, zf=None, lambda0=0.8e-6,
               cep_phase=0., phi2_chirp=0., theta_pol=0.,
               gamma_boost=None, method='direct',
               fw_propagating=True, update_spectral=True,
               z0_antenna=None, v_antenna=0. ):
    """
    Introduce a linearly-polarized, Gaussian laser in the simulation.

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

    The laser is either added directly to the interpolation grid initially
    (method= ``direct``) or it is progressively emitted by an antenna
    (method= ``antenna``).

    Parameters
    ----------
    sim: a Simulation object
       The structure that contains the simulation.

    a0: float (dimensionless)
        The peak normalized vector potential at the focal plane, defined
        as :math:`a_0` in the above formula.

    w0: float (in meter)
        Laser waist at the focal plane, defined as :math:`w_0` in the
        above formula.

    ctau: float (in meter)
        The duration of the laser (in the lab frame),
        defined as :math:`c\\tau` in the above formula.

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

    gamma_boost: float, optional
        When initializing the laser in a boosted frame, set the value of
        `gamma_boost` to the corresponding Lorentz factor. All the other
        quantities (ctau, zf, etc.) are to be given in the lab frame.

    method: string, optional
        Whether to initialize the laser directly in the box (method=`direct`)
        or through a laser antenna (method=`antenna`)

    fw_propagating: bool, optional
       Whether the laser is propagating in the forward or backward direction.

    update_spectral: bool, optional
       Only for the `direct` method: Wether to update the fields in spectral
       space after modifying the fields on the interpolation grid.

    z0_antenna: float, optional (meters)
       Only for the `antenna` method: initial position (in the lab frame)
       of the antenna. If not provided, then the z0_antenna is set to zf.

    v_antenna: float, optional (meters per second)
       Only used for the ``antenna`` method: velocity of the antenna
       (in the lab frame)
    """
    # Pick the propagation direction
    if fw_propagating:
        propagation_direction = 1
    else:
        propagation_direction = -1

    # Create a Gaussian laser profile
    laser_profile = GaussianLaser( a0, waist=w0, tau=ctau/c, z0=z0,
        zf=zf, theta_pol=theta_pol, lambda0=lambda0,
        cep_phase=cep_phase, phi2_chirp=phi2_chirp,
        propagation_direction=propagation_direction )

    # Add it to the simulation
    add_laser_pulse( sim, laser_profile, gamma_boost=gamma_boost,
        method=method, z0_antenna=z0_antenna, v_antenna=v_antenna )
