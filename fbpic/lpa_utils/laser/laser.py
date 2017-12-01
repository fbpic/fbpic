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
                    method='direct', z0_antenna=None, fw_propagating=True ):
    """
    Introduce a laser pulse in the simulation

    The laser is either added directly to the interpolation grid initially
    (method=`direct`) or it is progressively emitted by an antenna
    (method=`antenna`)

    Parameters
    ----------
    sim: a Simulation object
       The structure that contains the simulation.

    laser_profile: a valid laser profile object
        Laser profiles can be imported from fbpic.lpa_utils.laser

    gamma_boost: float, optional
        When initializing the laser in a boosted frame, set the value of
        `gamma_boost` to the corresponding Lorentz factor.

    method: string, optional
        Whether to initialize the laser directly in the box (method=`direct`)
        or through a laser antenna (method=`antenna`)

    fw_propagating: bool, optional
       Only for the `direct` method: Whether the laser is propagating in the
       forward or backward direction.

    z0_antenna: float, optional (meters)
       Only for the `antenna` method: initial position (in the lab frame)
       of the antenna. If not provided, then the z0_antenna is set to zf.
    """
    # Prepare the boosted frame converter
    if (gamma_boost is not None) and (fw_propagating==True):
        boost = BoostConverter( gamma_boost )
    else:
        boost = None

    # Handle the introduction method of the laser
    if method == 'direct':
        # Directly add the laser to the interpolation object
        add_laser_direct( sim, laser_profile, fw_propagating, boost )

    elif method == 'antenna':
        # Add a laser antenna to the simulation object
        if z0_antenna is None:
            raise ValueError('You need to provide `z0_antenna`.')
        dr = sim.fld.interp[0].dr
        Nr = sim.fld.interp[0].Nr
        Nm = sim.fld.Nm
        sim.laser_antennas.append(
            LaserAntenna( laser_profile, z0_antenna, dr, Nr, Nm, boost ))

    else:
        raise ValueError('Unknown laser method: %s' %method)


# The function below is kept for backward-compatibility

def add_laser( sim, a0, w0, ctau, z0, zf=None, lambda0=0.8e-6,
               cep_phase=0., phi2_chirp=0., theta_pol=0.,
               gamma_boost=None, method='direct',
               fw_propagating=True, update_spectral=True, z0_antenna=None ):
    """
    Introduce a linearly-polarized, Gaussian laser in the simulation

    The laser is either added directly to the interpolation grid initially
    (method=`direct`) or it is progressively emitted by an antenna
    (method=`antenna`)

    Parameters
    ----------
    sim: a Simulation object
       The structure that contains the simulation.

    a0: float (unitless)
       The a0 of the pulse at focus (in the lab-frame).

    w0: float (in meters)
       The waist of the pulse at focus.

    ctau: float (in meters)
       The pulse length (in the lab-frame), defined as:
       E_laser ~ exp( - (z-ct)**2 / ctau**2 )

    z0: float (in meters)
       The position of the laser centroid relative to z=0 (in the lab-frame).

    zf: float (in meters), optional
       The position of the laser focus relative to z=0 (in the lab-frame).
       Default: the laser focus is at z0.

    lambda0: float (in meters), optional
       The central wavelength of the laser (in the lab-frame).
       Default: 0.8 microns (Ti:Sapph laser).

    cep_phase: float (rad)
        Carrier Enveloppe Phase (CEP), i.e. the phase of the laser
        oscillations, at the position where the laser enveloppe is maximum.

    phi2_chirp: float (in second^2)
        The amount of temporal chirp, at focus *in the lab frame*
        Namely, a wave packet centered on the frequency (w0 + dw) will
        reach its peak intensity at a time z(dw) = z0 - c*phi2*dw.
        Thus, a positive phi2 corresponds to positive chirp, i.e. red part
        of the spectrum in the front of the pulse and blue part of the
        spectrum in the back.

    theta_pol: float (in radians), optional
       The angle of polarization with respect to the x axis.
       Default: 0 rad.

    gamma_boost: float, optional
        When initializing the laser in a boosted frame, set the value of
        `gamma_boost` to the corresponding Lorentz factor. All the other
        quantities (ctau, zf, etc.) are to be given in the lab frame.

    method: string, optional
        Whether to initialize the laser directly in the box (method=`direct`)
        or through a laser antenna (method=`antenna`)

    fw_propagating: bool, optional
       Only for the `direct` method: Wether the laser is propagating in the
       forward or backward direction.

    update_spectral: bool, optional
       Only for the `direct` method: Wether to update the fields in spectral
       space after modifying the fields on the interpolation grid.

    z0_antenna: float, optional (meters)
       Only for the `antenna` method: initial position (in the lab frame)
       of the antenna. If not provided, then the z0_antenna is set to zf.
    """
    # Create a Gaussian laser profile
    laser_profile = GaussianLaser( a0, waist=w0, tau=ctau/c, z0=z0,
        zf=zf, theta_pol=theta_pol, lambda0=lambda0,
        cep_phase=cep_phase, phi2_chirp=phi2_chirp )

    # Add it to the simulation
    add_laser_pulse( sim, laser_profile, gamma_boost=gamma_boost,
        method=method, z0_antenna=z0_antenna, fw_propagating=fw_propagating )
