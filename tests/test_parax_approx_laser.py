# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It tests the Paraxial Approximation laser profile, using combinations of
different transverse and longitudinal laser profiles.
In each case, the laser is initialized before the focal plane propagated to
the focus. This test then automatically checks that the pulse energy is
correct. For a Gaussian laser, the peak normalized vector potential in focus
is also checked.

Usage :
-------
In order to let Python check the agreement:
$ python tests/test_parax_approx_laser.py
or
$ py.test -q tests/test_parax_approx_laser.py
"""
import numpy as np
from scipy.constants import c, epsilon_0, m_e, e
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser_pulse, ParaxialApproximationLaser,\
    GaussianChirpedLongitudinalProfile, GaussianTransverseProfile, \
    FlattenedGaussianTransverseProfile, DonutLikeLaguerreGaussTransverseProfile,\
    CustomSpectrumLongitudinalProfile, GaussianLaser

# Parameters
# ----------

use_cuda = True

# Simulation box
Nz = 800
zmin = -20.e-6
zmax = 20.e-6
Nr = 300
rmax = 150.e-6
Nm = 3
n_order = -1
# Laser pulse
w0 = 17.e-6
ctau = 5.e-6
k0 = 2*np.pi/0.8e-6
E_laser = 1. # a0 = 2.22
# Corresponding a0 for a Gaussian pulse, at focus
a0_gauss = 192 * 0.8e-6/w0 * np.sqrt(E_laser*c/(ctau*1.e15))
phi2_chirp = 200.e-30
# Propagation
zfoc = 1600.e-6
Lprop = 1600.e-6

# Whether to plot profiles for comparison
plot = False

# Checking the results
rtol = 1.e-2

def retrieve_pulse_energy(Er, r, dr, dz):
    """
    Function that calculates the pulse energy from the electric field data
    """
    I = c * epsilon_0 * (2 * Er) ** 2
    P = np.sum(I * 2 * np.pi * r[:, np.newaxis] * dr, axis=0)
    E_laser = np.sum(P * dz / c)
    return E_laser

def test_laser_periodic(case='gaussian'):
    """
    Function that is run by py.test
    Test the propagation of a laser in a periodic box.
    """
    # Propagate the pulse in a single step
    dt = Lprop*1./c

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
                      n_order=n_order, zmin=zmin,
                      boundaries={'z':'periodic', 'r':'reflective'} )

    # Define the profile
    reference_profile = None
    if case == 'custom':
        long_prof = CustomSpectrumLongitudinalProfile(
            z0=0., spectrum_file='./laser_spectrum.csv')
        lambda0 = long_prof.get_mean_wavelength()
        trans_prof = GaussianTransverseProfile(
            waist=w0, zf=zfoc, lambda0=lambda0)
        # For comparison: spectrum data is that of a Gaussian pulse with chirp
        reference_profile = GaussianLaser(a0_gauss, w0, ctau/c,
                                z0=0, zf=zfoc, phi2_chirp=phi2_chirp)
    elif case == 'gaussian':
        long_prof = GaussianChirpedLongitudinalProfile(
            tau=ctau/c, z0=0., phi2_chirp=0.)
        trans_prof = GaussianTransverseProfile(waist=w0, zf=zfoc)
        # For comparison
        reference_profile = GaussianLaser(a0_gauss, w0, ctau/c, z0=0, zf=zfoc)
    elif case == 'flattened_chirped':
        long_prof = GaussianChirpedLongitudinalProfile(
            tau=ctau/c, z0=0., phi2_chirp=phi2_chirp)
        trans_prof = FlattenedGaussianTransverseProfile(
            w0=w0, N=30, zf=zfoc)
    elif case == 'donut_chirped':
        long_prof = GaussianChirpedLongitudinalProfile(
            tau=ctau/c, z0=0., phi2_chirp=phi2_chirp)
        trans_prof = DonutLikeLaguerreGaussTransverseProfile(
            waist=w0, zf=zfoc, p=2, m=1)
    else:
        raise ValueError('Unknown case')
    # Construct Paraxial Approximation Laser
    profile = ParaxialApproximationLaser(long_prof, trans_prof, E_laser)
    # Initialize the laser fields
    add_laser_pulse( sim, profile )

    # Plot and compare with reference pulse
    if reference_profile is not None:
        r_2d, z_2d = np.meshgrid( sim.fld.interp[0].r,
                                    sim.fld.interp[0].z, indexing='ij')
        Ex_reference = reference_profile.E_field( r_2d, 0, z_2d, 0 )[0]
    if plot:
        import matplotlib.pyplot as plt
        plt.subplot(211)
        plt.imshow( 2*sim.fld.interp[1].Er.real.T )
        plt.colorbar()
        if reference_profile is not None:
            plt.subplot(212)
            plt.imshow( Ex_reference )
            plt.colorbar()
        plt.show()
    if reference_profile is not None:
        assert np.allclose( 2*sim.fld.interp[1].Er.real.T,
                            Ex_reference, atol=rtol*Ex_reference.max() )

    # Propagate the pulse
    sim.step(1)

    # Get pulse energy and peak electric field
    if case == 'donut_chirped':
        # mode 2 & scale Er by a factor of 2 for correct pulse energy
        Er = sim.fld.interp[2].Er.real.T.copy()*2
    else:
        # mode 1
        Er = sim.fld.interp[1].Er.real.T.copy()

    r = sim.fld.interp[1].r
    dr = sim.fld.interp[1].dr
    dz = sim.fld.interp[1].dz
    E_laser_sim = retrieve_pulse_energy(Er, r, dr, dz)

    # Check the validity
    assert np.allclose( E_laser_sim, E_laser, atol=rtol*E_laser )

    # Check peak normalized amplitude for Gaussian laser
    if case == 'gaussian':
        E0_sim = 2*Er.max()
        a0_sim = E0_sim / (m_e * c ** 2 * k0 / e)
        assert np.allclose( a0_sim, 2.22, atol=3*rtol*2.22 )

if __name__ == '__main__' :
    cases = ['custom', 'gaussian', 'flattened_chirped', 'donut_chirped']
    for case in cases:
        # Run the testing function
        test_laser_periodic(case)
