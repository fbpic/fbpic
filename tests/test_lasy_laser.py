# Copyright 2023, FBPIC contributors
# Authors: Remi Lehe
# License: 3-Clause-BSD-LBNL
#
# This file is inspired from a WarpX test for a lasy laser. The test does the following:
#
# - Generate an input lasy file with a Laguerre Gaussian laser pulse.
# - Run an FBPIC simulation, until time T when the pulse is fully injected
# - Compute the theory for laser envelope at time T
# - Compare theory and simulation in RZ, for both envelope and central frequency
import os, shutil
import numpy as np
from scipy.constants import c, epsilon_0
from scipy.signal import hilbert
from scipy.special import genlaguerre
from openpmd_viewer.addons import LpaDiagnostics
from fbpic.main import Simulation
from fbpic.openpmd_diag import FieldDiagnostic, BackTransformedFieldDiagnostic
from fbpic.lpa_utils.laser import add_laser_pulse, FromLasyFileLaser

from lasy.laser import Laser
from lasy.profiles import CombinedLongitudinalTransverseProfile
from lasy.profiles.longitudinal import GaussianLongitudinalProfile
from lasy.profiles.transverse import LaguerreGaussianTransverseProfile

# Maximum acceptable error for this test
relative_error_threshold = 0.065

# Whether to plot the results
plot = True

#Parameters of the Laguerre Gaussian beam
wavelength = 1.e-6
w0 = 12.e-6
tau = 10.e-15
t_c = 3 * tau
laser_energy = 1.0
E_max = np.sqrt( 2*(2/np.pi)**(3/2)*laser_energy / (epsilon_0*w0**2*c*tau) )

# Function for the envelope
def laguerre_env(T, X, Y, Z, p, m):
    if m>0:
        complex_position= X -1j * Y
    else:
        complex_position= X +1j * Y
    inv_w0_2 = 1.0/(w0**2)
    inv_tau2 = 1.0/(tau**2)
    radius = abs(complex_position)
    scaled_rad_squared = (radius**2)*inv_w0_2
    envelope = (
            ( np.sqrt(2) * complex_position / w0 )** m
            *  genlaguerre(p, m)(2 * scaled_rad_squared)
            * np.exp(-scaled_rad_squared)
            * np.exp(-( inv_tau2 / (c**2) ) * (Z-T*c)**2)
        )
    return E_max * np.real(envelope)

def compare_simulation_with_theory(data_dir):

    ts = LpaDiagnostics(os.path.join(data_dir, 'hdf5/'))
    F_laser, info = ts.get_field( 'E', 'x', iteration=ts.iterations[-1])
    t = ts.current_t
    X, Z = np.meshgrid(info.r, info.z, sparse=False, indexing='ij')
    Y = np.zeros_like(X)

    # Compute the theory for envelope
    env_theory = abs( laguerre_env(+t_c-t, X,Y,Z,p=0,m=1) + laguerre_env(-t_c+t, X,Y,Z,p=0,m=1) )

    # Read laser field in PIC simulation, and compute envelope
    env = abs(hilbert(F_laser))

    # Plot results
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        extent = info.imshow_extent
        plt.figure(figsize=(8,6))
        plt.subplot(221)
        plt.title('PIC field')
        plt.imshow(F_laser, extent=extent, aspect='auto')
        plt.colorbar()
        plt.subplot(222)
        plt.title('PIC envelope')
        plt.imshow(env, extent=extent, aspect='auto')
        plt.colorbar()
        plt.subplot(223)
        plt.title('Theory envelope')
        plt.imshow(env_theory, extent=extent, aspect='auto')
        plt.colorbar()
        plt.subplot(224)
        plt.title('Difference')
        plt.imshow(env-env_theory, extent=extent, aspect='auto')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('comparison.png', bbox_inches='tight')

    relative_error_env = np.sum(np.abs(env-env_theory)) / np.sum(np.abs(env))
    print("Relative error envelope: ", relative_error_env)
    assert(relative_error_env < relative_error_threshold)

    fft_F_laser = np.fft.fftn(F_laser)

    k_x = np.fft.fftfreq(F_laser.shape[0],info.dr)
    k_z = np.fft.fftfreq(F_laser.shape[1],info.dz)

    pos_max = np.unravel_index(np.abs(fft_F_laser).argmax(), fft_F_laser.shape)

    freq = c * np.sqrt((k_x[pos_max[0]])**2 + (k_z[pos_max[1]])**2)
    exp_freq = c/wavelength
    relative_error_freq = np.abs(freq-exp_freq)/exp_freq
    print("Relative error frequency: ", relative_error_freq)
    assert(relative_error_freq < relative_error_threshold)


def run_and_check_laser_emission(gamma_b, data_dir, lasy_geometry):
    """
    Run a simulation that emits a laser from a lasy file, either in the lab frame
    (gamma_b = None) or in the boosted frame (gamma_b > 1.)

    Parameter
    ---------
    gamma_b: float or None
        The Lorentz factor of the boosted frame

    data_dir: string
        Directory in which the simulation writes data

    lasy_geometry: string
        Either "rt" or "xyt"
    """
    # Move into directory `tests`
    os.chdir('./tests')

    # Create a Laguerre Gaussian laser in RZ geometry using lasy
    pol = (1, 0)
    profile = CombinedLongitudinalTransverseProfile(
        wavelength, pol, laser_energy,
        GaussianLongitudinalProfile(wavelength, tau, t_peak=0),
        LaguerreGaussianTransverseProfile(w0, p=0, m=1),
    )
    if lasy_geometry == "rt":
        lo = (0e-6, -3*tau)
        hi = (3*w0, 3*tau)
        npoints = (100,100)
        laser = Laser(lasy_geometry, lo, hi, npoints, profile, n_azimuthal_modes=2)
    elif lasy_geometry == "xyt":
        lo = (-3*w0, -3*w0, -3*tau)
        hi = (3*w0, 3*w0, 3*tau)
        npoints = (200,200,200)
        laser = Laser(lasy_geometry, lo, hi, npoints, profile)
    laser.propagate(-3 * c * tau)
    laser.write_to_file("laguerrelaserRZ")

    # Create an FBPIC simulation that reads this lasy file
    # Initialize the simulation object
    Nz = 1024
    Nr = 32
    zmax = 0
    zmin = -6 * c * tau
    rmax = 3 * w0
    Nm = 3
    dt = (zmax-zmin)/Nz/c

    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, use_cuda=True, zmin=zmin,
                     boundaries={'z':'open', 'r':'reflective'}, gamma_boost=gamma_b)

    # Remove the particles
    sim.ptcl = []

    # Add a moving window
    sim.set_moving_window(v=c)

    # Add the laser
    laser_profile = FromLasyFileLaser( 'laguerrelaserRZ_00000.h5' )
    add_laser_pulse(sim, laser_profile, method='antenna',
                    z0_antenna=0, gamma_boost=gamma_b)

    # Add diagnostic
    if gamma_b is None:
        sim.diags = [
            FieldDiagnostic( Nz, sim.fld, comm=sim.comm )
        ]
    else:
        sim.diags = [
            BackTransformedFieldDiagnostic( zmin_lab=zmin, zmax_lab=zmax,
                dt_snapshots_lab=6.e-14, v_lab=c,
                Ntot_snapshots_lab=2, gamma_boost=gamma_b,
                period=100, fldobject=sim.fld, comm=sim.comm)
        ]

    # Run the simulation
    sim.step( Nz + 1 )

    # Perform test
    compare_simulation_with_theory(data_dir)

    # Remove openPMD and lasy files
    shutil.rmtree(data_dir)
    os.remove('laguerrelaserRZ_00000.h5')
    os.chdir('../')

def test_laser_emission_labframe():
    run_and_check_laser_emission( gamma_b=None,
        data_dir='./diags', lasy_geometry='rt' )

def test_laser_emission_boostedframe():
    run_and_check_laser_emission( gamma_b=10,
        data_dir='./lab_diags', lasy_geometry='xyt' )


if __name__ == "__main__":

    # Emit the laser in the boosted frame
    test_laser_emission_boostedframe()

    # Emit the laser in the lab frame
    test_laser_emission_labframe()