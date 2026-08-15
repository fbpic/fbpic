import numpy as np
from scipy.constants import c, e, m_e, hbar, epsilon_0
import math
from scipy.special import kv
from scipy.integrate import quad

from fbpic.main import Simulation
from fbpic.lpa_utils.bunch import add_particle_bunch_gaussian
from fbpic.lpa_utils.external_fields import ExternalField
from fbpic.openpmd_diag import SynchrotronRadiationDiagnostic
from openpmd_viewer import OpenPMDTimeSeries

# Whether to use the GPU
use_cuda = True
n_order = -1

sig_r = 1e-6
sig_z = 1e-6
gamma0 = 200.0
sig_gamma = 0.0
n_emit = 0
n_macroparticles = 1000
n_physical_particles = 1e-15 / e

# undulator parameters
lambda_u = 20e-6
kz_u = 2 * np.pi / lambda_u
K_u = 9.0

omega_c = 1.5 * gamma0**2 * K_u * kz_u * c
E_u = K_u * kz_u * m_e * c**2 / e
B_u = K_u * kz_u * m_e * c / e
theta_cone = np.sqrt( (K_u/gamma0)**2 + (n_emit/sig_r/gamma0)**2 )

# The simulation box
Nz = 64                # Number of gridpoints along z
zmax =  3.5 * sig_z     # Right end of the simulation box (meters)
zmin = -3.5 * sig_z     # Left end of the simulation box (meters)
dz = (zmax-zmin) / Nz

Nr = 8                  # Number of gridpoints along r
rmax = 4 * sig_r        # Length of the box along r (meters)
Nm = 2                  # Number of modes used

# The interaction length of the simulation (meters)
L_interact = 120.0e-6

# The simulation timestep
dt = (zmax-zmin)/Nz/c   # Timestep (seconds)

theta_max = 2.5 * theta_cone
omega_max = 6 * omega_c

theta_x_axis = -theta_max, theta_max, 128
theta_y_axis = -theta_max, theta_max, 128
photon_energy_axis = 1e3 * e, omega_max * hbar, 256

def field_func_x( F, x, y, z, t, amplitude, length_scale ):
    return( F + amplitude * math.cos( length_scale * z ) )

# The moving window
v_window = c       # Speed of the window

# Interaction time (seconds) (to calculate number of PIC iterations)
T_interact = L_interact / v_window
# (i.e. the time it takes for the moving window to slide across the plasma)

# Theory
P_rad = np.pi * e**2 * c * gamma0**2 * K_u**2 / 3 / epsilon_0 / lambda_u**2
rad_energy_theory = P_rad * T_interact * n_physical_particles

k_53 = lambda x : kv(5./3, x)
S0 = lambda x : 9 * 3**0.5 / 8 / np.pi * x \
                * quad(k_53, x, np.inf)[0]
S0 =  np.vectorize(S0)

def run_simulation():

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
        n_order=n_order, use_cuda=use_cuda,
        boundaries={'z':'open', 'r':'reflective'}
    )

    sim.external_fields = [
        ExternalField( field_func_x, 'By', B_u, kz_u ),
    ]

    bunch = add_particle_bunch_gaussian( sim, -e, m_e, sig_r, sig_z * 0.1,
                                         n_emit, gamma0, sig_gamma,
                                         n_physical_particles,
                                         n_macroparticles,
                                         tf=0.0, zf=0.0,
                                         initialize_self_field=False,)
    # Configure the moving window
    sim.set_moving_window( v=v_window )

    bunch.activate_synchrotron(
        photon_energy_axis,
        theta_x_axis, theta_y_axis,
        #radiation_reaction=True
    )
    bunch.track(sim.comm)

    # Number of iterations to perform
    N_step = int(T_interact/sim.dt)

    # Add diagnostics
    sim.diags = [
                  SynchrotronRadiationDiagnostic(period=N_step,
                    species={'bunch': bunch},
                    comm=sim.comm)
                ]

    ### Run the simulation
    sim.step( N_step + 1)

def check_energy():
    ts = OpenPMDTimeSeries('./diags/hdf5/')
    radiation_fbpic, info = ts.get_field('radiation_bunch', t=ts.t[-1], slice_across=None)
    rad_energy = radiation_fbpic.sum() * info.dx * info.dy * info.dz
    err = np.abs( rad_energy - rad_energy_theory ) / rad_energy
    assert err<0.06

def check_angle():
    ts = OpenPMDTimeSeries('./diags/hdf5/')
    radiation_fbpic, info = ts.get_field('radiation_bunch', t=ts.t[-1], slice_across=None)
    spot = radiation_fbpic.sum(-1)
    thx = info.x

    theta_x = np.sqrt(
        np.average(thx**2, weights=spot.sum(1)) \
        - np.average(thx, weights=spot.sum(1))**2
    )
    theta_x_theory = 0.5 * K_u / gamma0
    err = np.abs(theta_x - theta_x_theory) / theta_x_theory
    assert err<7e-3

def check_spectrum():
    ts = OpenPMDTimeSeries('./diags/hdf5/')
    radiation_fbpic, info = ts.get_field('radiation_bunch', t=ts.t[-1], slice_across=None)
    spect_1d = radiation_fbpic[
        radiation_fbpic.shape[0]//2, radiation_fbpic.shape[1]//2
    ]

    spect_1d_theory = S0(info.z / hbar / omega_c)

    spect_1d = spect_1d / spect_1d.max()
    spect_1d_theory = spect_1d_theory / spect_1d_theory.max()

    assert np.allclose(spect_1d, spect_1d_theory, atol=0, rtol=1e-2)

def test_synchrotron():
    run_simulation()
    check_energy()
    check_spectrum()
    check_angle()

if __name__ == '__main__':
    test_synchrotron()
