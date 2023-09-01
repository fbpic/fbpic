import numpy as np
from scipy.constants import c, e, m_e, hbar, epsilon_0
import math
# Import the relevant structures in FBPIC

from fbpic.main import Simulation
from fbpic.lpa_utils.bunch import add_particle_bunch_gaussian
from fbpic.lpa_utils.external_fields import ExternalField
from fbpic.openpmd_diag import SRDiagnostic
from openpmd_viewer import OpenPMDTimeSeries

# Whether to use the GPU
use_cuda = True
n_order = -1

sig_r = 1e-6
sig_z = 1e-6
gamma0 = 200.0
sig_gamma = 0.0
n_emit = 1.5e-6
n_macroparticles = 2000
n_physical_particles = 100e-12 / e

# The simulation box
Nz = 128              # Number of gridpoints along z
zmax = 3 * sig_z     # Right end of the simulation box (meters)
zmin = -3 * sig_z    # Left end of the simulation box (meters)
Nr = 160             # Number of gridpoints along r
rmax = 8 * sig_r    # Length of the box along r (meters)
Nm = 2               # Number of modes used

# The simulation timestep
dt = (zmax-zmin)/Nz/c   # Timestep (seconds)

# The interaction length of the simulation (meters)
L_interact = 120.0e-6

# undulator parameters
lambda_u = 20e-6
kz_u = 2 * np.pi / lambda_u
K_u = 8.

omega_c = 1.5 * gamma0**2 * K_u * kz_u * c
E_u = K_u * kz_u * m_e * c**2 / e
theta_cone = np.sqrt( (K_u/gamma0)**2 + (n_emit/sig_r/gamma0)**2 )

P_rad = np.pi * e**2 * c * gamma0**2 * K_u**2 / 3 / epsilon_0 / lambda_u**2

theta_max = 2.5 * theta_cone
omega_max = 6 * omega_c

theta_x_axis = -theta_max, theta_max, 128
theta_y_axis = -theta_max, theta_max, 128
photon_energy_axis = 1 * e, omega_max * hbar, 256

def field_func( F, x, y, z, t, amplitude, length_scale ):
    return( F + E_u * math.cos( kz_u * z ) )

# The moving window
v_window = c       # Speed of the window

# Interaction time (seconds) (to calculate number of PIC iterations)
T_interact = ( L_interact + (zmax-zmin) ) / v_window
# (i.e. the time it takes for the moving window to slide across the plasma)

rad_energy_theory = P_rad * T_interact * n_physical_particles

def run_simulation():

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
        n_order=n_order, use_cuda=use_cuda,
        boundaries={'z':'open', 'r':'reflective'})

    sim.external_fields = [ ExternalField( field_func, 'Ex', E_u, lambda_u ) ]

    bunch = add_particle_bunch_gaussian( sim, -e, m_e, sig_r, sig_z,
                                         n_emit, gamma0, sig_gamma,
                                         n_physical_particles,
                                         n_macroparticles,
                                         tf=0.0, zf=0.0,
                                         initialize_self_field=False,)
    # Configure the moving window
    sim.set_moving_window( v=v_window )
    bunch.activate_synchrotron(photon_energy_axis, theta_x_axis, theta_y_axis)

    # Number of iterations to perform
    N_step = int(T_interact/sim.dt)

    # Add diagnostics
    sim.diags = [
                  SRDiagnostic(period=N_step,
                    sr_object=bunch.synchrotron_radiator,
                    comm=sim.comm)
                ]

    ### Run the simulation
    sim.step( N_step + 1)

def get_radiated_energy():
    ts = OpenPMDTimeSeries(f'./diags/hdf5/')
    radiation_fbpic, info = ts.get_field('radiation', t=ts.t[-1], slice_across=None)
    rad_energy = radiation_fbpic.sum() * info.dx * info.dy * info.dz
    return rad_energy

def test_synchrotron():
    run_simulation()
    rad_energy = get_radiated_energy()
    err = np.abs( rad_energy - rad_energy_theory ) / rad_energy
    assert err<0.06

if __name__ == '__main__':
    test_synchrotron()
