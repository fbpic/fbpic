# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It tests the Compton scattering module, in a configuration where a
relativistic electron bunch crosses a Gaussian laser pulse.
This test is performed in the lab frame and in the boosted frame.

Note: This test does not exercise the full loop: only the
Compton scattering and particle pusher are used.
"""

# -------
# Imports
# -------
import numpy as np
from scipy.constants import e, c, h, m_e, epsilon_0
# Import the relevant structures in FBPIC
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.main import Simulation
from fbpic.particles import Particles
from fbpic.lpa_utils.bunch import add_elec_bunch_gaussian

# ----------
# Parameters
# ----------
use_cuda = True

def run_simulation( gamma_boost ):
    """
    Run a simulation with a relativistic electron bunch crosses a laser

    Parameters
    ----------
    gamma_boost: float
        The Lorentz factor of the frame in which the simulation is carried out.
    """
    # Boosted frame
    boost = BoostConverter(gamma_boost)

    # The simulation box (in the lab frame)
    Nz = 200         # Number of gridpoints along z
    zmax_lab = 20.e-6    # Right end of the simulation box (meters)
    zmin_lab = -20.e-6   # Left end of the simulation box (meters)
    Nr = 50          # Number of gridpoints along r
    rmax = 20.e-6    # Length of the box along r (meters)
    Nm = 2           # Number of modes used

    # The electron bunch (in the lab frame)
    Q_bunch = 2080.5031144200598 * 30000 * e
    N_bunch = 300000   # Number of macroparticles
    gamma_bunch_mean = 30.205798028084185
    gamma_bunch_rms = 0. #0.58182474907848347
    bunch_sigma_z = 1.e-6

    # The scattering laser (in the lab frame)
    laser_energy = 1. # Joule
    laser_radius = 33.e-6 # meters
    laser_duration = 2.e-12 # seconds
    laser_waist = laser_radius * (2.)**.5
    laser_ctau = c*laser_duration
    laser_wavelength = h*c/e # Corresponds to 1 eV photons
    laser_initial_z0 = c*4*laser_duration # meters

    # The simulation timestep
    N_step = 101     # Number of iterations to perform
    # Calculate timestep to resolve the interaction with enough points
    laser_duration_boosted, = boost.copropag_length(
        [laser_duration], beta_object=-1 )
    bunch_sigma_z_boosted, = boost.copropag_length(
        [bunch_sigma_z], beta_object=1 )
    dt = (4*laser_duration_boosted + bunch_sigma_z_boosted/c)/N_step

    # Initialize the simulation object
    zmax, zmin = boost.copropag_length( [zmax_lab, zmin_lab], beta_object=1. )
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
        p_zmin=0, p_zmax=0, p_rmin=0, p_rmax=0,
        p_nz=1, p_nr=1, p_nt=1, n_e=1,
        dens_func=None, zmin=zmin, boundaries='periodic',
        use_cuda=use_cuda )
    # Remove particles that were previously created
    sim.ptcl = []
    print( 'Initialized simulation' )

    # Add electron bunch (automatically converted to boosted-frame)
    add_elec_bunch_gaussian( sim, sig_r=1.e-6, sig_z=bunch_sigma_z,
        n_emit=0., gamma0=gamma_bunch_mean, sig_gamma=gamma_bunch_rms,
        Q=Q_bunch, N=N_bunch, tf=0.0, zf=0.5*(zmax+zmin), boost=boost )
    print( 'Initialized electron bunch' )
    # Add a photon species
    photons = Particles( q=0, m=0, n=0, Npz=1, zmin=0, zmax=0,
                    Npr=1, rmin=0, rmax=0, Nptheta=1, dt=sim.dt,
                    ux_m=0., uy_m=0., uz_m=0.,
                    ux_th=0., uy_th=0., uz_th=0.,
                    dens_func=None, continuous_injection=False,
                    grid_shape=None, particle_shape='linear',
                    use_cuda=sim.use_cuda)
    sim.ptcl.append( photons )
    print( 'Initialized photons' )

    # Activate Compton scattering for electrons of the bunch
    sim.ptcl[0].activate_compton( target_species=photons,
        laser_energy=laser_energy, laser_wavelength=laser_wavelength,
        laser_waist=laser_waist, laser_ctau=laser_ctau,
        laser_initial_z0=laser_initial_z0, boost=boost )
    print( 'Activated Compton' )

    ### Run the simulation
    for i_step in range( N_step ):
        sim.ptcl[0].halfpush_x()
        sim.ptcl[0].handle_elementary_processes( sim.time + 0.5*sim.dt )
        sim.ptcl[0].halfpush_x()
        sim.time += sim.dt
        sim.iteration += 1
        # Print fraction of photons produced
        if i_step%10 == 0:
            simulated_frac = sim.ptcl[1].Ntot/sim.ptcl[0].Ntot
            print( 'Iteration %d: Photon fraction per electron = %f' \
                       %(i_step, simulated_frac) )

    # Calculate the expected photon fraction
    # - Total Klein-Nishina cross section in electron rest frame:
    beta_bunch_mean = np.sqrt(1-1./gamma_bunch_mean**2)
    photon_p_rest = gamma_bunch_mean*(1+beta_bunch_mean)*h/laser_wavelength
    k = photon_p_rest / (m_e*c)
    # For low k, the Klein-Nishina cross-section is essentially the
    # Compton cross-section
    assert (k<1.e-3)
    r_e = 1./(4*np.pi*epsilon_0) * e**2/(m_e*c**2)
    sigma = 8./3 * np.pi*r_e**2
    # - Total number of photons that go through this cross-section
    energy_per_surface = laser_energy / ( np.pi/2*laser_waist**2 )
    nphoton_per_surface = energy_per_surface / ( h*c/laser_wavelength )
    expected_frac = sigma * nphoton_per_surface

    # Automatically check that the obtained fraction is within 10%
    assert abs(simulated_frac-expected_frac) < 0.1*expected_frac
    print( 'Test passed.' )

def test_compton_labframe():
    print('\nTest Compton scattering in lab frame')
    print('------------------------------------')
    run_simulation(1.)

def test_compton_boostedframe():
    print('\nTest Compton scattering in boosted frame')
    print('----------------------------------------')
    run_simulation(16.6)

# Run the tests
if __name__ == '__main__':
    test_compton_labframe()
    test_compton_boostedframe()
