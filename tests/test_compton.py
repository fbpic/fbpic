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
from fbpic.lpa_utils.bunch import add_elec_bunch_gaussian
from fbpic.openpmd_diag import ParticleDiagnostic

# ----------------------
# Parameters of the test
# ----------------------
use_cuda = True

write_hdf5 = False
show_plots = True

# -------------------------------
# Parameters of the configuration
# -------------------------------


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
gamma_bunch_rms = 0.58182474907848347
bunch_sigma_z = 1.e-6

# The scattering laser (in the lab frame)
laser_energy = 1. # Joule
laser_radius = 33.e-6 # meters
laser_duration = 2.e-12 # seconds
laser_waist = laser_radius * (2.)**.5
laser_ctau = c*laser_duration
laser_wavelength = h*c/e # Corresponds to 1 eV photons
laser_initial_z0 = c*4*laser_duration # meters

def run_simulation( gamma_boost, show ):
    """
    Run a simulation with a relativistic electron bunch crosses a laser

    Parameters
    ----------
    gamma_boost: float
        The Lorentz factor of the frame in which the simulation is carried out.
    show: bool
        Whether to show a plot of the angular distribution
    """
    # Boosted frame
    boost = BoostConverter(gamma_boost)

    # The simulation timestep
    diag_period = 100
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
        dens_func=None, zmin=zmin, boundaries='periodic',
        use_cuda=use_cuda )
    # Remove particles that were previously created
    sim.ptcl = []
    print( 'Initialized simulation' )

    # Add electron bunch (automatically converted to boosted-frame)
    add_elec_bunch_gaussian( sim, sig_r=1.e-6, sig_z=bunch_sigma_z,
        n_emit=0., gamma0=gamma_bunch_mean, sig_gamma=gamma_bunch_rms,
        Q=Q_bunch, N=N_bunch, tf=0.0, zf=0.5*(zmax+zmin), boost=boost )
    elec = sim.ptcl[0]
    print( 'Initialized electron bunch' )
    # Add a photon species
    photons = sim.add_new_species( q=0, m=0 )
    print( 'Initialized photons' )

    # Activate Compton scattering for electrons of the bunch
    elec.activate_compton( target_species=photons,
        laser_energy=laser_energy, laser_wavelength=laser_wavelength,
        laser_waist=laser_waist, laser_ctau=laser_ctau,
        laser_initial_z0=laser_initial_z0, ratio_w_electron_photon=50,
        boost=boost )
    print( 'Activated Compton' )

    # Add diagnostics
    if write_hdf5:
        sim.diags = [ ParticleDiagnostic( diag_period,
            species={'electrons': elec, 'photons': photons}, comm=sim.comm ) ]

    # Get initial total momentum
    initial_total_elec_px = (elec.w*elec.ux).sum() * m_e * c
    initial_total_elec_py = (elec.w*elec.uy).sum() * m_e * c
    initial_total_elec_pz = (elec.w*elec.uz).sum() * m_e * c

    ### Run the simulation
    for species in sim.ptcl:
        species.send_particles_to_gpu()

    for i_step in range( N_step ):
        for species in sim.ptcl:
            species.push_x( 0.5*sim.dt )
        elec.handle_elementary_processes( sim.time + 0.5*sim.dt )
        for species in sim.ptcl:
            species.push_x( 0.5*sim.dt )
        # Increment time and run diagnostics
        sim.time += sim.dt
        sim.iteration += 1
        for diag in sim.diags:
            diag.write( sim.iteration )
        # Print fraction of photons produced
        if i_step%10 == 0:
            for species in sim.ptcl:
                species.receive_particles_from_gpu()
            simulated_frac = photons.w.sum()/elec.w.sum()
            for species in sim.ptcl:
                species.send_particles_to_gpu()
            print( 'Iteration %d: Photon fraction per electron = %f' \
                       %(i_step, simulated_frac) )

    for species in sim.ptcl:
        species.receive_particles_from_gpu()


    # Check estimation of photon fraction
    check_photon_fraction( simulated_frac )
    # Check conservation of momentum (is only conserved )
    if elec.compton_scatterer.ratio_w_electron_photon == 1:
        check_momentum_conservation( gamma_boost, photons, elec,
          initial_total_elec_px, initial_total_elec_py, initial_total_elec_pz )

    # Transform the photon momenta back into the lab frame
    photon_u = 1./photons.inv_gamma
    photon_lab_pz = boost.gamma0*( photons.uz + boost.beta0*photon_u )
    photon_lab_p = boost.gamma0*( photon_u + boost.beta0*photons.uz )

    # Plot the scaled angle and frequency
    if show:
        import matplotlib.pyplot as plt
        # Bin the photons on a grid in frequency and angle
        freq_min = 0.5
        freq_max = 1.2
        N_freq = 500
        gammatheta_min = 0.
        gammatheta_max = 1.
        N_gammatheta = 100
        hist_range = [[freq_min, freq_max], [gammatheta_min, gammatheta_max]]
        extent = [freq_min, freq_max, gammatheta_min, gammatheta_max]
        fundamental_frequency = 4*gamma_bunch_mean**2*c/laser_wavelength
        photon_scaled_freq = photon_lab_p*c / (h*fundamental_frequency)
        gamma_theta = gamma_bunch_mean * np.arccos(photon_lab_pz/photon_lab_p)
        grid, freq_bins, gammatheta_bins = np.histogram2d(
            photon_scaled_freq, gamma_theta, weights=photons.w,
            range=hist_range, bins=[ N_freq, N_gammatheta ] )
        # Normalize by solid angle, frequency and number of photons
        dw = (freq_bins[1]-freq_bins[0]) * 2*np.pi * fundamental_frequency
        dtheta = ( gammatheta_bins[1]-gammatheta_bins[0] )/gamma_bunch_mean
        domega = 2.*np.pi*np.sin( gammatheta_bins/gamma_bunch_mean )*dtheta
        grid /= dw * domega[np.newaxis, 1:] * elec.w.sum()
        grid = np.where( grid==0, np.nan, grid )
        plt.imshow( grid.T, origin='lower', extent=extent,
                    cmap='gist_earth', aspect='auto', vmax=1.8e-16 )
        plt.title('Particles, $d^2N/d\omega \,d\Omega$')
        plt.xlabel('Scaled energy ($\omega/4\gamma^2\omega_\ell$)')
        plt.ylabel(r'$\gamma \theta$' )
        plt.colorbar()
        # Plot theory
        plt.plot( 1./( 1 + gammatheta_bins**2), gammatheta_bins, color='r' )
        plt.show()
        plt.clf()


def check_momentum_conservation( gamma_boost, photons, elec,
                                 px_init, py_init, pz_init ):
    """Check conservation of momentum, i.e. that the current momentum of
    the electrons + the change in momentum of the photons is equal
    to the initial momentum of the electrons"""

    elec_px = (elec.w*elec.ux).sum() * m_e * c
    elec_py = (elec.w*elec.uy).sum() * m_e * c
    elec_pz = (elec.w*elec.uz).sum() * m_e * c

    photons_delta_px = (photons.w*photons.ux).sum()
    photons_delta_py = (photons.w*photons.uy).sum()
    photons_delta_pz = \
        (photons.w*(photons.uz+2*gamma_boost*h/laser_wavelength)).sum()

    atol = 1.e-9*abs(pz_init)
    assert np.allclose( px_init, elec_px + photons_delta_px, atol=atol )
    assert np.allclose( py_init, elec_py + photons_delta_py, atol=atol )
    assert np.allclose( pz_init, elec_pz + photons_delta_pz, atol=atol )


def check_photon_fraction( simulated_frac ):
    """Check that the photon fraction is close (within 10%) to
    the esimate, based on the Klein-Nishina formula"""
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



def test_compton_labframe( show=False ):
    print('\nTest Compton scattering in lab frame')
    print('------------------------------------')
    run_simulation(gamma_boost=1., show=show)

def test_compton_boostedframe( show=False ):
    print('\nTest Compton scattering in boosted frame')
    print('----------------------------------------')
    run_simulation(gamma_boost=16.6, show=show)

# Run the tests
if __name__ == '__main__':
    test_compton_labframe( show=show_plots )
    test_compton_boostedframe( show=show_plots )
