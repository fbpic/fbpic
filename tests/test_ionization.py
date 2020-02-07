# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It reproduces the test from Chen, JCP, 2013, figure 2.
A Gaussian laser pulse propagates through a gas of Nitrogen atoms. For the
intensity and duration of this laser, about 1/3 of the atoms remain in the
N5+ state, after the laser has passed through the gas.

This test simulates the interaction and checks that the final number N5+ ions
is indeed close to 1/3. (Due to the low number of macroparticles and the
corresponding statistical fluctuations in this test, it is difficult to have
a precise number, but the ionization fraction is checked at the 10% level.)

Because the original test from Chen (2013) is essentially 1D, in the FBPIC
simulation, the number of radial cells is very low, and the laser is produced
through external fields.
"""

# -------
# Imports
# -------
import shutil, math
import numpy as np
from scipy.constants import c, m_e, m_p, e
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.external_fields import ExternalField
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.openpmd_diag import ParticleDiagnostic, \
    BackTransformedParticleDiagnostic
# Import openPMD-viewer for checking output files
from openpmd_viewer import OpenPMDTimeSeries

# ----------
# Parameters
# ----------
use_cuda = True

def run_simulation( gamma_boost, use_separate_electron_species ):
    """
    Run a simulation with a laser pulse going through a gas jet of ionizable
    N5+ atoms, and check the fraction of atoms that are in the N5+ state.

    Parameters
    ----------
    gamma_boost: float
        The Lorentz factor of the frame in which the simulation is carried out.
    use_separate_electron_species: bool
        Whether to use separate electron species for each level, or
        a single electron species for all levels.
    """
    # The simulation box
    zmax_lab = 20.e-6    # Length of the box along z (meters)
    zmin_lab = 0.e-6
    Nr = 3           # Number of gridpoints along r
    rmax = 10.e-6    # Length of the box along r (meters)
    Nm = 2           # Number of modes used

    # The particles of the plasma
    p_zmin = 5.e-6   # Position of the beginning of the plasma (meters)
    p_zmax = 15.e-6
    p_rmin = 0.      # Minimal radial position of the plasma (meters)
    p_rmax = 100.e-6 # Maximal radial position of the plasma (meters)
    n_atoms = 0.2    # The atomic density is chosen very low,
                     # to avoid collective effects
    p_nz = 2         # Number of particles per cell along z
    p_nr = 1         # Number of particles per cell along r
    p_nt = 4         # Number of particles per cell along theta

    # Boosted frame
    boost = BoostConverter(gamma_boost)
    # Boost the different quantities
    beta_boost = np.sqrt( 1. - 1./gamma_boost**2 )
    zmin, zmax = boost.static_length( [zmin_lab, zmax_lab] )
    p_zmin, p_zmax = boost.static_length( [p_zmin, p_zmax] )
    n_atoms, = boost.static_density( [n_atoms] )
    # Increase the number of particles per cell in order to keep sufficient
    # statistics for the evaluation of the ionization fraction
    if gamma_boost > 1:
        p_nz = int( 2 * gamma_boost * (1+beta_boost) * p_nz )

    # The laser
    a0 = 1.8         # Laser amplitude
    lambda0_lab = 0.8e-6 # Laser wavelength
    # Boost the laser wavelength before calculating the laser amplitude
    lambda0, = boost.copropag_length( [ lambda0_lab ], beta_object=1. )
    # Duration and initial position of the laser
    ctau = 10.*lambda0
    z0 = -2*ctau
    # Calculate laser amplitude
    omega = 2*np.pi*c/lambda0
    E0 = a0 * m_e*c*omega/e
    B0 = E0/c
    def laser_func( F, x, y, z, t, amplitude, length_scale ):
        """
        Function that describes a Gaussian laser with infinite waist
        """
        return( F + amplitude * math.cos( 2*np.pi*(z-c*t)/lambda0 ) * \
                math.exp( - (z - c*t - z0)**2/ctau**2 ) )

    # Resolution and number of timesteps
    dz = lambda0/16.
    dt = dz/c
    Nz = int( (zmax-zmin)/dz ) + 1
    N_step = int( (2.*40.*lambda0 + zmax-zmin)/(dz*(1+beta_boost)) ) + 1

    # Get the speed of the plasma
    uz_m, = boost.longitudinal_momentum( [ 0. ] )
    v_plasma, = boost.velocity( [ 0. ] )

    # The diagnostics
    diag_period = N_step-1 # Period of the diagnostics in number of timesteps

    # Initial ionization level of the Nitrogen atoms
    level_start = 2
    # Initialize the simulation object, with the neutralizing electrons
    # No particles are created because we do not pass the density
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
        v_comoving=v_plasma, use_galilean=False,
        boundaries='open', use_cuda=use_cuda )

    # Add the charge-neutralizing electrons
    elec = sim.add_new_species( q=-e, m=m_e, n=level_start*n_atoms,
                        p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
                        p_zmin=p_zmin, p_zmax=p_zmax,
                        p_rmin=p_rmin, p_rmax=p_rmax,
                        continuous_injection=False, uz_m=uz_m )
    # Add the N atoms
    ions = sim.add_new_species( q=0, m=14.*m_p, n=n_atoms,
                        p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
                        p_zmin=p_zmin, p_zmax=p_zmax,
                        p_rmin=p_rmin, p_rmax=p_rmax,
                        continuous_injection=False, uz_m=uz_m )
    # Add the target electrons
    if use_separate_electron_species:
        # Use a dictionary of electron species: one per ionizable level
        target_species = {}
        level_max = 6 # N can go up to N7+, but here we stop at N6+
        for i_level in range(level_start, level_max):
            target_species[i_level] = sim.add_new_species( q=-e, m=m_e )
    else:
        # Use the pre-existing, charge-neutralizing electrons
        target_species = elec
        level_max = None # Default is going up to N7+
    # Define ionization
    ions.make_ionizable( element='N', level_start=level_start,
                         level_max=level_max, target_species=target_species )
    # Set the moving window
    sim.set_moving_window( v=v_plasma )

    # Add a laser to the fields of the simulation (external fields)
    sim.external_fields = [
        ExternalField( laser_func, 'Ex', E0, 0. ),
        ExternalField( laser_func, 'By', B0, 0. ) ]

    # Add a particle diagnostic
    sim.diags = [ ParticleDiagnostic( diag_period, {"ions":ions},
        particle_data=["position", "gamma", "weighting", "E", "B"],
        # Test output of fields and gamma for standard
        # (non-boosted) particle diagnostics
        write_dir='tests/diags', comm=sim.comm) ]
    if gamma_boost > 1:
        T_sim_lab = (2.*40.*lambda0_lab + zmax_lab-zmin_lab)/c
        sim.diags.append(
            BackTransformedParticleDiagnostic(zmin_lab, zmax_lab, v_lab=0.,
                dt_snapshots_lab=T_sim_lab/2., Ntot_snapshots_lab=3,
                gamma_boost=gamma_boost, period=diag_period,
                fldobject=sim.fld, species={"ions":ions},
                comm=sim.comm, write_dir='tests/lab_diags') )

    # Run the simulation
    sim.step( N_step, use_true_rho=True )

    # Check the fraction of N5+ ions at the end of the simulation
    w = ions.w
    ioniz_level = ions.ionizer.ionization_level
    # Get the total number of N atoms/ions (all ionization levels together)
    ntot = w.sum()
    # Get the total number of N5+ ions
    n_N5 = w[ioniz_level == 5].sum()
    # Get the fraction of N5+ ions, and check that it is close to 0.32
    N5_fraction = n_N5 / ntot
    print('N5+ fraction: %.4f' %N5_fraction)
    assert ((N5_fraction > 0.30) and (N5_fraction < 0.34))

    # When different electron species are created, check the fraction of
    # each electron species
    if use_separate_electron_species:
        for i_level in range(level_start, level_max):
            n_N = w[ioniz_level == i_level].sum()
            assert np.allclose( target_species[i_level].w.sum(), n_N )

    # Check consistency in the regular openPMD diagnostics
    ts = OpenPMDTimeSeries('./tests/diags/hdf5/')
    last_iteration = ts.iterations[-1]
    w, q = ts.get_particle( ['w', 'charge'], species="ions",
                                iteration=last_iteration )
    # Check that the openPMD file contains the same number of N5+ ions
    n_N5_openpmd = np.sum(w[ (4.5*e < q) & (q < 5.5*e) ])
    assert np.isclose( n_N5_openpmd, n_N5 )
    # Remove openPMD files
    shutil.rmtree('./tests/diags/')

    # Check consistency of the back-transformed openPMD diagnostics
    if gamma_boost > 1.:
        ts = OpenPMDTimeSeries('./tests/lab_diags/hdf5/')
        last_iteration = ts.iterations[-1]
        w, q = ts.get_particle( ['w', 'charge'], species="ions",
                                iteration=last_iteration )
        # Check that the openPMD file contains the same number of N5+ ions
        n_N5_openpmd = np.sum(w[ (4.5*e < q) & (q < 5.5*e) ])
        assert np.isclose( n_N5_openpmd, n_N5 )
        # Remove openPMD files
        shutil.rmtree('./tests/lab_diags/')

def test_ionization_labframe():
    run_simulation(1., use_separate_electron_species=True)

def test_ionization_boostedframe():
    run_simulation(2., use_separate_electron_species=False)

# Run the tests
if __name__ == '__main__':
    test_ionization_labframe()
    test_ionization_boostedframe()
