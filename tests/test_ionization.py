# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It reproduces the test from Chen, JCP, 2013, figure 2:
A Gaussian laser pulse propagates through a gas of Nitrogen atoms. For the
intensity and duration of this laser, about 1/3 of the atoms stay in the
N5+ state, after the laser has passed through the gas.

This test simulates the interaction and checks that the final number N5+ ions
is indeed close to 1/3.

Because the original test from Chen (2013) is essentially 1D, in the FBPIC
simulation, the number of radial cells is very low, and the laser is produced
through external fields.
"""

# -------
# Imports
# -------
import math
import numpy as np
from scipy.constants import c, m_e, m_p, e
# Import the relevant structures in FBPIC
from fbpic.main import Simulation, adapt_to_grid
from fbpic.particles import Particles
from fbpic.lpa_utils.external_fields import ExternalField
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic, \
                  BoostedFieldDiagnostic, BoostedParticleDiagnostic
# ----------
# Parameters
# ----------
use_cuda = True
output_hdf5_files = False

def run_simulation( gamma_boost ):
    """
    Run a simulation with a laser pulse going through a gas jet of ionizable
    N5+ atoms, and check the fraction of atoms that are in the N5+ state.

    Parameters
    ----------
    gamma_boost: float
        The Lorentz factor of the frame in which the simulation is carried out.
    """
    # The simulation box
    Nz = 400         # Number of gridpoints along z
    zmax = 20.e-6    # Length of the box along z (meters)
    zmin = 0.e-6
    Nr = 3           # Number of gridpoints along r
    rmax = 10.e-6    # Length of the box along r (meters)
    Nm = 2           # Number of modes used
    n_guard = 40     # Number of guard cells
    exchange_period = 10
    # The simulation timestep
    dt = (zmax-zmin)/Nz/c   # Timestep (seconds)
    N_step = 1601    # Number of iterations to perform

    # Boosted frame
    boost = BoostConverter(gamma_boost)

    # The laser
    a0 = 1.8         # Laser amplitude
    ctau = 8.e-6     # Laser duration
    z0 = -16.e-6     # Laser centroid
    lambda0 = 0.8e-6 # Laser wavelength
    omega = 2*np.pi*c/lambda0
    E0 = a0 * m_e*c*omega/e
    B0 = E0/c
    def laser_func( F, x, y, z, t, amplitude, length_scale ):
        """
        Function that describes a Gaussian laser with infinite waist
        """
        return( F + amplitude * math.cos( 2*np.pi*(z-c*t)/lambda0 ) * \
                math.exp( - (z - c*t - z0)**2/ctau**2 ) )

    # The particles of the plasma
    p_zmin = 5.e-6   # Position of the beginning of the plasma (meters)
    p_zmax = 15.e-6
    p_rmin = 0.      # Minimal radial position of the plasma (meters)
    p_rmax = 100.e-6 # Maximal radial position of the plasma (meters)
    n_e = 1.         # The density in the labframe (electrons.meters^-3)
    p_nz = 2         # Number of particles per cell along z
    p_nr = 1         # Number of particles per cell along r
    p_nt = 4         # Number of particles per cell along theta

    # Get the speed of the plasma
    v_plasma, = boost.velocity( [ 0. ] )

    # The diagnostics
    diag_period = 200 # Period of the diagnostics in number of timesteps
    # Whether to write the fields in the lab frame
    Ntot_snapshot_lab = 20
    dt_snapshot_lab = (zmax-zmin)/c

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
        p_zmax, p_zmax, # No electrons get created because we pass p_zmin=p_zmax
        p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
        zmin=zmin, initialize_ions=False,
        v_comoving=v_plasma, use_galilean=False,
        n_guard=n_guard, exchange_period=exchange_period,
        gamma_boost=gamma_boost, boundaries='open', use_cuda=use_cuda )

    # Add the N atoms
    p_zmin, p_zmax, Npz = adapt_to_grid( sim.fld.interp[0].z,
                                         p_zmin, p_zmax, p_nz )
    p_rmin, p_rmax, Npr = adapt_to_grid( sim.fld.interp[0].r,
                                         p_rmin, p_rmax, p_nr )
    sim.ptcl.append(
        Particles(q=e, m=14.*m_p, n=0.2*n_e, Npz=Npz, zmin=p_zmin,
                  zmax=p_zmax, Npr=Npr, rmin=p_rmin, rmax=p_rmax,
                  Nptheta=p_nt, dt=dt, use_cuda=use_cuda, uz_m=0,
                  grid_shape=sim.fld.interp[0].Ez.shape ) )
    sim.ptcl[1].make_ionizable(element='N', level_start=0,
                               target_species=sim.ptcl[0])

    # Add a laser to the fields of the simulation (external fields)
    sim.external_fields = [
        ExternalField( laser_func, 'Ex', E0, 0. ),
        ExternalField( laser_func, 'By', B0, 0. ) ]

    # Add a field diagnostic
    if output_hdf5_files:
        if gamma_boost == 1:
            sim.diags = [ FieldDiagnostic(diag_period, sim.fld, sim.comm ),
                        ParticleDiagnostic(diag_period,
                        {"ions":sim.ptcl[1]}, sim.comm) ]
        else:
            sim.diags = [
                     BoostedFieldDiagnostic( zmin, zmax, c,
                        dt_snapshot_lab, Ntot_snapshot_lab, gamma_boost,
                        period=diag_period, fldobject=sim.fld, comm=sim.comm),
                    BoostedParticleDiagnostic( zmin, zmax, c, dt_snapshot_lab,
                        Ntot_snapshot_lab, gamma_boost, diag_period, sim.fld,
                        species={'ions':sim.ptcl[1]}, comm=sim.comm ) ]

    # Run the simulation
    sim.step( N_step, use_true_rho=True )

    # Check the fraction of N5+ ions at the end of the simulation
    w_neutral = sim.ptcl[1].ionizer.neutral_weight
    ioniz_level = sim.ptcl[1].ionizer.ionization_level
    # Get the total number of N atoms/ions (all ionization levels together)
    ntot = w_neutral.sum()
    # Get the total number of N5+ ions
    n_N5 = w_neutral[ioniz_level == 5].sum()
    # Get the fraction of N5+ ions, and check that it is close to 0.32
    N5_fraction = n_N5 / ntot
    print(N5_fraction)
    assert ((N5_fraction > 0.30) and (N5_fraction < 0.34))

def test_ionization_labframe():
    run_simulation(1.)

# Run the tests
if __name__ == '__main__':
    test_ionization_labframe()
