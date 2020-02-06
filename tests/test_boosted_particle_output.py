# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It tests the boosted-frame particle output routines.
This is done by initializing a set of known particles and making sure that
they are all retrieved by the boosted-frame diagnostics.
"""

# -------
# Imports
# -------
import os, shutil
import numpy as np
from scipy.constants import c
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.lpa_utils.bunch import add_elec_bunch_gaussian
from fbpic.openpmd_diag import BackTransformedParticleDiagnostic
# Import openPMD-viewer for checking output files
from openpmd_viewer import OpenPMDTimeSeries

# ----------
# Parameters
# ----------
use_cuda = True

def test_boosted_output( gamma_boost=10. ):
    """
    # TODO

    Parameters
    ----------
    gamma_boost: float
        The Lorentz factor of the frame in which the simulation is carried out.
    """
    # The simulation box
    Nz = 500         # Number of gridpoints along z
    zmax_lab = 0.e-6    # Length of the box along z (meters)
    zmin_lab = -20.e-6
    Nr = 10          # Number of gridpoints along r
    rmax = 10.e-6    # Length of the box along r (meters)
    Nm = 2           # Number of modes used

    # Number of timesteps
    N_steps = 500
    diag_period = 20 # Period of the diagnostics in number of timesteps
    dt_lab = (zmax_lab - zmin_lab)/Nz * 1./c
    T_sim_lab = N_steps * dt_lab

    # Move into directory `tests`
    os.chdir('./tests')

    # Initialize the simulation object
    sim = Simulation( Nz, zmax_lab, Nr, rmax, Nm, dt_lab,
        0, 0, # No electrons get created because we pass p_zmin=p_zmax=0
        0, rmax, 1, 1, 4,
        n_e=0, zmin=zmin_lab, initialize_ions=False, gamma_boost=gamma_boost,
        v_comoving=-0.9999*c, boundaries='open', use_cuda=use_cuda )
    sim.set_moving_window( v=c )
    # Remove the electron species
    sim.ptcl = []

    # Add a Gaussian electron bunch
    # Note: the total charge is 0 so all fields should remain 0
    # throughout the simulation. As a consequence, the motion of the beam
    # is a mere translation.
    N_particles = 3000
    add_elec_bunch_gaussian( sim, sig_r=1.e-6, sig_z=1.e-6, n_emit=0.,
        gamma0=100, sig_gamma=0., Q=0., N=N_particles,
        zf=0.5*(zmax_lab+zmin_lab), boost=BoostConverter(gamma_boost) )
    sim.ptcl[0].track( sim.comm )

    # openPMD diagnostics
    sim.diags = [
        BackTransformedParticleDiagnostic( zmin_lab, zmax_lab, v_lab=c,
            dt_snapshots_lab=T_sim_lab/3., Ntot_snapshots_lab=3,
            gamma_boost=gamma_boost, period=diag_period, fldobject=sim.fld,
            species={"bunch": sim.ptcl[0]}, comm=sim.comm) ]

    # Run the simulation
    sim.step( N_steps )

    # Check consistency of the back-transformed openPMD diagnostics:
    # Make sure that all the particles were retrived by checking particle IDs
    ts = OpenPMDTimeSeries('./lab_diags/hdf5/')
    ref_pid = np.sort( sim.ptcl[0].tracker.id )
    for iteration in ts.iterations:
        pid, = ts.get_particle( ['id'], iteration=iteration )
        pid = np.sort( pid )
        assert len(pid) == N_particles
        assert np.all( ref_pid == pid )

    # Remove openPMD files
    shutil.rmtree('./lab_diags/')
    os.chdir('../')

# Run the tests
if __name__ == '__main__':
    test_boosted_output()
