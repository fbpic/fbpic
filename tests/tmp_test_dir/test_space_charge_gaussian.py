# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file tests the space charge initialization of a beam
defined in an external text file, implemented
in lpa_utils.py, by initializing a charged bunch.

This file is used by the automated test `test_space_charge.py`
"""
from scipy.constants import c
import numpy as np
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.openpmd_diag import FieldDiagnostic
from fbpic.lpa_utils.bunch import add_elec_bunch_gaussian

# Set the seed since the gaussian is drawn randomly
np.random.seed(0)

# The simulation box
Nz = 400         # Number of gridpoints along z
zmax = 0.e-6     # Length of the box along z (meters)
zmin = -40.e-6
Nr = 100         # Number of gridpoints along r
rmax = 100.e-6   # Length of the box along r (meters)
Nm = 2           # Number of modes used
n_order = 32     # Order of the stencil

# Bunch parameters
sig_r = 3.e-6
sig_z = 3.e-6
n_emit = 1.e-6
gamma0 = 15.
sig_gamma = 1.
Q = 10.e-12
N = 100000
tf = 0
zf = -20.e-6

# The simulation timestep
dt = (zmax-zmin)/Nz/c   # Timestep (seconds)

# Initialize the simulation object
sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
    0, 0, 0, 0, 2, 2, 4, 0., zmin=zmin,
    n_order=n_order, boundaries='open' )
# Configure the moving window
sim.set_moving_window( v=c )
# Suppress the particles that were intialized by default and add the bunch
sim.ptcl = [ ]
add_elec_bunch_gaussian( sim, sig_r, sig_z, n_emit, gamma0, sig_gamma,
                         Q, N, tf, zf )
# Set the diagnostics
sim.diags = [ FieldDiagnostic(10, sim.fld, comm=sim.comm) ]
# Perform one simulation step (essentially in order to write the diags)
sim.step(1)
