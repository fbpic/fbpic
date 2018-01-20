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
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.openpmd_diag import FieldDiagnostic
from fbpic.lpa_utils.bunch import add_elec_bunch_file

# The simulation box
Nz = 400         # Number of gridpoints along z
zmax = 40.e-6    # Length of the box along z (meters)
Nr = 100         # Number of gridpoints along r
rmax = 100.e-6   # Length of the box along r (meters)
Nm = 2           # Number of modes used
n_order = 32     # Order of the stencil

# The simulation timestep
dt = zmax/Nz/c   # Timestep (seconds)

# Initialize the simulation object
sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
    0, 0, 0, 0, 2, 2, 4, 0.,
    n_order=n_order, boundaries='open' )
# Configure the moving window
sim.set_moving_window( v=c )
# Suppress the particles that were intialized by default and add the bunch
sim.ptcl = [ ]
add_elec_bunch_file( sim, filename='test_space_charge_file_data.txt',
                     Q_tot=1.e-12, z_off=20.e-6 )
# Set the diagnostics
sim.diags = [ FieldDiagnostic(10, sim.fld, comm=sim.comm) ]
# Perform one simulation step (essentially in order to write the diags)
sim.step(1)
