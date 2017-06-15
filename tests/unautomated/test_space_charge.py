# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file tests the space charge initialization implemented
in lpa_utils.py, by initializing a charged bunch and propagating
it for a few steps

Usage :
from the top-level directory of FBPIC run
$ python tests/test_space_charge.py
"""
import matplotlib.pyplot as plt
from scipy.constants import c
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.bunch import add_elec_bunch

# The simulation box
Nz = 400         # Number of gridpoints along z
zmax = 40.e-6    # Length of the box along z (meters)
Nr = 100         # Number of gridpoints along r
rmax = 100.e-6   # Length of the box along r (meters)
Nm = 2           # Number of modes used
n_order = -1     # The order of the stencil in z

# The simulation timestep
dt = zmax/Nz/c   # Timestep (seconds)
N_step = 300     # Number of iterations to perform
N_show = 300     # Number of timestep between every plot

# The particles
gamma0 = 25.
p_zmin = 15.e-6  # Position of the beginning of the bunch (meters)
p_zmax = 25.e-6  # Position of the end of the bunch (meters)
p_rmin = 0.      # Minimal radial position of the bunch (meters)
p_rmax = 5.e-6   # Maximal radial position of the bunch (meters)
n_e = 4.e18*1.e6 # Density (electrons.meters^-3)
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r
p_nt = 4         # Number of particles per cell along theta

# Initialize the simulation object
sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
    p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
    n_order=n_order, boundaries='open' )

# Configure the moving window
sim.set_moving_window(v=c)

# Suppress the particles that were intialized by default and add the bunch
sim.ptcl = [ ]
add_elec_bunch( sim, gamma0, n_e, p_zmin, p_zmax, p_rmin, p_rmax )


# Show the initial fields
plt.figure(0)
sim.fld.interp[0].show('Ez')
plt.figure(1)
sim.fld.interp[0].show('Er')
plt.show()
print( 'Done' )

# Carry out the simulation
for k in range(N_step/N_show) :
    sim.step(N_show)

    plt.figure(0)
    plt.clf()
    sim.fld.interp[0].show('Ez')
    plt.figure(1)
    plt.clf()
    sim.fld.interp[0].show('Er')
    plt.show()
