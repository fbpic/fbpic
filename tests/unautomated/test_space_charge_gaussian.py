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
import numpy as np
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.lpa_utils.bunch import add_elec_bunch_gaussian

# The simulation box
Nz = 400         # Number of gridpoints along z
zmin = -40.e-6   # Start of the box in z (meters)
zmax = 0.e-6     # End of the box in z (meters)
Nr = 100         # Number of gridpoints along r
rmax = 100.e-6   # Length of the box along r (meters)
Nm = 2           # Number of modes used
n_order = -1     # The order of the stencil in z

# The simulation timestep
dt = (zmax-zmin)/Nz/c   # Timestep (seconds)
N_step = 800     # Number of iterations to perform
N_show = 40     # Number of timestep between every plot
show_fields = False
l_boost = False

if l_boost:
    # Boosted frame
    gamma_boost = 15.
    boost = BoostConverter(gamma_boost)
else:
    gamma_boost = 1.
    boost = None

# Initialize the simulation object
sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
    -1.e-6, 0., -1.e-6, 0., 1, 1, 1, 1.e18,
    zmin=zmin, n_order=n_order, boundaries='open',
    gamma_boost=gamma_boost )

# Configure the moving window
sim.set_moving_window(v=c, gamma_boost=gamma_boost)

# Suppress the particles that were intialized by default and add the bunch
sim.ptcl = [ ]

# Bunch parameters
sig_r = 3.e-6
sig_z = 3.e-6
n_emit = 1.e-6
gamma0 = 100.
sig_gamma = 1.
Q = 10.e-12
N = 100000
tf = 40.e-6 / (np.sqrt(1.-1./gamma0**2)*c)
zf = 20.e-6

# Add gaussian electron bunch
add_elec_bunch_gaussian( sim, sig_r, sig_z, n_emit, gamma0, sig_gamma,
                         Q, N, tf, zf, boost=boost )

if show_fields:
    # Show the initial fields
    plt.figure(0)
    sim.fld.interp[0].show('Ez')
    plt.figure(1)
    sim.fld.interp[0].show('Er')
    plt.show()
    print( 'Done' )

# Create empty arrays for saving rms bunch sizes
sig_zp = np.zeros(int(N_step/N_show)+1)
sig_xp = np.zeros(int(N_step/N_show)+1)
sig_yp = np.zeros(int(N_step/N_show)+1)

# Set initial bunch sizes
sig_zp[0] = np.std(sim.ptcl[0].z)
sig_xp[0] = np.std(sim.ptcl[0].x)
sig_yp[0] = np.std(sim.ptcl[0].y)

# Create array corresponding to the propagation distance of the bunch
z_prop  = np.arange(int(N_step/N_show)+1) * ((zmax-zmin)/Nz) * N_show
z_prop += -20.e-6
if l_boost:
    z_prop, = boost.copropag_length([z_prop], beta_object = boost.beta0)

# Carry out the simulation
for k in range(int(N_step/N_show)) :
    sim.step(N_show)
    sig_zp[k+1] = np.std(sim.ptcl[0].z)
    sig_xp[k+1] = np.std(sim.ptcl[0].x)
    sig_yp[k+1] = np.std(sim.ptcl[0].y)
    if show_fields:
        # Show the fields
        plt.figure(0)
        plt.clf()
        sim.fld.interp[0].show('Ez')
        plt.figure(1)
        plt.clf()
        sim.fld.interp[0].show('Er')
        plt.show()

# Plot the evolution of the rms bunch size
plt.figure(0)
plt.clf()
plt.plot(z_prop*1.e6, sig_zp*1.e6, label = 'sig_z')
plt.plot(z_prop*1.e6, sig_xp*1.e6, label = 'sig_x')
plt.plot(z_prop*1.e6, sig_yp*1.e6, label = 'sig_y')
plt.xlabel('Propagation distance [mu]')
plt.ylabel('rms of bunch size [mu]')
plt.legend()
plt.show()
