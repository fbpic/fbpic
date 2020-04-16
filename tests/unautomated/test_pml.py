# Copyright 2019, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Soeren Jalas
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It tests the Perfectly-Matched-Layers, by initializing a laser with
a small waist and letting it diffract into the radial PML.
The test then checks that the profile of the laser (inside the
physical domain) is identical to the theoretical profile
(i.e. that the reflections are negligible).

Usage :
-------
In order to show the images of the laser, and manually check the
agreement between the simulation and the theory:
$ python tests/test_pml.py
(except when setting show to False in the parameters below)

In order to let Python check the agreement between the curve without
having to look at the plots
$ py.test -q tests/test_pml.py
or
$ python setup.py test
"""
import numpy as np
from scipy.constants import c
from fbpic.main import Simulation
from fbpic.openpmd_diag import FieldDiagnostic, \
    restart_from_checkpoint, set_periodic_checkpoint
from fbpic.lpa_utils.laser import add_laser_pulse, \
    GaussianLaser, LaguerreGaussLaser

# Parameters
# ----------
# (See the documentation of the function propagate_pulse
# below for their definition)

use_cuda = True

# Simulation box
Nz = 360
zmin = -6.e-6
zmax = 6.e-6
Nr = 50
Lr = 4.e-6
Nm = 2
res_r  = Lr/Nr
n_order = 32
# Laser pulse
w0 = 1.5e-6
lambda0 = 0.8e-6
tau = 10.e-15
a0 = 1.
zf = 0.
z0 = 0.
# Propagation
L_prop = 40.e-6
dt = (zmax-zmin)*1./c/Nz
N_diag = 4 # Number of diagnostic points along the propagation
restart = False

# Initialize the simulation object
sim = Simulation( Nz=Nz, zmax=zmax, Nr=Nr, rmax=Lr, Nm=Nm, dt=dt,
                  n_order=n_order, zmin=zmin, use_cuda=use_cuda )

# Simultaneously check the mode m=0 and m=1 within the same simulation
# by initializing and laser in m=0 and m=1
# - Mode 0: Build a radially-polarized pulse from 2 Laguerre-Gauss profiles
profile0 = LaguerreGaussLaser( 0, 1, 0.5*a0, w0, tau, z0, zf=zf,
            lambda0=lambda0, theta_pol=0., theta0=0. ) \
         + LaguerreGaussLaser( 0, 1, 0.5*a0, w0, tau, z0, zf=zf,
            lambda0=lambda0, theta_pol=np.pi/2, theta0=np.pi/2 )
# - Mode 1: Use a regular linearly-polarized pulse
profile1 = GaussianLaser( a0=a0, waist=w0, tau=tau,
            lambda0=lambda0, z0=z0, zf=zf )

if not restart:
    # Add the profiles to the simulation
    add_laser_pulse( sim, profile0 )
    add_laser_pulse( sim, profile1 )
else:
    restart_from_checkpoint( sim )

# Calculate the total number of steps
N_step = int( round( L_prop/(c*dt) ) )
diag_period = int( round( N_step/N_diag ) )

# Add openPMD diagnostics
sim.diags = [
    FieldDiagnostic( diag_period, sim.fld, fieldtypes=["E"], comm=sim.comm ) ]

set_periodic_checkpoint(sim, N_step//2)

# Do only half the steps
sim.step( N_step//2+1 )
