# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It simulates a relativistic plasma flowing through a periodic box.
This test checks that the Galilean option prevents the Cherenkov instability.

The parameters are the same as in the article 
"Elimination of Numerical Cherenkov Instability in flowing-plasma Particle-In-Cell simulations by using Galilean coordinates"

Usage
-----
In order to show the images of instability
$ python tests/test_boosted.py
(except when setting show to False in the parameters below)

In order to let Python check the agreement
$ py.test -q tests/test_boosted.py
or
$ python setup.py test
"""

# -------
# Imports
# -------
import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
# Import the relevant structures in FBPIC
from fbpic.main import Simulation

# ----------
# Parameters
# ----------
use_cuda = True

# Whether to show the plots of the instability
show = True

# Speed of galilean frame (set to 0 for a normal simulation)
v_galilean = -0.999999*c

# The simulation box
Nz = 50         # Number of gridpoints along z
zmax = 9.82     # Length of the box along z (meters)
zmin = -9.82
Nr = 25          # Number of gridpoints along r
rmax = 9.82     # Length of the box along r (meters)
Nm = 2           # Number of modes used
# The simulation timestep
dt = (zmax-zmin)/Nz/c   # Timestep (seconds)
N_step = 100     # Number of iterations to perform

# The boost
gamma_boost = 130.
uz_m = np.sqrt(gamma_boost**2-1)

# The particles
p_zmin = zmin  # Position of the beginning of the plasma (meters)
p_zmax = zmax  # Position of the end of the plasma (meters)
p_rmin = 0.    # Minimal radial position of the plasma (meters)
p_rmax = rmax  # Maximal radial position of the plasma (meters)
n_e = gamma_boost/(4*3.14*2.81e-15)
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r
p_nt = 4         # Number of particles per cell along theta

# The moving window
v_window = c       # Speed of the window

# ---------------------------
# Carrying out the simulation
# ---------------------------
def get_Er_rms(sim):
    Er0 = sim.fld.interp[0].Er
    Er1 = sim.fld.interp[1].Er
    Er_tot = np.mean( abs(Er0)**2 + abs(Er1)**2 )
    return( np.sqrt( Er_tot ) )

def test_instability_standard( show=False ):
    """
    Run a simulation with the standard scheme and check 
    that it is unstable
    """
    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
        p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
        zmin=zmin, initialize_ions=True,
        exchange_period=1, n_guard=5,
        boundaries='periodic', use_cuda=use_cuda )

    # Give a relativistic velocity to the particle, with some noise
    np.random.seed(0)
    sim.ptcl[0].uz[:] = uz_m + 1.e-10*np.random.normal(size=sim.ptcl[0].Ntot)
    sim.ptcl[0].inv_gamma[:] = 1./np.sqrt( 1 + sim.ptcl[0].uz**2 )
    sim.ptcl[1].uz[:] = uz_m + 1.e-10*np.random.normal(size=sim.ptcl[0].Ntot)
    sim.ptcl[1].inv_gamma[:] = 1./np.sqrt( 1 + sim.ptcl[1].uz**2 )
    
    # Perform the simulation; record the rms electric field every 50 timestep
    Er_rms = np.zeros(int(N_step/10)+1)
    t = np.zeros(int(N_step/10+1))
    Er_rms[0] = get_Er_rms(sim)
    t[0] += sim.time
    for i in range(int(N_step/10)):
        sim.step( 10, show_progress=True, move_momenta=False )
        print('Checkpoint %d' %i)
        Er_rms[i+1] = get_Er_rms(sim)
        t[i+1] += sim.time
        print('Calculated RMS')
        
    # Check/plot the results
    if show:
        plt.plot( t, Er_rms, label='Standard scheme' )
        plt.ylabel('RMS(Er)')
        plt.xlabel('Time')


    
if __name__ == '__main__':

    test_instability_standard( show=show )
    
    if show==True:
        plt.show()
