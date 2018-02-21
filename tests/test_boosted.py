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
# Import the relevant structures in FBPIC
from fbpic.main import Simulation

# ----------
# Parameters
# ----------
use_cuda = True

# Whether to show the plots of the instability
show = True

# The simulation box
Nz = 40         # Number of gridpoints along z
zmax = 7.86     # Length of the box along z (meters)
zmin = -7.86
Nr = 20          # Number of gridpoints along r
rmax = 7.86      # Length of the box along r (meters)
Nm = 2           # Number of modes used
# The simulation timestep
dt = (zmax-zmin)/Nz/c   # Timestep (seconds)
N_step = 600     # Number of iterations to perform

# The boost
gamma_boost = 130.
uz_m = np.sqrt(gamma_boost**2-1)

# The particles
p_zmin = zmin  # Position of the beginning of the plasma (meters)
p_zmax = zmax  # Position of the end of the plasma (meters)
p_rmin = 0.    # Minimal radial position of the plasma (meters)
p_rmax = rmax  # Maximal radial position of the plasma (meters)
n_e = gamma_boost/(4*3.14*2.81e-15)
p_nz = 2        # Number of particles per cell along z
p_nr = 2        # Number of particles per cell along r
p_nt = 4         # Number of particles per cell along theta

# The moving window
v_window = c       # Speed of the window

# ---------------------------
# Carrying out the simulation
# ---------------------------
def get_Er_rms(sim):
    Er0 = sim.fld.interp[0].Er
    Er1 = sim.fld.interp[1].Er
    Er_tot = np.sqrt( np.average( abs(Er0)**2 + abs(Er1)**2 ) )
    return( Er_tot )

def test_cherenkov_instability( show=False ):
    """
    Run a simulation with the standard and Galilean scheme respectively
    and check that the first one is unstable while the second one is stable
    """
    # Dictionary to record the final value of E
    slope_Erms = {}

    for scheme in [ 'standard', 'galilean', 'pseudo-galilean']:

        # Choose the correct parameters for the scheme
        if scheme == 'standard':
            v_comoving = 0.
            use_galilean = False
        else:
            v_comoving = 0.9999*c
            if scheme == 'galilean':
                use_galilean = True
            else:
                use_galilean = False

        # Initialize the simulation object
        sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
            p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
            zmin=zmin, initialize_ions=True,
            v_comoving=v_comoving, use_galilean=use_galilean,
            boundaries='periodic', use_cuda=use_cuda )

        # Give a relativistic velocity to the particle, with some noise
        sim.ptcl[0].uz[:] = uz_m
        sim.ptcl[0].inv_gamma[:] = 1./np.sqrt( 1 + sim.ptcl[0].uz**2 )
        sim.ptcl[1].uz[:] = uz_m
        sim.ptcl[1].inv_gamma[:] = 1./np.sqrt( 1 + sim.ptcl[1].uz**2 )

        # Perform the simulation;
        # record the rms electric field every 50 timestep
        Er_rms = np.zeros(int(N_step/30)+1)
        t = np.zeros(int(N_step/30+1))
        Er_rms[0] = get_Er_rms(sim)
        t[0] += sim.time
        for i in range(int(N_step/30)):
            sim.step( 30, show_progress=False )
            print('Checkpoint %d' %i)
            Er_rms[i+1] = get_Er_rms(sim)
            t[i+1] += sim.time
            print('Calculated RMS')

        # Check/plot the results
        if show:
            import matplotlib.pyplot as plt
            # Add a plot
            plt.semilogy( t, Er_rms, '-', label=scheme )
            plt.ylabel('RMS(Er)')
            plt.xlabel('Time')
        else:
            # Registed the final value of the slope of the electric field
            slope_Erms[scheme] = np.log( Er_rms[-1] ) - np.log(Er_rms[-2] )

    if show:
        # Show the plot
        plt.legend(loc=0)
        plt.show()
    else:
        # Check that, in the standard case, the electric field is
        # growing much faster, due to the Cherenkov instability
        assert slope_Erms['standard'] > 3.5*slope_Erms['galilean']
        assert slope_Erms['standard'] > 3.5*slope_Erms['pseudo-galilean']

if __name__ == '__main__':

    test_cherenkov_instability( show=show )
