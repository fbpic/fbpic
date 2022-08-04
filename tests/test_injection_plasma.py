# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It verifies that the continuous injection with a moving window
velocity of zero by:
- Initializing a plasma inside the box and injecting new plasma
in each injection period
- Checking that the deposited particles 

This is done in the lab frame v_moving_window=0

Usage :
from the top-level directory of FBPIC run
$ python tests/test_continuous_injection.py
"""
from scipy.constants import c, e, m_e
from fbpic.main import Simulation
import numpy as np

# Parameters
# ----------
use_cuda = True

show = True     # Whether to show the results to the user, or to
                # automatically determine if they are correct.

# Dimensions of the box
Nz = 100
Nr = 50
Nm = 1
zmin = -0.5
zmax = 0.5
rmax = 0.5
dz = (zmax-zmin)/Nz
# Particles
p_nr = 1
p_nz = 1
p_nt = 1
n = 1e19

# Injection scheme
exchange_period = 1             # Exhange period
injection_period = 2            # Multiple of exhange period
injection_duration = np.inf     # Injection period [s]

# -------------
# Test function
# -------------

def test_inject_plasma(show=False):
    "Run test in lab frame with some plasma at t=0"
    run_continuous_injection( show )

def run_continuous_injection( show, N_check=2, cartesian=False ):
    # Chose the time step
    dt = (zmax-zmin)/Nz/c

    # define ellipse for initial charge density profile
    # center of ellipse (x0,y0)
    r0 = 0.
    z0 = 0.        
    rr2 = (0.05)**2 # semi-minor radius squared
    rz2 = (0.05)**2

    def dens_func_cylindrical( z, r ): 
        """Returns relative density at position z and r"""
        # Allocate relative density
        n = np.ones_like(z)
        n = np.where( ( (r-r0)**2/rr2 + (z-z0)**2/rz2 ) >= 1,
            0.0, n)
        return(n)
    
    if cartesian:
        def dens_func( x, y, z ):
            return dens_func_cylindrical( z, (x**2 + y**2)**.5 )
    else:
        dens_func = dens_func_cylindrical

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
        n_order=-1, use_cuda=use_cuda, 
        exchange_period=exchange_period,
        injection={'p':injection_period, 't':injection_duration},
        boundaries={'z':'open', 'r':'reflective'})
        # 'r': 'open' can also be used, but is more computationally expensive

    # Create the plasma electrons
    u_th = 0.00001
    sim.add_new_species( q=-e, m=m_e, n=n,
        dens_func=dens_func, ux_th=u_th, uy_th=u_th, uz_th=u_th,
        p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
        continuous_injection=True)

    Ntot_init = 0
    for species in sim.ptcl:
        Ntot_init += len(species.x)

    sim.set_moving_window(v=0) 
    
    # Check that the number of particles are correct
    # after different timesteps
    N_step = int( Nz/N_check/2 )
    for i in range( N_check ):
        sim.step( N_step, move_momenta=False )
        check_particle_number( sim, Ntot_init, (i+1) * N_step + 1 )

def check_particle_number( sim, Ntot_init, iteration ):
    Ntot = 0
    for species in sim.ptcl:
        Ntot += len(species.x)

    Ntot_expected = Ntot_init
    if iteration >= injection_period:
        k = int( iteration / injection_period )
        Ntot_expected = k * Ntot_init

    assert Ntot_expected==Ntot

if __name__ == '__main__' :
    test_inject_plasma(show)
