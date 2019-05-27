# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file tests the particle pusher and the external fields by
by studying the motion of particles in plane laser wave.

Usage :
from the top-level directory of FBPIC run
$ python tests/test_external_fields.py

"""
import numpy as np
from scipy.constants import e, m_e, c
from fbpic.main import Simulation
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.lpa_utils.external_fields import ExternalField
import math

# Parameters
# ----------
use_cuda = True

show = True     # Whether to show the results to the user, or to
                # automatically determine if they are correct.

# Dimensions of the box
# (low_resolution since the external field are not resolved on the grid)
Nz = 5
Nr = 10
Nm = 2
zmin = 0.e-6
zmax = 0.8e-6
rmax = 2.e-6

# Properties of the laser
a0 = 1.
lambda0 = 0.8e-6
k0 = 2*np.pi/lambda0

# Particles (one per cell, only initialized near the axis)
p_rmax = rmax/Nr
p_nt = 1
p_nr = 1
p_nz = 1
n = 1.

def run_external_laser_field_simulation(show, gamma_boost=None):
    """
    Runs a simulation with a set of particles whose motion corresponds
    to that of a particle that is initially at rest (in the lab frame)
    before being reached by a plane wave (propagating to the right)

    In the lab frame, the motion is given by
    ux = a0 sin ( k0(z-ct) )
    uz = ux^2 / 2    (from the conservation of gamma - uz)

    In the boosted frame, the motion is given by
    ux = a0 sin ( k0 gamma0 (1-beta0) (z-ct) )
    uz = - gamma0 beta0 + gamma0 (1-beta0) ux^2 / 2
    """
    # Time parameters
    dt = lambda0/c/200  # 200 points per laser period
    N_step = 400        # Two laser periods

    # Initialize BoostConverter object
    if gamma_boost is None:
        boost = BoostConverter(gamma0=1.)
    else:
        boost = BoostConverter(gamma_boost)
    # Reduce time resolution, for the case of a boosted simulation
    if gamma_boost is not None:
        dt = dt*(1.+boost.beta0)/boost.gamma0

    # Initialize the simulation
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
        initialize_ions=False, zmin=zmin,
        use_cuda=use_cuda, boundaries='periodic',
        gamma_boost=gamma_boost )
    # Add electrons
    sim.ptcl = []
    sim.add_new_species( -e, m_e, n=n, p_rmax=p_rmax,
                          p_nz=p_nz, p_nr=p_nr, p_nt=p_nt )

    # Add the external fields
    sim.external_fields = [
        ExternalField(laser_func, 'Ex', a0*m_e*c**2*k0/e,
                      lambda0, gamma_boost=gamma_boost),
        ExternalField(laser_func, 'By', a0*m_e*c*k0/e,
                      lambda0, gamma_boost=gamma_boost)
    ]

    # Prepare the arrays for the time history of the pusher
    Nptcl = sim.ptcl[0].Ntot
    x = np.zeros( (N_step, Nptcl) )
    y = np.zeros( (N_step, Nptcl) )
    z = np.zeros( (N_step, Nptcl) )
    ux = np.zeros( (N_step, Nptcl) )
    uy = np.zeros( (N_step, Nptcl) )
    uz = np.zeros( (N_step, Nptcl) )

    # Prepare the particles with proper transverse and longitudinal momentum,
    # at t=0 in the simulation frame
    k0p = k0*boost.gamma0*(1.-boost.beta0)
    sim.ptcl[0].ux = a0*np.sin( k0p*sim.ptcl[0].z )
    sim.ptcl[0].uz[:] = -boost.gamma0*boost.beta0 \
                    + boost.gamma0*(1-boost.beta0)*0.5*sim.ptcl[0].ux**2

    # Push the particles over N_step and record the corresponding history
    for i in range(N_step) :
        # Record the history
        x[i,:] = sim.ptcl[0].x[:]
        y[i,:] = sim.ptcl[0].y[:]
        z[i,:] = sim.ptcl[0].z[:]
        ux[i,:] = sim.ptcl[0].ux[:]
        uy[i,:] = sim.ptcl[0].uy[:]
        uz[i,:] = sim.ptcl[0].uz[:]
        # Take a simulation step
        sim.step(1)

    # Compute the analytical solution
    t = sim.dt*np.arange(N_step)
    # Conservation of ux
    ux_analytical = np.zeros( (N_step, Nptcl) )
    uz_analytical = np.zeros( (N_step, Nptcl) )
    for i in range(N_step):
        ux_analytical[i,:] = a0*np.sin( k0p*(z[i,:] - c*t[i]) )
        uz_analytical[i,:] = -boost.gamma0*boost.beta0 \
                    + boost.gamma0*(1-boost.beta0)*0.5*ux_analytical[i,:]**2

    # Show the results
    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))

        plt.subplot(211)
        plt.plot( t, ux_analytical, '--' )
        plt.plot( t, ux, 'o' )
        plt.xlabel('t' )
        plt.ylabel('ux')

        plt.subplot(212)
        plt.plot( t, uz_analytical, '--' )
        plt.plot( t, uz, 'o')
        plt.xlabel('t' )
        plt.ylabel('uz')

        plt.show()
    else:
        assert np.allclose( ux, ux_analytical, atol=5.e-2 )
        assert np.allclose( uz, uz_analytical, atol=5.e-2 )

def laser_func( F, x, y, z, t, amplitude, length_scale ):
    """
    Function to be called at each timestep on the particles
    """
    return( F + amplitude*math.cos( 2*np.pi*(z-c*t)/length_scale ) )

def test_external_fields_lab(show=False):
    "Function that is run by py.test, when doing `python setup.py test`"
    run_external_laser_field_simulation( show, None )

def test_external_fields_boost(show=False):
    "Function that is run by py.test, when doing `python setup.py test`"
    run_external_laser_field_simulation( show, gamma_boost=10 )

if __name__ == '__main__' :

    test_external_fields_lab( show )
    test_external_fields_boost( show )
