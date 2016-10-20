# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file tests the particle pusher implemented in particles.py,
by studying the motion of particles in different sets of fields.

Usage :
from the top-level directory of FBPIC run
$ python tests/test_fields.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, c
from fbpic.particles import Particles

# To be done :
# - Abstraction with general field function (x,y,z,t) -> (E,B)
#   and analytical solution
# - Add a function for a laser (linearly polarized : figure of eight,
#   circularly polarized : constant motion)

def test_constant_B( Bz, q, m, rmin, rmax, gammamin, gammamax,
                     Npart, Npts_per_gyr, N_gyr ) :
    """
    Test for a constant magnetic field
    """

    # Find the minimum dt
    w_larmor = q*Bz/m
    dt = 2*np.pi*gammamin/( abs(w_larmor)*Npts_per_gyr )
    
    # Initialize the particle structure
    ptcl = Particles( q, m, 0., 1, 0., 0., Npart, rmin, rmax, 1, dt )

    # Set the magnetic field
    ptcl.Bz[:] = Bz

    # Initialize the right momenta for the particles
    gamma = np.linspace( gammamin, gammamax, Npart )
    utheta = np.sqrt( gamma**2 - 1 )
    invr = 1./np.sqrt( ptcl.x**2 + ptcl.y**2 )
    ptcl.ux = utheta * ptcl.y * invr * np.sign(q)
    ptcl.uy = - utheta * ptcl.x * invr * np.sign(q)
    ptcl.inv_gamma = 1./gamma
    # NB : the particles positions and momenta are considered
    # to be at t=0 initially
    
    # Compute the total number of steps
    Nstep = N_gyr * Npts_per_gyr
    it = np.arange(1, Nstep+1)
    t = np.zeros( (Nstep, Npart) )
    t[:,:] = dt*it[:, np.newaxis]
    # Compute the analytical solution
    phi = np.angle( ptcl.uy + 1.j*ptcl.ux )
    w = w_larmor/gamma
    xc = ptcl.x + c*ptcl.uy/(gamma*w)
    yc = ptcl.y - c*ptcl.ux/(gamma*w)
    ux_analytic = utheta[np.newaxis,:] * np.sin( w*t + phi[np.newaxis,:] )
    uy_analytic = utheta[np.newaxis,:] * np.cos( w*t + phi[np.newaxis,:] )
    x_analytic = xc[np.newaxis,:] - c*uy_analytic/((gamma*w)[np.newaxis,:])
    y_analytic = yc[np.newaxis,:] + c*ux_analytic/((gamma*w)[np.newaxis,:])
    # Prepare the arrays for the time history of the pusher
    x = np.zeros( (Nstep, Npart) )
    y = np.zeros( (Nstep, Npart) )
    ux = np.zeros( (Nstep, Npart) )
    uy = np.zeros( (Nstep, Npart) )
    
    # Push the particles over Nstep and record the corresponding history
    for i in range(Nstep) :
        # Push the particles
        ptcl.halfpush_x()
        ptcl.push_p()
        ptcl.halfpush_x()
        # Record the history
        x[i,:] = ptcl.x[:]
        y[i,:] = ptcl.y[:]
        ux[i,:] = ptcl.ux[:]
        uy[i,:] = ptcl.uy[:]

    # Plot the results
    plt.figure()
    plt.plot( t, x_analytic, '--' )
    plt.plot( t, x, 'o' )
    plt.xlabel('t')
    plt.ylabel('x')

    plt.figure()
    plt.plot( t, y_analytic, '--' )
    plt.plot( t, y, 'o' )
    plt.xlabel('t')
    plt.ylabel('y')
    
    plt.figure()
    plt.subplot(aspect='equal')
    plt.plot( x_analytic, y_analytic, '--' )
    plt.plot( x, y, 'o' )
    plt.xlabel('x')
    plt.ylabel('y')
    
if __name__ == '__main__' :

    Bz = 1.
    q = -e
    m = m_e
    rmin = 0.5*m*c/(abs(q)*Bz)
    rmax = m*c/(abs(q)*Bz)
    gammamin = 200.
    gammamax = 400.
    Npart = 4
    Npts_per_gyr = 20
    N_gyr = 5
    
    test_constant_B( Bz, q, m, rmin, rmax, gammamin, gammamax,
            Npart, Npts_per_gyr, N_gyr )

    plt.show()
