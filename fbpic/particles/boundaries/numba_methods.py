# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kristoffer Lindvall
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the particle boundary methods on the CPU with numba.
"""
import numba
from fbpic.utils.threading import njit_parallel, prange

@njit_parallel
def reflect_particles_left_numba( zmin, z, uz, Ntot ):
    """
    Reflect particles at left boundary
    Parameters
    ----------
    zmin : left boundary position
    z : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)
    uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles
    Ntot : total number of particles
    """

    # Particle reflect (in parallel if threading is installed)
    for ip in prange(Ntot):
        if z[ip] < zmin:
            uz[ip] *= -1
            z[ip] = ( zmin - z[ip] ) + zmin

    return z, uz

@njit_parallel
def reflect_particles_right_numba( zmax, z, uz, Ntot ):
    """
    Reflect particles at left boundary
    Parameters
    ----------
    zmax : right boundary position
    z : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)
    uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles
    Ntot : total number of particles
    """

    # Particle reflect (in parallel if threading is installed)
    for ip in prange(Ntot):
        if z[ip] > zmax:
            uz[ip] *= -1
            z[ip] = zmax - ( z[ip] - zmax )

    return z, uz

@njit_parallel
def stop_particles_left_numba( zmin, z, ux, uy, uz, Ntot ):
    """
    Stop particles at left boundary. Particle momenta are set to 0.
    Parameters
    ----------
    zmin : left boundary position
    z : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)
    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles
    Ntot : total number of particles
    """
    for ip in prange(Ntot):
        if z[ip] < zmin:
            ux[ip] = 0
            uy[ip] = 0
            uz[ip] = 0
            z[ip] = zmin

    return z, ux, uy, uz

@njit_parallel
def stop_particles_right_numba( zmax, z, ux, uy, uz, Ntot ):
    """
    Stop particles at right boundary. Particle momenta are set to 0.
    Parameters
    ----------
    zmax : right boundary position
    z : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)
    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles
    Ntot : total number of particles
    """
    for ip in prange(Ntot):
        if z[ip] > zmax:
            ux[ip] = 0
            uy[ip] = 0
            uz[ip] = 0
            z[ip] = zmax
    
    return z, ux, uy, 