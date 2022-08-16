# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kristoffer Lindvall
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the particle boundary methods on the GPU using CUDA.
"""
from numba import cuda
from fbpic.utils.cuda import compile_cupy

@compile_cupy
def reflect_particles_left( zmin, z, uz ):
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
    """
    i = cuda.grid(1)
    if i < z.shape[0]:
        if z[i] < zmin:
            uz[i] *= -1
            z[i] = ( zmin - z[i] ) + zmin

@compile_cupy
def reflect_particles_right( zmax, z, uz ):
    """
    Reflect particles at right boundary
    Parameters
    ----------
    zmax : right boundary position
    
    z : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)
    uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles
    """
    i = cuda.grid(1)
    if i < z.shape[0]:
        if z[i] > zmax:
            uz[i] *= -1
            z[i] = zmax - ( z[i] - zmax )

@compile_cupy
def stop_particles_left( zmin, z, ux, uy, uz ):
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
    """
    i = cuda.grid(1)
    if i < z.shape[0]:
        if z[i] < zmin:
            ux[i] = 0
            uy[i] = 0
            uz[i] = 0
            z[i] = zmin

@compile_cupy
def stop_particles_right( zmax, z, ux, uy, uz ):
    """
    Stop particles at right boundary. Particle momenta are set to 0.
    Parameters
    ----------
    zmax : left boundary position
    z : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)
    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles
    """
    i = cuda.grid(1)
    if i < z.shape[0]:
        if z[i] > zmax:
            ux[i] = 0
            uy[i] = 0
            uz[i] = 0
            z[i] = zmax