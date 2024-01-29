# Copyright 2023, FBPIC contributors
# Author: Michael J. Quin
# Scientific supervision: Matteo Tamburini
# Code optimization: Kristjan Poder
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a number of methods that are useful for spin processes
on CPU and GPU.
"""
import numpy as np
from scipy.constants import pi
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    import cupy


def random_point_sphere_cpu(Ntot, radius=1.):
    """
    Random distribution of points on the surface of a sphere, centred at
    the origin, is given by a projecting a uniform distribution of points
    on a cylinder onto a sphere.
    """
    # generate z coord
    z = np.random.uniform(-radius, radius, Ntot)
    # azimuthal angle
    phi = np.random.uniform(0, 2 * pi, Ntot)
    # polar angle
    sin_theta = np.sin(np.arccos(z))
    # cartesian coords x and y
    x = sin_theta*np.cos(phi)
    y = sin_theta*np.sin(phi)
    return x, y, z


def random_point_sphere_gpu(Ntot, radius=1.):
    """
    Random distribution of points on the surface of a sphere, centred at
    the origin, is given by a projecting a uniform distribution of points
    on a cylinder onto a sphere.
    """
    # generate z coord
    z = cupy.random.uniform(-radius, radius, Ntot)
    # azimuthal angle
    phi = cupy.random.uniform(0, 2 * pi, Ntot)
    # polar angle
    sin_theta = cupy.sin(cupy.arccos(z))
    # cartesian coords x and y
    x = sin_theta*cupy.cos(phi)
    y = sin_theta*cupy.sin(phi)
    return x, y, z
