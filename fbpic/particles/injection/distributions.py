# Copyright 2020, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of simple classes for particle distribution
"""

class CartesianPyFuncDistribution( object ):
    """
    Distribution defined by a Python function, which depends on x, y, z
    """
    def __init__(self, dens_func):
        # TODO: Check the arguments
        self.dens_func = dens_func

    def __call__(self, x, y, z):
        return self.dens_func(x, y, z)


class CylindricalPyFuncDistribution( object ):
    """
    Distribution defined by a Python function, which depends on z and r
    """
    def __init__(self, dens_func):
        # TODO: Check the arguments
        self.dens_func = dens_func

    def __call__(self, z, r):
        return self.dens_func(z, r)
