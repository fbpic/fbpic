# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the linear and cubic particle shape factors for the deposition.
"""
import math

# -------------------------------
# Particle shape Factor functions
# -------------------------------

# Linear shapes

# longitudinal (z)
def Sz_linear(cell_position, index):
    s = math.ceil(cell_position) - cell_position
    if index == 1:
        s = 1.-s
    return s

# transversal (r)
def Sr_linear(cell_position, index, flip, beta_n):
    # Get radial cell index
    ir = int(math.ceil(cell_position)) - 1
    # u: position of the particle with respect to its left neighbor gridpoint
    # (u is between 0 and 1)
    u = cell_position - ir
    s = (1.-u) + beta_n*(1.-u)*u
    if index == 1:
        s = 1.-s
    # Check if the cell to which the particle deposits is below the axis
    if index + ir < 0:
        # In this case, flip the sign of the particle contribution
        s *= flip
    return s

# Cubic shapes

# longitudinal (z)
def Sz_cubic(cell_position, index):
    iz = int(math.ceil(cell_position)) - 2
    # u: position of the particle with respect to its left neighbor gridpoint
    # (u is between 0 and 1)
    u = cell_position - iz - 1
    s = 0.
    if index == 0:
        s = (1./6.)*(1.-u)**3
    elif index == 1:
        s = (1./6.)*(3.*u**3 - 6.*u**2 + 4.)
    elif index == 2:
        s = (1./6.)*(3.*(1.-u)**3 - 6.*(1.-u)**2 + 4.)
    elif index == 3:
        s = (1./6.)*u**3
    return s

# transversal (r)
def Sr_cubic(cell_position, index, flip, beta_n):
    # Get radial cell index
    ir = int(math.ceil(cell_position)) - 2
    # u: position of the particle with respect to its left neighbor gridpoint
    # (u is between 0 and 1)
    u = cell_position - ir - 1
    s = 0.
    if index == 0:
        s = (1./6.)*(1.-u)**3
    elif index == 1:
        s = (1./6.)*(3.*u**3 - 6.*u**2 + 4.)
        s += beta_n*(1.-u)*u # Add Ruyten correction
    elif index == 2:
        s = (1./6.)*(3.*(1.-u)**3 - 6.*(1.-u)**2 + 4.)
        s -= beta_n*(1.-u)*u # Add Ruyten correction
    elif index == 3:
        s = (1./6.)*u**3
    # Check if the cell to which the particle deposits is below the axis
    if index + ir < 0:
        # In this case, flip the sign of the particle contribution
        s *= flip
    return s
