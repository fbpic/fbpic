# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of generic functions for multithreaded CPU execution.
"""
try:
    # Try to import the threading function prange
    from numba import prange
    threading_installed = True
except ImportError:
    # If not replace threading functions by single-thread functions
    prange = range
    threading_installed = False
