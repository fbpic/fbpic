# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of generic functions for multithreaded CPU execution.
"""
import os
from numba import njit

# By default threading is enabled
threading_enabled = True

# Check if the environment variable FBPIC_DISABLE_THREADING is set to 1
# and in that case, disable threading
if 'FBPIC_DISABLE_THREADING' in os.environ:
    if int(os.environ['FBPIC_DISABLE_THREADING']) == 1:
        threading_enabled = False

# If the user request threading (by not setting FBPIC_DISABLE_THREADING)
# check if it is indeed installed
if threading_enabled:
    try:
        # Try to import the threading function prange
        from numba import prange
    except ImportError:
        threading_enabled = False
        print('*** Threading not available for the simulation.')
        print('*** (Please make sure that numba>0.34 is installed)')

# Set the function njit_parallel and prange to the correct object
if not threading_enabled:
    # Use regular serial compilation function
    njit_parallel = njit
    prange = range
else:
    # Use the parallel compilation function
    njit_parallel = njit( parallel=True )
    from numba import prange
