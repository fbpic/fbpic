# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of generic functions for multithreaded CPU execution.
"""
import os, sys
import warnings
import numpy as np
from numba import njit
import numba
numba_minor_version = int(numba.__version__.split('.')[1])

# By default threading is enabled, except on Windows (not supported by Numba)
threading_enabled = True
if sys.platform == 'win32':
    threading_enabled = False

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
        from numba import prange as numba_prange
        # Check that numba is version 0.34 or higher than 0.36
        # (other versions fail)
        assert ( numba_minor_version==34 or numba_minor_version >= 36 )
    except (ImportError, AssertionError):
        threading_enabled = False
        warnings.warn(
            'Threading not available for the simulation.\n'
            'Simulations will still run, but using only 1 thread on CPU.\n'
            'Please ensure that numba 0.34 or >=0.36 is installed.\n'
            '(e.g. by typing `conda update numba` in a terminal)')

# Check if the environment variable FBPIC_DISABLE_CACHING is set to 1
# and in that case, disable threading
caching = True
if 'FBPIC_DISABLE_CACHING' in os.environ:
    if int(os.environ['FBPIC_DISABLE_CACHING']) == 1:
        caching = False

# Set the function njit_parallel and prange to the correct object
if not threading_enabled:
    # Use regular serial compilation function
    # Uses `cache=True` to avoid re-compilation
    njit_parallel = njit( cache=caching )
    prange = range
    nthreads = 1
else:
    # Use the parallel compilation function (Use caching if available)
    if numba_minor_version < 45:
        caching = False
    njit_parallel = njit( parallel=True, cache=caching )
    prange = numba_prange
    nthreads = numba.config.NUMBA_NUM_THREADS


def get_chunk_indices( Ntot, nthreads ):
    """
    Divide `Ntot` in `nthreads` chunks (almost equal in size), and
    return the indices that bound the chunks

    Parameters
    ----------
    Ntot: int
        Typically, the number of particles in a species
    nthreads: int
        The number of threads among which the work is divided

    Return
    ------
    ptcl_chunk_indices: a 1d array of integers (uint64)
        An array of size nthreads+1, that contains the integers that
        bound the chunks (its first element is 0 and its last element is Ntot)
    """
    # Calculate average number of particles per chunk
    n_avg_per_chunk = int( Ntot/nthreads )
    # Attribute n_avg_per_chunk to each thread
    ptcl_chunk_indices = np.array(
        [ i_chk*n_avg_per_chunk for i_chk in range(nthreads+1) ],
        dtype=np.uint64 )
    # Modify for the last thread: take the remaining particles up to Ntot
    ptcl_chunk_indices[-1] = Ntot

    return( ptcl_chunk_indices )
