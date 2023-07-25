# Copyright 2023, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It provides a function to fix the random seed in FBPIC runs.
"""
import numpy as np
from .mpi import MPI
from .cuda import cupy_installed
if cupy_installed:
    import cupy


def set_random_seed( random_seed ):
    """
    Fix the seed of the random number generators.

    Fixing a seed helps ensure that repeatedly running the
    same simulation gives the same result (despite the Monte Carlo
    parts of the code, e.g. ionization, gaussian beam generation, etc.)

    random_seed: int
        The seed of the random number generator.
    """
    # Use a different seed for each MPI rank
    # - Set seed for numpy
    np.random.seed( random_seed + MPI.COMM_WORLD.rank )
    if cupy_installed:
        # - Set seed for cupy
        cupy.random.seed( random_seed + MPI.COMM_WORLD.rank )
