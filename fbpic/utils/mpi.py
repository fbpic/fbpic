# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It imports a set of MPI objects - or a set of dummy replacements when MPI
is not installed
"""
import os
try:
    # Try to import MPI objects
    import mpi4py
    from mpi4py import MPI
    from mpi4py.MPI import COMM_WORLD as comm
    # Dictionary of correspondance between numpy types and mpi types
    # (Necessary when calling Gatherv)
    mpi_type_dict = { 'float32': MPI.REAL4,
                      'float64': MPI.REAL8,
                      'complex64': MPI.COMPLEX8,
                      'complex128': MPI.COMPLEX16,
                      'uint64': MPI.UINT64_T }
    mpi_installed = True

    # Check if the environment variable FBPIC_ENABLE_GPUDIRECT is set to 1
    # and in that case, enable direct MPI communication of CUDA GPU arrays
    # with a CUDA-aware MPI Implementation
    if 'FBPIC_ENABLE_GPUDIRECT' in os.environ:
        if int(os.environ['FBPIC_ENABLE_GPUDIRECT']) == 1:
            gpudirect_enabled = True
        else:
            gpudirect_enabled = False
    else:
        gpudirect_enabled = False

    if gpudirect_enabled:
        mpi4py_version_number = mpi4py.__version__.split('.')
        mpi4py_major_version = int(mpi4py_version_number[0])
        mpi4py_minor_version = int(mpi4py_version_number[1])
        if (mpi4py_major_version < 3) or (mpi4py_minor_version < 1):
            raise RuntimeError(
                "In order to use GPU Direct, you need to install mpi4py>=3.1.")

except ImportError:
    # If MPI is not installed, define dummy replacements
    import warnings
    warnings.warn(
        'MPI is not properly installed.\n'
        'Simulations without domain decomposition will still run properly.\n'
        'In order to diagnose the problem, type:\n'
        'mpirun -np 2 python -c "from mpi4py.MPI import COMM_WORLD"')

    class DummyCommunicator(object):
        """Dummy replacement for COMM_WORLD when mpi4py is not installed."""

        def __init__(self):
            self.rank = 0
            self.size = 1

        def barrier(self):
            pass

    class DummyMPI(object):
        """Dummy replacement for mpi4py.MPI when mpi4py is not installed."""

        def __init__(self):
            self.COMM_WORLD = DummyCommunicator()

    MPI = DummyMPI()
    comm = DummyCommunicator()
    mpi_type_dict = {}
    mpi_installed = False
    gpudirect_enabled = False
