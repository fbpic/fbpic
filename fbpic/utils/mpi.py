# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It imports a set of MPI objects - or a set of dummy replacements when MPI
is not installed
"""
try:
    # Try to import MPI objects
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

except ImportError:
    # If MPI is not installed, define dummy replacements
    print("*** MPI is not properly installed.")
    print("*** In order to diagnose the problem, type:")
    print("*** `mpirun -np 2 python -c `from mpi4py.MPI import COMM_WORLD`")

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
