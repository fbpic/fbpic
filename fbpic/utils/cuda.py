# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of generic functions that operate on a GPU.
"""
import os
import numba
from numba import cuda

# Check if CUDA is available and set variable accordingly
try:
    cuda_installed = cuda.is_available()
except Exception:
    cuda_installed = False

if cuda_installed:
    # Infer if GPU is P100 or V100 or other
    if "P100" in str(cuda.gpus[0]._device.name):
        cuda_gpu_model = "P100"
    elif "V100" in str(cuda.gpus[0]._device.name):
        cuda_gpu_model = "V100"
    else:
        cuda_gpu_model = "other"

try:
    import cupy
    cupy_installed = cupy.is_available()
    assert int(cupy.__version__[0]) >= 6 # Require cupy version 6
except Exception:
    cupy_installed = False

# Check for CUDA threads per block environment variables
if 'FBPIC_1D_TPB' in os.environ:
    FBPIC_1D_TPB = int(os.environ['FBPIC_1D_TPB'])
else:
    FBPIC_1D_TPB = None
if 'FBPIC_2D_TPBX' in os.environ:
    FBPIC_2D_TPBX = int(os.environ['FBPIC_2D_TPBX'])
else:
    FBPIC_2D_TPBX = None
if 'FBPIC_2D_TPBY' in os.environ:
    FBPIC_2D_TPBY = int(os.environ['FBPIC_2D_TPBY'])
else:
    FBPIC_2D_TPBY = None

# -----------------------------------------------------
# CUDA grid utilities
# -----------------------------------------------------

def cuda_tpb_bpg_1d(x, TPB = 256):
    """
    Get the needed blocks per grid for a 1D CUDA grid.

    Parameters :
    ------------
    x : int
        Total number of threads

    TPB : int
        Threads per block

    Returns :
    ---------
    BPG : int
        Number of blocks per grid

    TPB : int
        Threads per block. Can also be set via
        FBPIC_1D_TPB environment variable.
    """
    # Get environment TBP if set
    TPB = FBPIC_1D_TPB if FBPIC_1D_TPB else TPB
    # Calculates the needed blocks per grid
    BPG = int(x/TPB + 1)
    return BPG, TPB

def cuda_tpb_bpg_2d(x, y, TPBx = 1, TPBy = 128):
    """
    Get the needed blocks per grid for a 2D CUDA grid.

    Parameters :
    ------------
    x, y  : int
        Total number of threads in first and second dimension

    TPBx, TPBy : int
        Threads per block in x and y

    Returns :
    ------------
    (BPGx, BPGy) : tuple of ints
        Number of blocks per grid in x and y

    (TPBx, TPBy) : tuple of ints
        Threads per block in x and y. Can also be set via
        FBPIC_2D_TPBX and FBPIC_2D_TPBY environment variable.
    """
    # Get environment TBP if set
    TPBx = FBPIC_2D_TPBX if FBPIC_2D_TPBX else TPBx
    TPBy = FBPIC_2D_TPBY if FBPIC_2D_TPBY else TPBy
    # Calculates the needed blocks per grid
    BPGx = int(x/TPBx + 1)
    BPGy = int(y/TPBy + 1)
    return (BPGx, BPGy), (TPBx, TPBy)

# -----------------------------------------------------
# CUDA memory management
# -----------------------------------------------------

def send_data_to_gpu(simulation):
    """
    Send the simulation data to the GPU.
    Calls the functions of the particle and field package
    that send the data to the GPU.

    Parameters :
    ------------
    simulation : object
        A simulation object that contains the particle
        (ptcl) and field object (fld)
    """
    # Send particles to the GPU (if CUDA is used)
    for species in simulation.ptcl :
        if species.use_cuda:
            species.send_particles_to_gpu()
    # Send fields to the GPU (if CUDA is used)
    simulation.fld.send_fields_to_gpu()

def receive_data_from_gpu(simulation):
    """
    Receive the simulation data from the GPU.
    Calls the functions of the particle and field package
    that receive the data from the GPU.

    Parameters :
    ------------
    simulation : object
        A simulation object that contains the particle
        (ptcl) and field object (fld)
    """
    # Receive the particles from the GPU (if CUDA is used)
    for species in simulation.ptcl :
        if species.use_cuda:
            species.receive_particles_from_gpu()
    # Receive fields from the GPU (if CUDA is used)
    simulation.fld.receive_fields_from_gpu()

# -----------------------------------------------------
# CUDA mpi management
# -----------------------------------------------------

def mpi_select_gpus(mpi):
    """
    Selects the correct GPU used by the current MPI process

    Parameters :
    ------------
    mpi: an mpi4py.MPI object
    """
    n_gpus = len(cuda.gpus)
    rank = mpi.COMM_WORLD.rank
    for i_gpu in range(n_gpus):
        if rank%n_gpus == i_gpu:
            cuda.select_device(i_gpu)
        mpi.COMM_WORLD.barrier()
