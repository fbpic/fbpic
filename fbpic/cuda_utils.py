# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of generic functions that operate on a GPU.
"""
from numba import cuda

# Check if CUDA is available and set variable accordingly
try:
    cuda_installed = cuda.is_available()
except Exception:
    cuda_installed = False

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
        Threads per block
    """
    # Calculates the needed blocks per grid
    BPG = int(x/TPB + 1)
    return BPG, TPB

def cuda_tpb_bpg_2d(x, y, TPBx = 8, TPBy = 8):
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
        Threads per block in x and y
    """
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
# CUDA information
# -----------------------------------------------------

def print_gpu_meminfo(gpu):
    """
    Prints memory information about the GPU.

    Parameters :
    ------------
    gpu : object
        A numba cuda gpu context object.
    """
    with gpu:
        meminfo = cuda.current_context().get_memory_info()
        print("GPU: %s, free: %s Mbytes, total: %s Mbytes \
              " % (gpu, meminfo[0]*1e-6, meminfo[1]*1e-6))

def print_available_gpus():
    """
    Lists all available CUDA GPUs.
    """
    cuda.detect()

def print_gpu_meminfo_all():
    """
    Prints memory information about all available CUDA GPUs.
    """
    gpus = cuda.gpus.lst
    for gpu in gpus:
        print_gpu_meminfo(gpu)

def print_current_gpu( mpi ):
    """
    Prints information about the currently selected GPU.

    Parameter:
    ----------
    mpi: an mpi4py.MPI object
    """
    gpu = cuda.gpus.current
    rank = mpi.COMM_WORLD.rank
    node = mpi.Get_processor_name()
    message = "MPI rank %d selected a %s GPU with id %s on node %s" %(
        rank, gpu.name, gpu.id, node)
    print(message)

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

    print_current_gpu( mpi )
