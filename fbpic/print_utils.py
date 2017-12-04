# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters, Soeren Jalas
# License: 3-Clause-BSD-LBNL
"""
Fourier-Bessel Particle-In-Cell (FB-PIC) main file
It defines a set of generic functions for printing simulation information.
"""
import sys, time
import numba
from numba import cuda
from fbpic import __version__
# Check availability of various computational setups
from fbpic.fields.spectral_transform.fourier import mkl_installed
from fbpic.cuda_utils import cuda_installed
from fbpic.mpi_utils import MPI, mpi_installed
from fbpic.threading_utils import threading_enabled

def print_simulation_setup( sim, level=1 ):
    """
    Print information about the simulation.
    - Version of FBPIC
    - CPU or GPU computation
    - Number of parallel MPI domains
    - Number of threads in case of CPU multi-threading
    - (Additional detailed information)

    Parameters
    ----------
    sim: an fbpic Simulation object
        Contains all the information of the simulation setup

    level: int, optional
        Level of detail of the simulation information
        0 - Print no information
        1 (Default) - Print basic information
        2 - Print detailed information
    """
    if level > 0 and sim.comm.rank == 0:
        # Print version of FBPIC
        message = '\nFBPIC (fbpic-%s)\n'%__version__
        # Basic information
        if level == 1:
            # Print information about computational setup
            if sim.use_cuda:
                message += "\nRunning on GPU "
            else:
                message += "\nRunning on CPU "
            if sim.comm.size > 1:
                message += "with %d MPI processes " %sim.comm.size
            if sim.use_threading and not sim.use_cuda:
                message += "(%d threads per process) " \
                    %numba.config.NUMBA_NUM_THREADS
        # Detailed information
        if level == 2:
                if mpi_installed:
                    message += '\nMPI available: Yes'
                    message += '\nMPI processes: %d' %sim.comm.size
                else:
                    message += '\nMPI available: No'
                if cuda_installed:
                    message += '\nCUDA available: Yes'
                else:
                    message += '\nCUDA available: No'
                if sim.use_cuda:
                    message += '\nCompute architecture: GPU (CUDA)'
                else:
                    message += '\nCompute architecture: CPU'
                    if threading_enabled:
                        message += '\nCPU multi-threading enabled: Yes'
                        message += '\nThreads: %s' \
                            %numba.config.NUMBA_NUM_THREADS
                    else:
                        message += '\nCPU multi-threading enabled: No'
                    if mkl_installed:
                        message += '\nFFT library: MKL'
                    else:
                        message += '\nFFT library: pyFFTW'

                message += '\n'
                if sim.fld.n_order == -1:
                    message += '\nPSAOTD stencil order (accuracy): infinite'
                else:
                    message += '\nPSAOTD stencil order (accuracy): %d' \
                        %sim.fld.n_order
                message += '\nParticle shape: %s' %sim.particle_shape
                message += '\nLongitudinal boundaries: %s' %sim.comm.boundaries
                message += '\nTransverse boundaries: reflective'
                message += '\nGuard region size: %d ' \
                    %sim.comm.n_guard + 'cells'
                message += '\nDamping region size: %d ' \
                    %sim.comm.n_damp + 'cells'
                message += '\nParticle exchange period: every %d ' \
                    %sim.comm.exchange_period + 'step'
                if sim.gamma_boost is not None:
                    message += '\nBoosted frame: Yes'
                    message += '\nBoosted frame gamma: %d' \
                        %sim.comm.gamma_boost
                    if sim.use_galilean:
                        message += '\nGalilean frame: Yes'
                    else:
                        message += '\nGalilean frame: No'
                else:
                    message += '\nBoosted frame: False'
        message += '\n'    
        print( message )
    if level == 2:
        # Sync MPI processes before MPI GPU selection
        sim.comm.mpi_comm.barrier()
        time.sleep(0.1)
        if sim.use_cuda:
            print_current_gpu( MPI )
            if sim.comm.rank == 0:
                print('')

def progression_bar( i, Ntot, avg_time_per_step, prev_time,
                     n_avg=20, Nbars=35, char=u'\u007C'):
    """
    Shows a progression bar with Nbars and the estimated
    remaining simulation time.
    """
    # Estimate average time per step
    curr_time = time.time()
    time_per_step = curr_time - prev_time
    avg_time_per_step += (time_per_step - avg_time_per_step)/n_avg
    if i <= 2:
        # Ignores first step in time estimation (compilation time)
        avg_time_per_step = time_per_step
    # Print progress bar
    if i == 0:
        # Let the user know that the first step is much longer
        sys.stdout.write('\r' + '1st iteration & ' + \
            'Just-In-Time compilation (up to one minute) ...')
        sys.stdout.flush()
    else:
        # Print the progression bar
        nbars = int( (i+1)*1./Ntot*Nbars )
        sys.stdout.write('\r' + nbars*char )
        sys.stdout.write((Nbars-nbars)*' ')
        sys.stdout.write(' %d/%d' %(i+1, Ntot))
        sys.stdout.write(', %4d ms/step' %(time_per_step*1.e3))
        if i < n_avg:
            # Time estimation is only printed after n_avg timesteps
            sys.stdout.write(', calc. ETA...')
            sys.stdout.flush()
        else:
            # Estimated time in seconds until it will finish
            eta = avg_time_per_step*(Ntot-i)
            # Conversion to H:M:S
            m, s = divmod(eta, 60)
            h, m = divmod(m, 60)
            sys.stdout.write(', %d:%02d:%02d left' % (h, m, s))
            sys.stdout.flush()
    # Clear line
    sys.stdout.write('\033[K')

    return avg_time_per_step, curr_time

def print_runtime_summary( N, duration ):
    """
    Print a summary about the total runtime of the simulation.

    Parameters
    ----------
    N: int
        The total number of iterations performed by the step loop

    duration: float, seconds
        The total time taken by the step loop
    """
    avg_tps = (duration / N)*1.e3
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    print('\nTime taken (with compilation): %d:%02d:%02d' %(h, m, s))
    print('Average time per iteration ' \
          '(with compilation): %d ms\n' %(avg_tps))

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
    # Convert bytestring to actual string
    try:
        gpu_name = gpu.name.decode()
    except AttributeError:
        gpu_name = gpu.name
    # Print the GPU that is being used
    if mpi.COMM_WORLD.size > 1:
        rank = mpi.COMM_WORLD.rank
        node = mpi.Get_processor_name()
        message = "MPI rank %d selected a %s GPU with id %s on node %s" %(
            rank, gpu_name, gpu.id, node)
    else:
        message = "FBPIC selected a %s GPU with id %s" %( gpu_name, gpu.id )
    print(message)
