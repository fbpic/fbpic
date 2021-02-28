# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters, Soeren Jalas
# License: 3-Clause-BSD-LBNL
"""
Fourier-Bessel Particle-In-Cell (FB-PIC) main file
It defines a set of generic functions for printing simulation information.
"""
import sys, time
from fbpic import __version__
from fbpic.utils.cuda import cuda, cuda_installed
from fbpic.utils.mpi import MPI, mpi_installed, gpudirect_enabled
# Check if terminal is correctly set to UTF-8 and set progress character
if sys.stdout.encoding == 'UTF-8':
    progress_char = u'\u2588'
else:
    progress_char = '-'

class ProgressBar(object):
    """
    ProgressBar class that keeps track of the time spent by the algorithm.
    It handles the calculation and printing of the progress bar and a
    summary of the total runtime.
    """

    def __init__(self, N, n_avg=20, Nbars=35, char=progress_char):
        """
        Initializes a timer / progression bar.
        Timing is done with respect to the absolute time at initialization.

        Parameters
        ----------
        N: int
            The total number of iterations performed by the step loop

        n_avg: int, optional
            The amount of recent timesteps used to calculate the average
            time taken by a step

        Nbar: int, optional
            The number of bars printed for the progression bar

        char: str, optional
            The character used to show the progression.
        """
        self.N = N
        self.n_avg = n_avg
        self.Nbars = Nbars
        self.bar_char = char

        # Initialize variables to measure the time taken by the simulation
        self.i_step = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.total_duration = 0.
        self.time_per_step = 0.
        self.avg_time_per_step = 0.
        self.eta = None

    def time( self, i_step ):
        """
        Calculates the time taken by the last iterations, the average time
        taken by the most recent iterations and the estimated remaining
        simulation time.

        Parameters
        ----------
        i_step : int
            The current iteration of the loop
        """
        # Register current step
        self.i_step = i_step
        # Calculate time taken by last step
        curr_time = time.time()
        self.total_duration = curr_time - self.start_time
        self.time_per_step = curr_time - self.prev_time
        # Estimate average time per step
        self.avg_time_per_step += \
            (self.time_per_step - self.avg_time_per_step)/self.n_avg
        if self.i_step <= 2:
            # Ignores first step in time estimation (compilation time)
            self.avg_time_per_step = self.time_per_step
        # Estimated time in seconds until it will finish
        if self.i_step < self.n_avg:
            self.eta = None
        else:
            self.eta = self.avg_time_per_step*(self.N-self.i_step)
        # Advance the previous time to the current time
        self.prev_time = curr_time

    def print_progress( self ):
        """
        Prints a progression bar with the estimated
        remaining simulation time and the time taken by the last step.
        """
        i = self.i_step
        # Print progress bar
        if i == 0:
            # Let the user know that the first step is much longer
            sys.stdout.write('\r' + \
                'Just-In-Time compilation (up to one minute) ...')
            sys.stdout.flush()
        else:
            # Print the progression bar
            nbars = int( (i+1)*1./self.N*self.Nbars )
            sys.stdout.write('\r|' + nbars*self.bar_char )
            sys.stdout.write((self.Nbars-nbars)*' ')
            sys.stdout.write('| %d/%d' %(i+1,self.N))
            if self.eta is None:
                # Time estimation is only printed after n_avg timesteps
                sys.stdout.write(', calc. ETA...')
            else:
                # Conversion to H:M:S
                m, s = divmod(self.eta, 60)
                h, m = divmod(m, 60)
                sys.stdout.write(', %d:%02d:%02d left' % (h, m, s))
            # Time taken by the last step
            sys.stdout.write(', %d ms/step' %(self.time_per_step*1.e3))
            sys.stdout.flush()
        # Clear line
        sys.stdout.write('\033[K')

    def print_summary( self ):
        """
        Print a summary about the total runtime of the simulation.
        """
        avg_tps = (self.total_duration / self.N)*1.e3
        m, s = divmod(self.total_duration, 60)
        h, m = divmod(m, 60)
        print('\nTotal time taken (with compilation): %d:%02d:%02d' %(h, m, s))
        print('Average time per iteration ' \
              '(with compilation): %d ms\n' %(avg_tps))

# -----------------------------------------------------
# Print utilities
# -----------------------------------------------------

def print_simulation_setup( sim, verbose_level=1 ):
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

    verbose_level: int, optional
        Level of detail of the simulation information
        0 - Print no information
        1 (Default) - Print basic information
        2 - Print detailed information
    """
    if verbose_level > 0:
        # Print version of FBPIC
        message = '\nFBPIC (%s)\n'%__version__
        # Basic information
        if verbose_level == 1:
            # Print information about computational setup
            if sim.use_cuda:
                message += "\nRunning on GPU "
            else:
                message += "\nRunning on CPU "
            if sim.comm.size > 1:
                message += "with %d MPI processes " %sim.comm.size
            if sim.use_threading and not sim.use_cuda:
                message += "(%d threads per process) " %sim.cpu_threads
        # Detailed information
        elif verbose_level == 2:
            # Information on MPI
            if mpi_installed:
                message += '\nMPI available: Yes'
                message += '\nMPI processes used: %d' %sim.comm.size
                message += '\nMPI Library Information: \n%s' \
                    %MPI.Get_library_version()
            else:
                message += '\nMPI available: No'
            # Information on Cuda
            if cuda_installed:
                message += '\nCUDA available: Yes'
            else:
                message += '\nCUDA available: No'
            # Information about the architecture and the node used
            if sim.use_cuda:
                message += '\nCompute architecture: GPU (CUDA)'
                if mpi_installed:
                    if gpudirect_enabled:
                        message += '\nCUDA GPUDirect (MPI) enabled: Yes'
                    else:
                        message += '\nCUDA GPUDirect (MPI) enabled: No'
                node_message = get_gpu_message()
            else:
                message += '\nCompute architecture: CPU'
                if sim.use_threading:
                    message += '\nCPU multi-threading enabled: Yes'
                    message += '\nThreads: %s' %sim.cpu_threads
                else:
                    message += '\nCPU multi-threading enabled: No'
                if sim.fld.trans[0].fft.use_mkl:
                    message += '\nFFT library: MKL'
                else:
                    message += '\nFFT library: pyFFTW'
                node_message = get_cpu_message()
            # Gather the information about where each node runs
            if sim.comm.size > 1:
                node_messages = sim.comm.mpi_comm.gather( node_message )
                if sim.comm.rank == 0:
                    node_message = ''.join( node_messages )
            message += node_message

            message += '\n'
            # Information on the numerical algorithm
            if sim.fld.n_order == -1:
                message += '\nPSATD stencil order: infinite'
            else:
                message += '\nPSATD stencil order: %d' %sim.fld.n_order
            message += '\nParticle shape: %s' %sim.particle_shape
            message += '\nLongitudinal boundaries: %s' %sim.comm.boundaries['z']
            message += '\nTransverse boundaries: %s' %sim.comm.boundaries['r']
            message += '\nGuard region size: %d ' %sim.comm.n_guard+'cells'
            message += '\nDamping region size: %d ' %sim.comm.nz_damp+'cells'
            message += '\nInjection region size: %d ' %sim.comm.n_inject+'cells'
            message += '\nParticle exchange period: every %d ' \
                %sim.comm.exchange_period + 'step'
            if sim.boost is not None:
                message += '\nBoosted frame: Yes'
                message += '\nBoosted frame gamma: %d' %sim.boost.gamma0
                if sim.use_galilean:
                    message += '\nGalilean frame: Yes'
                else:
                    message += '\nGalilean frame: No'
            else:
                message += '\nBoosted frame: False'
        message += '\n'

        # Only processor 0 prints the message:
        if sim.comm.rank == 0:
            print( message )

def print_available_gpus():
    """
    Lists all available CUDA GPUs.
    """
    cuda.detect()

def get_gpu_message():
    """
    Returns a string with information about the currently selected GPU.
    """
    gpu = cuda.gpus.current
    # Convert bytestring to actual string
    try:
        gpu_name = gpu.name.decode()
    except AttributeError:
        gpu_name = gpu.name
    # Print the GPU that is being used
    if MPI.COMM_WORLD.size > 1:
        rank = MPI.COMM_WORLD.rank
        node = MPI.Get_processor_name()
        message = "\nMPI rank %d selected a %s GPU with id %s on node %s" %(
            rank, gpu_name, gpu.id, node)
    else:
        message = "\nFBPIC selected a %s GPU with id %s" %( gpu_name, gpu.id )
        if mpi_installed:
            node = MPI.Get_processor_name()            
            message += " on node %s" %node
    return(message)

def get_cpu_message():
    """
    Returns a string with information about the node of each MPI rank
    """
    # Print the node that is being used
    if MPI.COMM_WORLD.size > 1:
        rank = MPI.COMM_WORLD.rank
        node = MPI.Get_processor_name()
        message = "\nMPI rank %d runs on node %s" %(rank, node)
    else:
        message = ""
    return(message)

def print_gpu_meminfo_all():
    """
    Prints memory information about all available CUDA GPUs.
    """
    gpus = cuda.gpus.lst
    for gpu in gpus:
        print_gpu_meminfo(gpu)

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

def catch_gpu_memory_error( f ):
    """
    Decorator that calls the function `f` and catches any GPU memory
    error, during the execution of f.

    If a memory error occurs, this decorator prints a corresponding message
    and aborts the simulation (using MPI abort if needed)
    """
    # Redefine the original function by calling it within a try/except
    def g(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except cuda.cudadrv.driver.CudaAPIError as e:
            handle_cuda_memory_error( e, f.__name__ )
    # Decorator: return the new function
    return(g)

def handle_cuda_memory_error( exception, function_name ):
    """
    Print a message indicating which GPU went out of memory,
    and abort the simulation (using MPI Abort if needed)
    """
    # Print a useful message
    message = '\nERROR: GPU reached OUT_OF_MEMORY'
    if MPI.COMM_WORLD.size > 1:
        message += ' on MPI rank %d' %MPI.COMM_WORLD.rank
    message += '\n(Error occured in fbpic function `%s`)\n' %function_name
    sys.stdout.write(message)
    sys.stdout.flush()
    # Abort the simulation
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort()
    else:
        raise( exception )
