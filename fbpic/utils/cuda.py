# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of generic functions that operate on a GPU.
"""
import warnings
import numba
numba_version = (int(numba.__version__.split('.')[0]),
                 int(numba.__version__.split('.')[1]))
from numba import cuda
import numpy as np

# Check if CUDA is available and set variable accordingly
try:
    numba_cuda_installed = cuda.is_available()
except Exception:
    numba_cuda_installed = False

if numba_cuda_installed:
    # Infer if GPU is P100, V100, A100 or other
    if "P100" in str(cuda.gpus[0]._device.name):
        cuda_gpu_model = "P100"
    elif "V100" in str(cuda.gpus[0]._device.name):
        cuda_gpu_model = "V100"
    elif "A100" in str(cuda.gpus[0]._device.name):
        cuda_gpu_model = "V100" # force to V100
    else:
        cuda_gpu_model = "other"

try:
    import cupy
    cupy_installed = cupy.is_available()
    cupy_version = (int(cupy.__version__.split('.')[0]),
                    int(cupy.__version__.split('.')[1]))
except (ImportError, AssertionError):
    cupy_installed = False
    cupy_version = None

cuda_installed = (numba_cuda_installed and cupy_installed)

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
        Threads per block.
    """
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
        Threads per block in x and y.
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

class GpuMemoryManager(object):
    """
    Context manager that temporarily moves the simulation data to the GPU,
    if the data is originally on the CPU when entering the context manager
    """

    def __init__(self, simulation):
        """
        Initialize the context manager

        Parameters:
        -----------
        simulation: object
            A simulation object that contains the particle
            (ptcl) and field object (fld)
        """
        # Check whether the data is initially on the CPU or GPU
        self.fields_were_on_gpu = simulation.fld.data_is_on_gpu
        self.species_were_on_gpu = [ species.data_is_on_gpu \
                                     for species in simulation.ptcl ]
        # Keep a reference to the simulation
        self.sim = simulation

    def __enter__(self):
        """
        Move the data to the GPU (if it was originally on the CPU)
        """
        if self.sim.use_cuda:
            if not self.fields_were_on_gpu:
                self.sim.fld.send_fields_to_gpu()
            for i, species in enumerate(self.sim.ptcl):
                if not self.species_were_on_gpu[i]:
                    species.send_particles_to_gpu()

    def __exit__(self, type, value, traceback):
        """
        Move the data back to the CPU (if it was originally on the CPU)
        """
        if self.sim.use_cuda:
            if not self.fields_were_on_gpu:
                self.sim.fld.receive_fields_from_gpu()
            for i, species in enumerate(self.sim.ptcl):
                if not self.species_were_on_gpu[i]:
                    species.receive_particles_from_gpu()


# -----------------------------------------------------
# CUDA mpi management
# -----------------------------------------------------

def get_uuid(gpu_id):
    """
    Returns the UUID of a GPU device (or None if it cannot determine it)

    Parameters:
    -----------
    gpu_id: Local device id of the GPU (int)

    Returns:
    --------
    uuid: Unique identifier (UUID) of the GPU (str)
    """
    # For cupy version below 8.1, we cannot determine the uuid
    if cupy_version < (8,1):
        return None

    # Get UUID using cupy
    uuid = cupy.cuda.runtime.getDeviceProperties(gpu_id)['uuid']
    # conversion strategy from numba PR #6700
    b = '%02x'
    b2 = b * 2
    b4 = b * 4
    b6 = b * 6
    fmt = f'GPU-{b4}-{b2}-{b2}-{b2}-{b6}'
    return fmt % tuple(bytes(uuid))


def check_consecutive_ranks_on_same_nodes(mpi):
    """
    Check that consecutive MPI ranks are on the same nodes, and
    return a corresponding boolean. Print a warning message if
    it is not the case, this this can affect performance.
    """
    # Skip this function if there is only one MPI rank
    if mpi.COMM_WORLD.size == 1:
        return True

    # Check that the decomposition is standard, raise a warning otherwise
    nodes = mpi.COMM_WORLD.gather( mpi.Get_processor_name(), root=0 )
    standard_decomp = None
    if mpi.COMM_WORLD.rank == 0:
        # Check that consecutive MPI ranks are on the same node
        standard_decomp = True
        unique_nodes = [ nodes[0] ]  # Initialize list of unique nodes
        for i in range(1, len(nodes)):
            if nodes[i] != nodes[i-1]: # Consecutive ranks not on the same node
                if nodes[i] not in unique_nodes:
                    unique_nodes.append( nodes[i] )
                else:
                    # This node was seen before ; this means that several
                    # MPI ranks selected it, but that they are not consecutive.
                    standard_decomp = False
                    break
        # Print a corresponding warning
        if not standard_decomp:
            warnings.warn(
            "It seems that the distribution of MPI ranks on compute nodes is such\n"
            "that consecutive MPI ranks are not located on the same node.\n"
            "(See the FBPIC output with `verbose_level = 2` for more details.)\n"
            "This type of MPI distribution can degrade the performance of FBPIC.\n"
            "Please check the options of your MPI launcher (e.g. `mpirun`, `srun`)\n"
            "in order to use a different MPI distribution.")
    standard_decomp = mpi.COMM_WORLD.bcast( standard_decomp, root=0 )
    return standard_decomp


def mpi_select_gpus(mpi):
    """
    Selects the correct GPU used by the current MPI process
    using the MPI rank

    Parameters :
    ------------
    mpi: an mpi4py.MPI object
    """
    std_decomp = check_consecutive_ranks_on_same_nodes(mpi)
    if not std_decomp:
        # Convert node name to integer, and create a local communicator
        # for MPI ranks that are on the same node
        node_name = mpi.Get_processor_name()
        color = int.from_bytes( node_name.encode(), 'little') % 100000000
        comm = mpi.COMM_WORLD.Split(color=color)
    else:
        comm = mpi.COMM_WORLD

    # Attribute a GPU to each rank
    n_gpus = len(cuda.gpus)
    rank = comm.rank
    for i_gpu in range(n_gpus):
        if rank%n_gpus == i_gpu:
            cuda.select_device(i_gpu)
            uuid = get_uuid(i_gpu)
        mpi.COMM_WORLD.barrier()

    # Gather unique GPU identifiers
    uuids = mpi.COMM_WORLD.gather(uuid)

    # Check that no GPU was selected more than once
    if rank == 0:
        if not (None in uuids) and (len(uuids) > len(set(uuids))):
            warnings.warn(
            "GPUs have been oversubscribed by MPI ranks.\n"
            "This means that the same GPU was selected by more than one "
            "parallel process,\nwhich will result in poor performance.\n"
            "(See the FBPIC output with `verbose_level = 2` for more details.)")


# -----------------------------------------------------
# CUDA kernel decorator
# -----------------------------------------------------

if cuda_installed:

    def get_args_hash(args):
        """
        Computes a hash from the argument types of a kernel call.
        This takes into account the data types as well as (for arrays) the
        number of dimensions.

        Parameters:
        -----------
        args: A list of arguments (scalars or Cupy arrays).

        Returns:
        --------
        hash: Hash value as an int.
        """
        types = []

        # Loop over the arguments
        for a in args:
            # For array arguments: save the data type and the number of
            # dimensions
            if isinstance(a, cupy.ndarray):
                types.append(a.dtype)
                types.append(a.ndim)

            # For scalar arguments: save only the data type
            else:
                types.append(np.dtype(type(a)))

        # Use the built-in Python hash function to compute the hash
        return hash(tuple(types))

    class compile_cupy(object):
        """
        This class defines a custom function decorator which compiles python
        functions into GPU CUDA kernels. This decorator is meant to be used
        in the same way as `numba.cuda.jit` but has lower kernel launch
        overheads.

        In practice, this is achieved by using `numba.cuda.jit` to compile the
        kernels into PTX code, and by launching the PTX code through the `cupy`
        framework (which has lower launch overhead than the `numba.cuda`
        framework).

        Similarly to `numba.cuda.jit`, this class decorates functions that
        are defined with arbitrary argument types. The actual types of the
        arguments is only known when the function is called, at which point
        the corresponding CUDA kernel is compiled (Just-In-Time compilation).
        This decorator stores previously-compiled kernels, so as to avoid
        re-compiling if the function is called several times with the same
        argument types.
        """

        def __init__(self, func):
            """
            Constructor of the decorator class.

            Parameters:
            -----------
            func: The python function the decorator is applied to, which will
                be compiled into a CUDA kernel.
            """

            self.python_func = func
            self.kernel_dict = {} # Stores compiled kernels to avoid re-compilation

            # Flag to save whether the kernel has been explicitly specialized
            self.is_specialized = False

        def specialize(self, signature):
            """
            Specialize a kernel for an explicit function signature. The kernel
            is then compiled immediately.

            Parameters:
            -----------
            signature: The signature of the kernel, in numba format.
            """

            # Compile a Numba kernel for the given signature
            # using cuda.jit
            numba_kernel = cuda.jit(signature)(self.python_func)

            # Create a Cupy kernel module and load the PTX code of the
            # numba kernel
            module = cupy.cuda.function.Module()
            module.load(bytes(numba_kernel.ptx, 'UTF-8'))

            # Save the resulting Cupy kernel
            if numba_version[1] > 50:
                kernel_name = numba_kernel.definition.entry_name
            else:
                kernel_name = numba_kernel.entry_name
            self.specialized_kernel = module.get_function( kernel_name )
            self.is_specialized = True

            return self

        def __getitem__(self, bt):
            """
            Called when the kernel is called with square brackets, e.g.
            ```
            kernel[blocks_per_grid, threads_per_block]( *args )
            ```
            This is used to mimic the Numba launch syntax.

            Parameters:
            -----------
            bt: A 2-tuple (blocks_per_grid, threads_per_block) giving the
                thread and block size on the GPU.
                Both blocks_per_grid and threads_per_block should themselves
                be tuples, even in the 1D case.

            Returns:
            --------
            call_kernel: A wrapper function which represents the kernel
                specialized to the specified thread and block size, and
                which can then be called with the kernel arguments.
            """
            blocks_per_grid = bt[0]
            threads_per_block = bt[1]

            # Cast the thread and block size to tuples if neccessary
            # since Cupy does not accept them as simple numbers
            if not isinstance(blocks_per_grid, tuple):
                blocks_per_grid = (blocks_per_grid, )

            if not isinstance(threads_per_block, tuple):
                threads_per_block = (threads_per_block, )

            # Define function that will be returned by the decorator
            def call_kernel(*args):
                """
                Wrapper function for the actual kernel call. Checks if a
                kernel for the specified argument types is already compiled,
                and compiles one if needed. Then calls the kernel.

                Parameters:
                -----------
                args: List of the kernel arguments.
                    They should all be either scalar values (float, int,
                    complex, bool) or Cupy arrays.
                """

                # For explicitly specialized, do not worry about the argument types
                if self.is_specialized:
                    kernel = self.specialized_kernel

                else:
                    # Calculate a hash from the argument types to check whether a
                    # compatible kernel is already compiled.
                    hash = get_args_hash(args)

                    if hash not in self.kernel_dict:

                        # Compile a Numba kernel for the specified arguments
                        # using cuda.jit
                        numba_kernel = cuda.jit()(self.python_func) \
                            .specialize(*args)

                        # Create a Cupy kernel module and load the PTX code of the
                        # numba kernel
                        module = cupy.cuda.function.Module()
                        module.load(bytes(numba_kernel.ptx, 'UTF-8'))

                        # Cache the resulting Cupy kernel in a dictionary using
                        # the hash
                        if numba_version[1] > 50:
                            kernel_name = numba_kernel.definition.entry_name
                        else:
                            kernel_name = numba_kernel.entry_name
                        self.kernel_dict[hash] = module.get_function( kernel_name )

                    # Get the correct kernel from the cache
                    kernel = self.kernel_dict[hash]

                # Prepare the arguments for the Cupy kernel.
                # Because of the way Numba JIT compilation works, the
                # resulting kernels expect multiple arguments for each array.
                kernel_args = []

                # Loop over the given arguments
                for a in args:

                    # Check whether the argument is an array and requires
                    # multiple kernel arguments.
                    if isinstance(a, cupy.ndarray):
                        # Append all required arguments to the list, in order:
                        # - Two zeroes (corresponding to null pointers in C)
                        # - The total size of the array
                        # - The size in bytes of the array datatype
                        # - The array itself
                        # - The shape of the array, as single integers
                        # - The strides of the array, as single integers
                        # Note that due to the latter two entries, the actual
                        # number of arguments per array depends on the number
                        # of array dimensions.
                        kernel_args.extend(
                            [0, 0, a.size, a.dtype.itemsize, a])
                        kernel_args.extend(a.shape)
                        kernel_args.extend(a.strides)
                    else:
                        # For scalar arguments, simply append the
                        # argument itself.
                        kernel_args.append(a)

                # Call the actual kernel.
                # The arguments of the call are:
                # - Blocks per grid (tuple)
                # - Threads per blocks (tuple)
                # - The prepared list of kernel arguments
                kernel (blocks_per_grid, threads_per_block, kernel_args)

            # __getitem__ returns the created wrapper method.
            return call_kernel
