# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of generic functions that operate on a GPU.
"""
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import warnings
import numba
from numba import cuda
import numpy as np
from fbpic.utils.cuda_tuning import select_cuda_launch_config

numba_version = (int(numba.__version__.split('.')[0]),
                 int(numba.__version__.split('.')[1]),
                 int(numba.__version__.split('.')[2]))

# Check if CUDA is available and set variable accordingly
try:
    numba_cuda_installed = cuda.is_available()
except Exception:
    numba_cuda_installed = False


def legacy_cuda_gpu_model(device_or_name):
    """Return the deprecated tuning label from the selected device name."""
    name = getattr(device_or_name, "name", device_or_name)
    if isinstance(name, bytes):
        name = name.decode("utf-8", "replace")
    name = str(name).upper()
    if "P100" in name:
        return "P100"
    if "V100" in name or "A100" in name:
        return "V100"
    return "other"


def get_cuda_launch_config(device=None, environ=None):
    """Return the launch policy for the selected CUDA device."""
    if device is None:
        device = cuda.get_current_device()
    max_threads_per_block = getattr(device, "MAX_THREADS_PER_BLOCK", None)
    if max_threads_per_block is None:
        max_threads_per_block = device.max_threads_per_block
    warp_size = getattr(device, "WARP_SIZE", None)
    if warp_size is None:
        warp_size = device.warp_size
    return select_cuda_launch_config(
        device.compute_capability,
        max_threads_per_block=max_threads_per_block,
        warp_size=warp_size,
        environ=environ,
    )


class _LazyCudaGpuModel(object):
    """Resolve the deprecated compatibility label only when it is used."""

    def _value(self):
        return legacy_cuda_gpu_model(cuda.get_current_device())

    def __eq__(self, other):
        return self._value() == other

    def __ne__(self, other):
        return self._value() != other

    def __str__(self):
        return self._value()

    def __repr__(self):
        return repr(self._value())

    def __format__(self, format_spec):
        return format(self._value(), format_spec)

    def __hash__(self):
        return hash(self._value())

    def __getattr__(self, name):
        return getattr(self._value(), name)

    def __add__(self, other):
        return self._value() + other

    def __radd__(self, other):
        return other + self._value()

    def __contains__(self, item):
        return item in self._value()

    def __len__(self):
        return len(self._value())

    def __iter__(self):
        return iter(self._value())

    def __getitem__(self, item):
        return self._value()[item]

    def __nonzero__(self):
        return bool(self._value())

    __bool__ = __nonzero__


# Deprecated compatibility alias; internal tuning uses launch policies.  This
# proxy must stay lazy because importing this module precedes MPI GPU
# selection.
cuda_gpu_model = _LazyCudaGpuModel()


def refresh_cuda_gpu_model(device=None):
    """Expose the selected device's deprecated label as an actual string."""
    global cuda_gpu_model
    if device is None:
        device = cuda.get_current_device()
    cuda_gpu_model = legacy_cuda_gpu_model(device)
    return cuda_gpu_model

try:
    import cupy
    cupy_installed = cupy.is_available()
    cupy_version = (int(cupy.__version__.split('.')[0]),
                    int(cupy.__version__.split('.')[1]))
except (ImportError, AssertionError):
    cupy_installed = False
    cupy_version = None

cuda_installed = (numba_cuda_installed and cupy_installed)

_cuda_source_digest = None


def _get_cuda_source_digest():
    """Hash FBPIC Python sources that can contribute to generated PTX."""
    global _cuda_source_digest
    if _cuda_source_digest is None:
        package_root = Path(__file__).resolve().parent.parent
        digest = hashlib.sha256()
        for source_path in sorted(package_root.rglob("*.py")):
            digest.update(str(source_path.relative_to(package_root)).encode())
            digest.update(source_path.read_bytes())
        _cuda_source_digest = digest.hexdigest()
    return _cuda_source_digest


def _cuda_argument_descriptors(args):
    """Return stable Numba-specialization descriptors for kernel arguments."""
    descriptors = []
    for arg in args:
        if isinstance(arg, cupy.ndarray):
            descriptors.append((
                "array", arg.dtype.str, arg.ndim,
                bool(arg.flags.c_contiguous),
                bool(arg.flags.f_contiguous)))
        else:
            descriptors.append(("scalar", np.dtype(type(arg)).str))
    return tuple(descriptors)


def _installed_version(distribution):
    """Return an installed package version without making caching mandatory."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _nvvm_version():
    """Return the CUDA compiler-library version when it is discoverable."""
    try:
        from numba.cuda.cudadrv import nvvm
        return tuple(nvvm.get_version())
    except Exception:
        return None


def _cuda_cache_path(func, args):
    """Return the architecture- and signature-specific PTX cache path."""
    if os.environ.get("FBPIC_DISABLE_CUDA_KERNEL_CACHE") == "1":
        return None
    try:
        import scipy

        cache_root = os.environ.get("FBPIC_CUDA_KERNEL_CACHE_DIR")
        if cache_root is None:
            cache_root = Path.home() / ".cache" / "fbpic" / "cuda-kernels"
        else:
            cache_root = Path(cache_root).expanduser()

        device = cuda.get_current_device()
        identity = {
            "schema": 1,
            "function": "%s.%s" % (func.__module__, func.__qualname__),
            "arguments": _cuda_argument_descriptors(args),
            "compute_capability": list(device.compute_capability),
            "fbpic_source": _get_cuda_source_digest(),
            "numba": numba.__version__,
            "numba_cuda": _installed_version("numba-cuda"),
            "llvmlite": _installed_version("llvmlite"),
            "nvvm": _nvvm_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "cupy": cupy.__version__,
            "cuda_runtime": cupy.cuda.runtime.runtimeGetVersion(),
            "python": platform.python_version(),
        }
        key = hashlib.sha256(json.dumps(
            identity, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        return Path(cache_root) / (key + ".json")
    except Exception:
        return None


def _load_cuda_kernel_cache(cache_path):
    """Load a CuPy function from one atomic PTX cache record."""
    if cache_path is None:
        return None
    try:
        if not cache_path.is_file():
            return None
        record = json.loads(cache_path.read_text())
        if record.get("schema") != 1:
            return None
        module = cupy.cuda.function.Module()
        module.load(record["ptx"].encode("utf-8"))
        return module.get_function(record["entry_name"])
    except Exception:
        return None


def _store_cuda_kernel_cache(cache_path, ptx, entry_name):
    """Atomically store generated PTX; cache failures never stop a run."""
    if cache_path is None:
        return
    temporary_path = None
    try:
        cache_path.parent.mkdir(
            mode=0o700, parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                mode="w", dir=str(cache_path.parent), delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump({
                "schema": 1,
                "entry_name": entry_name,
                "ptx": ptx,
            }, stream, separators=(",", ":"))
        os.replace(str(temporary_path), str(cache_path))
    except OSError:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _store_compiled_cuda_kernel(cache_path, numba_kernel):
    """Best-effort extraction and storage after a usable kernel exists."""
    if cache_path is None:
        return
    try:
        if numba_version[1] >= 56:
            definition = next(iter(numba_kernel.overloads.values()))
            ptx = definition._codelibrary.get_asm_str()
            entry_name = definition.entry_name
        elif numba_version[1] >= 53:
            definition = next(iter(numba_kernel.overloads.values()))
            ptx = definition.ptx
            entry_name = definition.entry_name
        else:
            ptx = numba_kernel.ptx
            entry_name = numba_kernel.entry_name
        _store_cuda_kernel_cache(cache_path, ptx, entry_name)
    except Exception:
        pass


def _compile_cupy_ptx(func, args):
    """Try generating the usual PTX without binding an unused Numba binary.

    The dispatcher normally binds its kernel before returning a specialization.
    CuPy only needs the pre-link PTX, so constructing the same kernel
    definition suffices. Numba's private interface is optional: callers retain
    regular specialization if it changes or the module cannot be loaded.
    """
    if (numba_version < (0, 56, 1)
            or os.environ.get("FBPIC_DISABLE_CUDA_PTX_ONLY") == "1"):
        return None
    try:
        from numba.cuda.dispatcher import _Kernel, global_compiler_lock

        dispatcher = cuda.jit()(func)
        with global_compiler_lock:
            argtypes = tuple(dispatcher.typeof_pyval(arg) for arg in args)
            definition = _Kernel(func, argtypes, **dispatcher.targetoptions)
            ptx = definition._codelibrary.get_asm_str()
            entry_name = definition.entry_name
        module = cupy.cuda.function.Module()
        module.load(ptx.encode("utf-8"))
        kernel = module.get_function(entry_name)
        return kernel, ptx, entry_name
    except Exception:
        return None

try:
    import pynvml
    pynvml_installed = True
except ImportError:
    pynvml_installed = False

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
    if x < 0:
        raise ValueError("The work size must be nonnegative.")
    if TPB <= 0:
        raise ValueError("The block size must be positive.")

    # Calculates the needed blocks per grid. Keep one block for zero work,
    # since CUDA rejects launches with a zero-sized grid dimension.
    BPG = max(1, (x + TPB - 1) // TPB)
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
    if x < 0 or y < 0:
        raise ValueError("The work sizes must be nonnegative.")
    if TPBx <= 0 or TPBy <= 0:
        raise ValueError("The block sizes must be positive.")

    # Calculates the needed blocks per grid. Keep one block for zero work,
    # since CUDA rejects launches with a zero-sized grid dimension.
    BPGx = max(1, (x + TPBx - 1) // TPBx)
    BPGy = max(1, (y + TPBy - 1) // TPBy)
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
    Temporarily move CPU-resident simulation data to the GPU.

    On exit, this restores only the field and particle arrays that were on the
    CPU when the manager was created. This allows managers to be nested without
    an inner scope downloading data owned by an outer scope.
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
        Move data to the GPU if it was originally on the CPU.
        """
        if self.sim.use_cuda:
            if not self.fields_were_on_gpu:
                self.sim.fld.send_fields_to_gpu()
            for i, species in enumerate(self.sim.ptcl):
                if not self.species_were_on_gpu[i]:
                    species.send_particles_to_gpu()
        return self

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

    # Check the UUID length to prevent crashes
    if len(uuid) != 16:
        warnings.warn(f"Failed to detect UUID of GPU {gpu_id} (invalid UUID length: {len(uuid)})")
        return None
    
    # conversion strategy from numba PR #6700
    b = '%02x'
    b2 = b * 2
    b4 = b * 4
    b6 = b * 6
    fmt = f'GPU-{b4}-{b2}-{b2}-{b2}-{b6}'
    return fmt % tuple(bytes(uuid))

def get_uuid_alt(gpu_id):
    """
    Returns the UUID of a GPU device using `pynvml`.

    Parameters:
    -----------
    gpu_id: Local device id of the GPU (int)

    Returns:
    --------
    uuid: Unique identifier (UUID) of the GPU (str)
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    uuid = pynvml.nvmlDeviceGetUUID(handle)
    pynvml.nvmlShutdown()
    return uuid

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
            if pynvml_installed:
                uuid = get_uuid_alt(i_gpu)
            else:
                uuid = get_uuid(i_gpu)
        mpi.COMM_WORLD.barrier()

    # Device selection is complete. New imports should see the historical
    # module-level string, while already-imported proxy references stay lazy.
    refresh_cuda_gpu_model()

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
        return hash(_cuda_argument_descriptors(args))

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

        def make_cupy_kernel(self, numba_kernel):
            """
            Helper function to convert an already compiled numba kernel into
            a cupy kernel.

            Parameters:
            -----------
            numba_kernel: a numba kernel object.
            """

            # Create a Cupy kernel module and load the PTX code of the
            # numba kernel
            module = cupy.cuda.function.Module()
            if numba_version[1] >= 56:
                if (numba_version[1] == 56) and (numba_version[2] == 0):
                    raise RuntimeError(
                        'FBPIC is incompatible with numba 0.56.0.\n'
                        'Please install either a later or an earlier version.')
                kernel = next(iter(numba_kernel.overloads.values()))
                ptx = kernel._codelibrary.get_asm_str()
                module.load(bytes(ptx, 'UTF-8'))
            elif numba_version[1] >= 53:
                ptx = next(iter(numba_kernel.overloads.values())).ptx
                module.load(bytes(ptx, 'UTF-8'))
            else:
                module.load(bytes(numba_kernel.ptx, 'UTF-8'))

            # Extract the cupy kernel
            if numba_version[1] >= 53:
                definition = next(iter(numba_kernel.overloads.values()))
                kernel_name = definition.entry_name
            elif numba_version[1] > 50:
                kernel_name = numba_kernel.definition.entry_name
            else:
                kernel_name = numba_kernel.entry_name

            return module.get_function( kernel_name )

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

            # Convert the kernel into a cupy kernel
            self.specialized_kernel = self.make_cupy_kernel( numba_kernel )
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

                        cache_path = _cuda_cache_path(
                            self.python_func, args)
                        kernel = _load_cuda_kernel_cache(cache_path)
                        if kernel is None:
                            compiled = _compile_cupy_ptx(
                                self.python_func, args)
                            if compiled is not None:
                                kernel, ptx, entry_name = compiled
                                _store_cuda_kernel_cache(
                                    cache_path, ptx, entry_name)
                            else:
                                # Retain the regular dispatcher path for
                                # unsupported Numba interfaces or kernels.
                                numba_kernel = cuda.jit()(self.python_func) \
                                    .specialize(*args)
                                kernel = self.make_cupy_kernel(numba_kernel)
                                _store_compiled_cuda_kernel(
                                    cache_path, numba_kernel)

                        # Keep the loaded kernel alive for this process.
                        self.kernel_dict[hash] = kernel

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
