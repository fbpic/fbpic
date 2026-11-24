# Copyright 2016-2026, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
Unified GPU backend layer for FBPIC (NVIDIA CUDA and AMD ROCm/HIP).

Owns all vendor-specific detection, kernel compilation/caching, device
properties, wavefront-aware launch geometry, UUID resolution, MPI GPU
selection, and math-library namespaces (FFT, BLAS, Thrust).
"""
import hashlib
import json
import math
import os
import tempfile
import time
import types as py_types
import warnings

import numpy as np
import numba
from numba import types as nb_types
from numba.core import cgutils, sigutils
from numba.extending import intrinsic

# ---------------------------------------------------------------------------
# Numba version checks
# ---------------------------------------------------------------------------
numba_version = tuple(int(v) for v in numba.__version__.split('.')[:2])
if numba_version == (0, 56):
    raise RuntimeError(
        "FBPIC does not support numba version 0.56.0 because PTX export is "
        "broken in 0.56.0. Please upgrade or downgrade numba."
    )

# ---------------------------------------------------------------------------
# Backend detection & FBPIC_GPU_BACKEND override
# ---------------------------------------------------------------------------
_BACKEND_OVERRIDE = os.environ.get('FBPIC_GPU_BACKEND', '').strip().lower()
if _BACKEND_OVERRIDE not in ('', 'cuda', 'hip', 'cpu'):
    raise ValueError(
        f"Invalid FBPIC_GPU_BACKEND={_BACKEND_OVERRIDE!r}. "
        "Expected one of: '', 'cuda', 'hip', 'cpu'."
    )

gpu_backend = 'cpu'
cuda_installed = False
numba_cuda_installed = False
cupy_installed = False
cupy_version = None
pynvml_installed = False
cupy = None
cufft = None
cublas = None
get_cublas_handle = None
thrust = None
OutOfMemoryError = MemoryError


def _parse_cupy_version(cp_mod):
    parts = cp_mod.__version__.split('.')[:2]
    return (int(parts[0]), int(parts[1]))


def _try_enable_hip(strict=False):
    global gpu_backend, cuda_installed, numba_cuda_installed
    global cupy_installed, cupy_version, cupy
    global cufft, cublas, get_cublas_handle, thrust, OutOfMemoryError

    try:
        import cupy as _cp
    except Exception as exc:
        if strict:
            raise RuntimeError(
                "FBPIC_GPU_BACKEND='hip' requested, but importing CuPy "
                f"failed: {exc}"
            ) from exc
        return False

    if not bool(getattr(_cp.cuda.runtime, 'is_hip', False)):
        if strict:
            raise RuntimeError(
                "FBPIC_GPU_BACKEND='hip' requested, but installed CuPy was "
                "not built for ROCm/HIP (cupy.cuda.runtime.is_hip is False)."
            )
        return False

    try:
        n_dev = int(_cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        if strict:
            raise RuntimeError(
                "FBPIC_GPU_BACKEND='hip' requested, but HIP runtime failed "
                f"to query devices: {exc}"
            ) from exc
        return False

    if n_dev <= 0:
        if strict:
            raise RuntimeError(
                "FBPIC_GPU_BACKEND='hip' requested, "
                "but no AMD HIP GPUs are visible."
            )
        return False

    try:
        import numba.hip as _nb_hip
        from numba.hip.compiler import compile_hip as _compile_hip
        _ = (_nb_hip, _compile_hip)
    except Exception as exc:
        if strict:
            raise RuntimeError(
                "FBPIC_GPU_BACKEND='hip' requested, but numba-hip could "
                f"not be imported: {exc}"
            ) from exc

    gpu_backend = 'hip'
    cupy = _cp
    cupy_installed = True
    cupy_version = _parse_cupy_version(_cp)
    numba_cuda_installed = True
    cuda_installed = True
    cufft = _cp.cuda.cufft
    cublas = _cp.cuda.cublas
    get_cublas_handle = _cp.cuda.device.get_cublas_handle
    thrust = _cp.cuda.thrust
    OutOfMemoryError = _cp.cuda.memory.OutOfMemoryError
    return True


def _try_enable_cuda(strict=False):
    global gpu_backend, cuda_installed, numba_cuda_installed
    global cupy_installed, cupy_version, cupy, pynvml_installed
    global cufft, cublas, get_cublas_handle, thrust, OutOfMemoryError

    try:
        from numba import cuda as _nb_cuda
        nb_avail = bool(_nb_cuda.is_available())
    except Exception as exc:
        if strict:
            raise RuntimeError(
                "FBPIC_GPU_BACKEND='cuda' requested, but numba.cuda failed: "
                f"{exc}"
            ) from exc
        return False

    if not nb_avail:
        if strict:
            raise RuntimeError(
                "FBPIC_GPU_BACKEND='cuda' requested, "
                "but numba.cuda.is_available() is False."
            )
        return False

    try:
        import cupy as _cp
        if bool(getattr(_cp.cuda.runtime, 'is_hip', False)):
            if strict:
                raise RuntimeError(
                    "FBPIC_GPU_BACKEND='cuda' requested, "
                    "but installed CuPy is a HIP build."
                )
            return False
        cp_avail = bool(_cp.is_available())
    except Exception as exc:
        if strict:
            raise RuntimeError(
                "FBPIC_GPU_BACKEND='cuda' requested, but CuPy CUDA failed: "
                f"{exc}"
            ) from exc
        return False

    if not cp_avail:
        if strict:
            raise RuntimeError(
                "FBPIC_GPU_BACKEND='cuda' requested, "
                "but cupy.is_available() is False."
            )
        return False

    try:
        import pynvml
        _ = pynvml
        pynvml_installed = True
    except ImportError:
        pynvml_installed = False

    gpu_backend = 'cuda'
    cupy = _cp
    cupy_installed = True
    cupy_version = _parse_cupy_version(_cp)
    numba_cuda_installed = True
    cuda_installed = True
    cufft = _cp.cuda.cufft
    cublas = _cp.cuda.cublas
    get_cublas_handle = _cp.cuda.device.get_cublas_handle
    thrust = _cp.cuda.thrust
    OutOfMemoryError = _cp.cuda.memory.OutOfMemoryError
    return True


if _BACKEND_OVERRIDE == 'cpu':
    gpu_backend = 'cpu'
elif _BACKEND_OVERRIDE == 'hip':
    _try_enable_hip(strict=True)
elif _BACKEND_OVERRIDE == 'cuda':
    _try_enable_cuda(strict=True)
else:
    # Auto-detect: prefer HIP if CuPy was built for HIP, else CUDA, else CPU
    try:
        import cupy as _probe_cp
        _probe_is_hip = bool(getattr(_probe_cp.cuda.runtime, 'is_hip', False))
    except Exception:
        _probe_cp = None
        _probe_is_hip = False

    if _probe_is_hip:
        _try_enable_hip(strict=False)
    else:
        _try_enable_cuda(strict=False)


# ---------------------------------------------------------------------------
# Numba @intrinsic atomic.add workaround + AMDGPU coarse-grained lowering
# ---------------------------------------------------------------------------
# 1. numba-hip 0.2.1's upstream hip.atomic.add lowers pointer arguments in
#    hipdevicelib into return tuples and rejects N-D float64 array indexing.
# 2. All FBPIC GPU deposition and synchrotron buffers (Jr, Jt, Jz, rho,
#    radiation_data) use CuPy's default `hipMalloc` (coarse-grained device
#    memory, never host-mapped or managed memory). Casting pointers to
#    addrspace(1) with syncscope("agent") and AMDGPU coarse-grained metadata
#    lets LLVM lower atomicrmw fadd cleanly:
#      - gfx90a / gfx942 / gfx950 (CDNA): native `global_atomic_add_f64`
#      - gfx1201 (RDNA4): `global_atomic_cmpswap_b64 scope:SCOPE_DEV`
#      - gfx908 (CDNA1): `global_atomic_cmpswap_x2`
# Keeping this LLVM @intrinsic inside fbpic avoids editing site-packages.
@intrinsic
def _hip_atomic_add(typingctx, ary_ty, idx_ty, val_ty):
    if not isinstance(ary_ty, nb_types.Array):
        return None
    dtype = ary_ty.dtype
    if isinstance(idx_ty, nb_types.BaseTuple):
        if len(idx_ty) != ary_ty.ndim:
            return None
        if not all(isinstance(t, nb_types.Integer) for t in idx_ty):
            return None
    elif isinstance(idx_ty, nb_types.Integer):
        if ary_ty.ndim != 1:
            return None
    else:
        return None
    if not isinstance(val_ty, (nb_types.Float, nb_types.Integer)):
        return None

    sig = dtype(ary_ty, idx_ty, dtype)

    def codegen(context, builder, signature, args):
        from llvmlite import ir

        class _AgentAtomicRMW(ir.Instruction):
            def __init__(self, parent, op, ptr, val, typ, align, name=''):
                super().__init__(
                    parent, typ, 'atomicrmw', [ptr, val], name=name
                )
                self.op = op
                self.align = align

            def descr(self, buf):
                op0 = self.operands[0]
                op1 = self.operands[1]
                md = self._stringify_metadata(leading_comma=True)
                buf.append(
                    f"atomicrmw {self.op} {op0.type} {op0.get_reference()}, "
                    f"{op1.type} {op1.get_reference()} "
                    f"syncscope(\"agent\") monotonic, align {self.align}{md}\n"
                )

        ary_t, idx_t, _ = signature.args
        ary_v, idx_v, val_v = args
        ary_struct = context.make_array(ary_t)(context, builder, value=ary_v)
        if isinstance(idx_t, nb_types.BaseTuple):
            unpacked = cgutils.unpack_tuple(builder, idx_v, count=len(idx_t))
            indices = [
                context.cast(builder, unpacked[k], idx_t[k], nb_types.intp)
                for k in range(len(idx_t))
            ]
        else:
            indices = [context.cast(builder, idx_v, idx_t, nb_types.intp)]
        ptr = cgutils.get_item_pointer(
            context, builder, ary_t, ary_struct, indices
        )
        g_ptr = builder.addrspacecast(
            ptr, ir.PointerType(ptr.type.pointee, addrspace=1)
        )
        if isinstance(dtype, nb_types.Float):
            op = 'fadd'
        elif isinstance(dtype, nb_types.Integer):
            op = 'add'
        else:
            raise NotImplementedError(f"Unsupported atomic.add dtype: {dtype}")

        inst = _AgentAtomicRMW(
            builder.block, op, g_ptr, val_v, val_v.type,
            align=(dtype.bitwidth // 8),
        )
        empty_md = builder.module.add_metadata([])
        inst.set_metadata('amdgpu.no.fine.grained.memory', empty_md)
        inst.set_metadata('amdgpu.no.remote.memory', empty_md)
        builder._insert(inst)
        return inst

    return sig, codegen


# ---------------------------------------------------------------------------
# Unified `cuda`-compatible adapter (HIP / CUDA / CPU-noop)
# ---------------------------------------------------------------------------
class _DummyCudaAdapter:
    """Fallback namespace when no GPU backend is active."""

    def is_available(self):
        return False

    def jit(self, func_or_sig=None, *args, **kwargs):
        if (
            callable(func_or_sig)
            and not isinstance(func_or_sig, nb_types.Type)
        ):
            return func_or_sig

        def _decorator(fn):
            return fn

        return _decorator

    def synchronize(self):
        return None

    def pinned_array(self, shape, dtype=np.float64):
        return np.empty(shape, dtype=dtype)


class _HipAtomicNamespace(py_types.ModuleType):
    """Module proxy exposing `atomic.add` plus `numba.hip.atomic` ops."""

    def __init__(self, nb_hip_atomic):
        super().__init__('fbpic.utils._hip_atomic')
        self._upstream = nb_hip_atomic
        self.add = _hip_atomic_add

    def __getattr__(self, name):
        return getattr(self._upstream, name)


class _HipDeviceProxy:
    def __init__(self, dev_id):
        self.id = int(dev_id)
        props = cupy.cuda.runtime.getDeviceProperties(self.id)
        raw_name = props.get('name', b'AMD HIP GPU')
        if isinstance(raw_name, bytes):
            self.name = raw_name
        else:
            self.name = str(raw_name).encode()


class _HipDeviceList:
    @property
    def lst(self):
        n = int(cupy.cuda.runtime.getDeviceCount())
        return [_HipDeviceProxy(i) for i in range(n)]

    @property
    def current(self):
        return _HipDeviceProxy(int(cupy.cuda.runtime.getDevice()))

    def __len__(self):
        return int(cupy.cuda.runtime.getDeviceCount())

    def __iter__(self):
        return iter(self.lst)


class _HipContextProxy:
    def get_memory_info(self):
        return cupy.cuda.runtime.memGetInfo()


class _HipCudaAdapter(py_types.ModuleType):
    """
    Module-typed proxy exposing the ``numba.cuda`` API on AMD HIP.

    Inherits from ``types.ModuleType`` so Numba's nopython frontend resolves
    ``cuda.grid``, ``cuda.local``, ``cuda.syncthreads``, ``cuda.atomic.add``,
    etc. inside ``@compile_cupy`` kernels without modifying ``numba.hip``.
    Runtime calls go strictly through CuPy so ``numba.hip`` never creates a
    duplicate driver context.
    """

    def __init__(self):
        super().__init__('fbpic.utils._hip_cuda')
        import numba.hip as _nb_hip
        self._hip = _nb_hip
        self.atomic = _HipAtomicNamespace(_nb_hip.atomic)
        self.gpus = _HipDeviceList()
        for attr in (
            'grid', 'gridsize', 'threadIdx', 'blockIdx', 'blockDim',
            'gridDim', 'syncthreads', 'shared', 'local', 'const',
        ):
            setattr(self, attr, getattr(_nb_hip, attr))

    def is_available(self):
        return True

    def jit(self, *args, **kwargs):
        return self._hip.jit(*args, **kwargs)

    def synchronize(self):
        cupy.cuda.runtime.deviceSynchronize()

    def select_device(self, gpu_id):
        cupy.cuda.Device(int(gpu_id)).use()

    def current_context(self):
        return _HipContextProxy()

    def pinned_array(self, shape, dtype=np.float64):
        dtype = np.dtype(dtype)
        if isinstance(shape, int):
            shape = (shape,)
        size = int(np.prod(shape)) if len(shape) > 0 else 1
        nbytes = size * dtype.itemsize
        if nbytes == 0:
            return np.empty(shape, dtype=dtype)
        mem = cupy.cuda.alloc_pinned_memory(nbytes)
        return np.frombuffer(mem, dtype=dtype, count=size).reshape(shape)

    def detect(self):
        n = int(cupy.cuda.runtime.getDeviceCount())
        print(f"Found {n} AMD HIP device(s):")
        for i in range(n):
            info = get_device_arch_info(i)
            free_b, total_b = cupy.cuda.Device(i).mem_info
            print(
                f"  id {i}: {info['name']} ({info['arch']}, "
                f"warpSize={info['warp_size']}, "
                f"mem={total_b / (1024 ** 3):.2f} GiB)"
            )


def _register_hip_cuda_adapter_typing(cuda_mod):
    """Register Numba typing and lowering for `_HipCudaAdapter` attrs."""
    import numba.hip as _nb_hip
    from numba.core.typing.templates import AttributeTemplate
    from numba.hip.descriptor import hip_target
    from numba.hip.typing_lowering.registries import (
        typing_registry,
        impl_registry,
    )
    from numba.hip.typing_lowering.hip.lowering import initialize_dim3

    cuda_mod_ty = nb_types.Module(cuda_mod)
    hip_mod_ty = nb_types.Module(_nb_hip)

    @typing_registry.register_attr
    class _FbpicHipCudaModuleTemplate(AttributeTemplate):
        key = cuda_mod_ty

        def generic_resolve(self, mod, attr):
            if attr == 'atomic':
                return nb_types.Module(cuda_mod.atomic)
            return self.context.resolve_getattr(hip_mod_ty, attr)

    for _dim_name in ('threadIdx', 'blockDim', 'blockIdx', 'gridDim'):
        def _register_dim(dname):
            @impl_registry.lower_getattr(cuda_mod_ty, dname)
            def _lower_dim(*all_args):
                return initialize_dim3(dname, *all_args)
        _register_dim(_dim_name)

    hip_target.typing_context.refresh()
    hip_target.target_context.refresh()


if gpu_backend == 'cuda':
    from numba import cuda as cuda  # noqa: F401
elif gpu_backend == 'hip':
    cuda = _HipCudaAdapter()
    _register_hip_cuda_adapter_typing(cuda)
else:
    cuda = _DummyCudaAdapter()


# ---------------------------------------------------------------------------
# Device architecture properties & launch geometry
# ---------------------------------------------------------------------------
_ARCH_INFO_CACHE = {}


def _classify_amd_family(arch_str):
    if arch_str.startswith('gfx9'):
        return 'cdna'
    if arch_str.startswith(('gfx10', 'gfx11', 'gfx12')):
        return 'rdna'
    return 'amd'


def _is_truthy_env(name):
    val = os.environ.get(name, '').strip().lower()
    return val in ('1', 'true', 'yes', 'on')


def _decode_prop_str(val):
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    return str(val)


def get_device_arch_info(dev_id=None):
    """
    Query and cache architecture properties for the active or given GPU.

    Parameters
    ----------
    dev_id : int or None, optional
        Device index to inspect. When None, uses the active device.

    Returns
    -------
    info : dict
        Dictionary with keys ``vendor``, ``name``, ``arch``, ``warp_size``,
        ``family``, ``cu_count``, ``pci_domain``, ``pci_bus``, and
        ``pci_device`` (plus ``'cpu'`` defaults when no GPU backend is active).
    """
    if not cuda_installed:
        return {
            'vendor': 'cpu',
            'name': 'CPU',
            'arch': 'cpu',
            'warp_size': 1,
            'family': 'cpu',
            'cu_count': 0,
        }
    if dev_id is None:
        if gpu_backend == 'hip':
            dev_id = int(cupy.cuda.runtime.getDevice())
        else:
            dev_id = int(cuda.gpus.current.id)
    dev_id = int(dev_id)

    wave64_override = (
        (gpu_backend == 'hip') and _is_truthy_env('FBPIC_HIP_WAVE64')
    )
    cache_key = (dev_id, wave64_override)
    if cache_key in _ARCH_INFO_CACHE:
        return _ARCH_INFO_CACHE[cache_key]

    props = cupy.cuda.runtime.getDeviceProperties(dev_id)
    name = _decode_prop_str(props.get('name', b'Unknown GPU'))

    if gpu_backend == 'hip':
        arch_full = _decode_prop_str(props.get('gcnArchName', b''))
        arch = arch_full.split(':')[0].strip()
        default_warp = 64 if arch.startswith('gfx9') else 32
        warp_size = 64 if wave64_override else int(
            props.get('warpSize', default_warp)
        )
        family = _classify_amd_family(arch)
        vendor = 'amd'
    else:
        major = int(props.get('major', 0))
        minor = int(props.get('minor', 0))
        arch = f"sm_{major}{minor}"
        warp_size = int(props.get('warpSize', 32))
        family = 'nvidia'
        vendor = 'nvidia'

    info = {
        'vendor': vendor,
        'name': name,
        'arch': arch,
        'warp_size': warp_size,
        'family': family,
        'cu_count': int(props.get('multiProcessorCount', 0)),
        'pci_domain': int(props.get('pciDomainID', 0)),
        'pci_bus': int(props.get('pciBusID', 0)),
        'pci_device': int(props.get('pciDeviceID', 0)),
    }
    _ARCH_INFO_CACHE[cache_key] = info
    return info


# Preserve `cuda_gpu_model` identically on NVIDIA CUDA; report `arch` on HIP.
if gpu_backend == 'cuda':
    _gpu_name = _decode_prop_str(cuda.gpus.current.name)
    if 'P100' in _gpu_name:
        cuda_gpu_model = 'P100'
    elif 'V100' in _gpu_name:
        cuda_gpu_model = 'V100'
    elif 'A100' in _gpu_name:
        cuda_gpu_model = 'V100'
    else:
        cuda_gpu_model = 'other'
elif gpu_backend == 'hip':
    cuda_gpu_model = get_device_arch_info()['arch']
else:
    cuda_gpu_model = None


def align_tpb_1d(tpb, warp_size):
    """
    Round a 1D block size up to a positive multiple of ``warp_size``.

    Parameters
    ----------
    tpb : int
        Requested threads per block.

    warp_size : int
        Hardware wavefront / warp size.

    Returns
    -------
    aligned_tpb : int
        Smallest positive integer >= ``tpb`` divisible by ``warp_size``.
    """
    tpb = max(1, int(tpb))
    rem = tpb % warp_size
    return tpb if rem == 0 else tpb + (warp_size - rem)


def align_tpb_2d(tpbx, tpby, warp_size):
    """
    Adjust ``tpby`` upward so ``tpbx * tpby`` is a multiple of ``warp_size``.

    Parameters
    ----------
    tpbx, tpby : int
        Requested 2D block dimensions.

    warp_size : int
        Hardware wavefront / warp size.

    Returns
    -------
    aligned_pair : tuple of (int, int)
        Aligned ``(tpbx, tpby)`` block dimensions.
    """
    tpbx = max(1, int(tpbx))
    tpby = max(1, int(tpby))
    if (tpbx * tpby) % warp_size == 0:
        return (tpbx, tpby)
    step = warp_size // math.gcd(tpbx, warp_size)
    rem = tpby % step
    if rem != 0:
        tpby += (step - rem)
    return (tpbx, tpby)


def cuda_tpb_bpg_1d(x, TPB=256):
    """
    Calculate the number of blocks and threads per block for a 1D grid.

    On AMD HIP, ``TPB`` is rounded up to a multiple of the device wavefront
    size.

    Parameters
    ----------
    x : int
        Total number of work items in 1D.

    TPB : int, optional
        Desired threads per block.

    Returns
    -------
    bpg, tpb : int
        Blocks per grid and threads per block.
    """
    tpb = TPB
    if gpu_backend == 'hip':
        warp_size = get_device_arch_info()['warp_size']
        tpb = align_tpb_1d(tpb, warp_size)
    bpg = int(x / tpb + 1)
    return bpg, tpb


def cuda_tpb_bpg_2d(x, y, TPBx=1, TPBy=128):
    """
    Calculate the blocks and threads per block for a 2D kernel launch.

    On AMD HIP, ``(TPBx, TPBy)`` is aligned so ``tpbx * tpby`` is a multiple
    of the device wavefront size.

    Parameters
    ----------
    x, y : int
        Number of work items along the first and second axes.

    TPBx, TPBy : int, optional
        Desired threads per block along each axis.

    Returns
    -------
    bpg, tpb : tuple of (int, int)
        2D grid dimensions and 2D block dimensions.
    """
    tpb_x, tpb_y = TPBx, TPBy
    if gpu_backend == 'hip':
        warp_size = get_device_arch_info()['warp_size']
        tpb_x, tpb_y = align_tpb_2d(tpb_x, tpb_y, warp_size)
    bpg_x = int(x / tpb_x + 1)
    bpg_y = int(y / tpb_y + 1)
    return (bpg_x, bpg_y), (tpb_x, tpb_y)


def _parse_tpb2_env(env_name, default_pair):
    raw = os.environ.get(env_name, '').strip()
    if not raw:
        return default_pair
    parts = [int(p.strip()) for p in raw.split(',')]
    if len(parts) != 2:
        raise ValueError(f"{env_name} must be 'TPBx,TPBy', got {raw!r}")
    return (parts[0], parts[1])


def get_copy_tpb_2d(arch_info=None):
    """
    Return the 2D block size for Fourier/Hankel buffer copy kernels.

    Parameters
    ----------
    arch_info : dict or None, optional
        Optional device metadata dict from ``get_device_arch_info``.

    Returns
    -------
    tpb_2d : tuple of (int, int)
        2D threads-per-block configuration.
    """
    if gpu_backend != 'hip':
        return (8, 32) if cuda_gpu_model == 'V100' else (2, 16)
    info = arch_info or get_device_arch_info()
    # (4, 32) = 128 threads/block (multiple of both wave32 and wave64);
    # measured 1.53x faster than (2, 16) on gfx1201 copy_2d_to_1d.
    default = (4, 32)
    pair = _parse_tpb2_env('FBPIC_HIP_COPY_TPB', default)
    return align_tpb_2d(pair[0], pair[1], info['warp_size'])


def get_spectral_tpb_2d(arch_info=None):
    """
    Return the 2D block size for spectral grid and transformer kernels.

    Parameters
    ----------
    arch_info : dict or None, optional
        Optional device metadata dict from ``get_device_arch_info``.

    Returns
    -------
    tpb_2d : tuple of (int, int)
        2D threads-per-block configuration.
    """
    if gpu_backend != 'hip':
        return (1, 16)
    info = arch_info or get_device_arch_info()
    is_w64 = (info['warp_size'] == 64 or info['family'] == 'cdna')
    default = (1, 64) if is_w64 else (4, 32)
    pair = _parse_tpb2_env('FBPIC_HIP_SPECTRAL_TPB', default)
    return align_tpb_2d(pair[0], pair[1], info['warp_size'])


def get_deposit_tpb(particle_shape, arch_info=None):
    """
    Return the 1D block size for current and charge deposition kernels.

    Parameters
    ----------
    particle_shape : {'linear', 'cubic'}
        Particle interpolation shape.

    arch_info : dict or None, optional
        Optional device metadata dict from ``get_device_arch_info``.

    Returns
    -------
    tpb : int
        Threads per block for deposition kernels.
    """
    if gpu_backend != 'hip':
        if particle_shape == 'cubic':
            return 32
        return 16 if cuda_gpu_model == 'V100' else 8
    info = arch_info or get_device_arch_info()
    env_val = os.environ.get('FBPIC_HIP_DEPOSIT_TPB', '').strip()
    if env_val:
        return align_tpb_1d(int(env_val), info['warp_size'])
    return align_tpb_1d(64, info['warp_size'])


def get_gather_tpb(particle_shape, arch_info=None):
    """
    Return the 1D block size for field gathering kernels.

    Parameters
    ----------
    particle_shape : {'linear', 'cubic'}
        Particle interpolation shape.

    arch_info : dict or None, optional
        Optional device metadata dict from ``get_device_arch_info``.

    Returns
    -------
    tpb : int
        Threads per block for gathering kernels.
    """
    if gpu_backend != 'hip':
        return 256 if particle_shape == 'cubic' else 128
    info = arch_info or get_device_arch_info()
    env_val = os.environ.get('FBPIC_HIP_GATHER_TPB', '').strip()
    if env_val:
        return align_tpb_1d(int(env_val), info['warp_size'])
    return align_tpb_1d(256, info['warp_size'])


# ---------------------------------------------------------------------------
# Data transfer & GPU memory manager
# ---------------------------------------------------------------------------
def send_data_to_gpu(simulation):
    """
    Copy particle and field arrays in ``simulation`` from CPU to GPU.

    Parameters
    ----------
    simulation : fbpic.main.Simulation
        Simulation instance whose fields and GPU-enabled species are copied.
    """
    for species in simulation.ptcl:
        if species.use_cuda:
            species.send_particles_to_gpu()
    simulation.fld.send_fields_to_gpu()


def receive_data_from_gpu(simulation):
    """
    Copy particle and field arrays in ``simulation`` from GPU to CPU.

    Parameters
    ----------
    simulation : fbpic.main.Simulation
        Simulation instance whose fields and GPU-enabled species are copied.
    """
    for species in simulation.ptcl:
        if species.use_cuda:
            species.receive_particles_from_gpu()
    simulation.fld.receive_fields_from_gpu()


class GpuMemoryManager(object):
    """
    Context manager moving simulation data temporarily to GPU if on CPU.

    Parameters
    ----------
    simulation : fbpic.main.Simulation
        Simulation instance whose data residency is managed across ``with``.
    """

    def __init__(self, simulation):
        self.fields_were_on_gpu = simulation.fld.data_is_on_gpu
        self.species_were_on_gpu = [
            species.data_is_on_gpu for species in simulation.ptcl
        ]
        self.sim = simulation

    def __enter__(self):
        if self.sim.use_cuda:
            if not self.fields_were_on_gpu:
                self.sim.fld.send_fields_to_gpu()
            for i, species in enumerate(self.sim.ptcl):
                if not self.species_were_on_gpu[i]:
                    species.send_particles_to_gpu()

    def __exit__(self, exc_type, value, traceback):
        if self.sim.use_cuda:
            if not self.fields_were_on_gpu:
                self.sim.fld.receive_fields_from_gpu()
            for i, species in enumerate(self.sim.ptcl):
                if not self.species_were_on_gpu[i]:
                    species.receive_particles_from_gpu()


# ---------------------------------------------------------------------------
# UUID & MPI GPU selection
# ---------------------------------------------------------------------------
def _format_uuid16(raw_bytes):
    if isinstance(raw_bytes, str):
        return raw_bytes
    b = bytes(raw_bytes)
    if len(b) == 16 and any(b):
        # Some HIP drivers return 16 ASCII hex chars (e.g. b'e01b1a22e38ff4b9')
        try:
            text = b.decode('ascii')
            if all(ch in '0123456789abcdefABCDEF' for ch in text):
                return f"GPU-{text.lower()}"
        except UnicodeDecodeError:
            pass
        hx = (
            '%02x%02x%02x%02x-%02x%02x-%02x%02x-'
            '%02x%02x-%02x%02x%02x%02x%02x%02x'
        )
        return ('GPU-' + hx) % tuple(b)
    return None


def get_uuid(gpu_id):
    """
    Return a unique identifier string for GPU ``gpu_id``, or ``None``.

    Parameters
    ----------
    gpu_id : int
        Device index among the visible GPUs.

    Returns
    -------
    uuid : str or None
        Formatted UUID string or ``None`` if unavailable.
    """
    if not cuda_installed or cupy_version < (8, 1):
        return None

    if gpu_backend == 'hip':
        # 1. Try CuPy properties if populated
        props = cupy.cuda.runtime.getDeviceProperties(int(gpu_id))
        raw_uuid = props.get('uuid', None)
        if raw_uuid is not None:
            formatted = _format_uuid16(raw_uuid)
            if formatted:
                return formatted

        # 2. Try hip-python hipDeviceGetUuid
        try:
            from hip import hip as hip_rt
            err, uuid_obj = hip_rt.hipDeviceGetUuid(int(gpu_id))
            if int(err) == 0:
                uuid_bytes = getattr(uuid_obj, 'bytes', uuid_obj)
                formatted = _format_uuid16(uuid_bytes)
                if formatted:
                    return formatted
        except Exception:
            pass

        # 3. Try optional amdsmi
        try:
            import amdsmi
            amdsmi.amdsmi_init()
            try:
                handles = amdsmi.amdsmi_get_processor_handles()
                if int(gpu_id) < len(handles):
                    asic_info = amdsmi.amdsmi_get_gpu_asic_info(
                        handles[int(gpu_id)]
                    )
                    serial = asic_info.get('asic_serial')
                    if serial:
                        return f"AMD-SN-{serial}"
            finally:
                amdsmi.amdsmi_shut_down()
        except Exception:
            pass

        # 4. Deterministic PCI BDF fallback
        info = get_device_arch_info(int(gpu_id))
        return (
            f"GPU-PCI-{info['pci_domain']:04x}:"
            f"{info['pci_bus']:02x}:{info['pci_device']:02x}"
        )

    # NVIDIA CUDA path (unchanged)
    uuid = cupy.cuda.runtime.getDeviceProperties(gpu_id)['uuid']
    if len(uuid) != 16:
        warnings.warn(
            f"Failed to detect UUID of GPU {gpu_id} "
            f"(invalid UUID length: {len(uuid)})"
        )
        return None
    b = '%02x'
    b2 = b * 2
    b4 = b * 4
    b6 = b * 6
    fmt = f'GPU-{b4}-{b2}-{b2}-{b2}-{b6}'
    return fmt % tuple(bytes(uuid))


def get_uuid_alt(gpu_id):
    """
    Return the device UUID string using optional ``pynvml`` on NVIDIA.

    Parameters
    ----------
    gpu_id : int
        Device index among the visible GPUs.

    Returns
    -------
    uuid : str
        NVML device UUID string.
    """
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    uuid = pynvml.nvmlDeviceGetUUID(handle)
    pynvml.nvmlShutdown()
    return uuid


def check_consecutive_ranks_on_same_nodes(mpi):
    """
    Check whether consecutive MPI ranks reside on the same compute node.

    Parameters
    ----------
    mpi : module
        Imported ``mpi4py.MPI`` module.

    Returns
    -------
    standard_decomp : bool
        True if consecutive ranks share nodes cleanly.
    """
    if mpi.COMM_WORLD.size == 1:
        return True
    nodes = mpi.COMM_WORLD.gather(mpi.Get_processor_name(), root=0)
    standard_decomp = None
    if mpi.COMM_WORLD.rank == 0:
        standard_decomp = True
        unique_nodes = [nodes[0]]
        for i in range(1, len(nodes)):
            if nodes[i] != nodes[i - 1]:
                if nodes[i] not in unique_nodes:
                    unique_nodes.append(nodes[i])
                else:
                    standard_decomp = False
                    break
        if not standard_decomp:
            warnings.warn(
                "It seems that the distribution of MPI ranks on compute "
                "nodes is such\n"
                "that consecutive MPI ranks are not located on the same "
                "node.\n"
                "(See the FBPIC output with `verbose_level = 2` for more "
                "details.)\n"
                "This type of MPI distribution can degrade the performance "
                "of FBPIC.\n"
                "Please check the options of your MPI launcher "
                "(e.g. `mpirun`, `srun`)\n"
                "in order to use a different MPI distribution."
            )
    standard_decomp = mpi.COMM_WORLD.bcast(standard_decomp, root=0)
    return standard_decomp


def mpi_select_gpus(mpi):
    """
    Select one GPU per MPI rank and warn on oversubscription.

    Parameters
    ----------
    mpi : module
        Imported ``mpi4py.MPI`` module.
    """
    if not cuda_installed:
        return

    std_decomp = check_consecutive_ranks_on_same_nodes(mpi)
    if not std_decomp:
        node_name = mpi.Get_processor_name()
        color = int.from_bytes(node_name.encode(), 'little') % 100000000
        comm = mpi.COMM_WORLD.Split(color=color)
    else:
        comm = mpi.COMM_WORLD

    n_gpus = len(cuda.gpus)
    if gpu_backend == 'hip' and n_gpus > 1 and comm.rank == 0:
        archs = {get_device_arch_info(i)['arch'] for i in range(n_gpus)}
        if len(archs) > 1:
            warnings.warn(
                f"Multiple heterogeneous AMD GPUs visible ({sorted(archs)}). "
                "Set ROCR_VISIBLE_DEVICES (or HIP_VISIBLE_DEVICES on "
                "bare-metal Linux) to restrict FBPIC to homogeneous GPUs."
            )

    rank = comm.rank
    uuid = None
    for i_gpu in range(n_gpus):
        if rank % n_gpus == i_gpu:
            cuda.select_device(i_gpu)
            if pynvml_installed and gpu_backend == 'cuda':
                uuid = get_uuid_alt(i_gpu)
            else:
                uuid = get_uuid(i_gpu)
        mpi.COMM_WORLD.barrier()

    uuids = mpi.COMM_WORLD.gather(uuid)
    if rank == 0:
        if not (None in uuids) and (len(uuids) > len(set(uuids))):
            warnings.warn(
                "GPUs have been oversubscribed by MPI ranks.\n"
                "This means that the same GPU was selected by more than one "
                "parallel process,\nwhich will result in poor performance.\n"
                "(See the FBPIC output with `verbose_level = 2` for more "
                "details.)"
            )


# ---------------------------------------------------------------------------
# Kernel compilation, caching & registry
# ---------------------------------------------------------------------------
_REGISTERED_KERNELS = []
_KERNEL_CACHE_STATS = {
    'hits': 0,
    'misses': 0,
    'compile_time_s': 0.0,
    'load_time_s': 0.0,
}


def get_kernel_cache_stats():
    """
    Return disk HSACO kernel cache hit/miss and timing statistics.

    Returns
    -------
    stats : dict
        Copy of cache statistics (``hits``, ``misses``, ``compile_time_s``,
        ``load_time_s``).
    """
    return dict(_KERNEL_CACHE_STATS)


def reset_kernel_cache_stats():
    """Reset the HSACO disk kernel cache statistics counters."""
    _KERNEL_CACHE_STATS['hits'] = 0
    _KERNEL_CACHE_STATS['misses'] = 0
    _KERNEL_CACHE_STATS['compile_time_s'] = 0.0
    _KERNEL_CACHE_STATS['load_time_s'] = 0.0


def get_registered_kernels():
    """
    Return the list of ``(module, qualname, wrapper)`` for ``@compile_cupy``.

    Returns
    -------
    kernels : list of tuple
        Registered ``CompileCupy`` kernel wrappers.
    """
    return list(_REGISTERED_KERNELS)


def get_args_hash(args):
    """
    Compute a signature hash for kernel specialization caching.

    Uses ``(dtype, ndim)`` on CUDA and ``(dtype, ndim, layout)`` on HIP.

    Parameters
    ----------
    args : tuple
        Runtime kernel arguments.

    Returns
    -------
    args_hash : int
        Hash key identifying the kernel specialization.
    """
    sig_parts = []
    for a in args:
        if cupy is not None and isinstance(a, cupy.ndarray):
            if gpu_backend == 'hip':
                layout = (
                    'C' if a.flags.c_contiguous
                    else ('F' if a.flags.f_contiguous else 'A')
                )
                sig_parts.append((a.dtype, a.ndim, layout))
            else:
                sig_parts.append(a.dtype)
                sig_parts.append(a.ndim)
        else:
            sig_parts.append(np.dtype(type(a)))
    return hash(tuple(sig_parts))


def _arg_to_numba_type(arg):
    if isinstance(arg, cupy.ndarray):
        layout = (
            'C' if arg.flags.c_contiguous
            else ('F' if arg.flags.f_contiguous else 'A')
        )
        return nb_types.Array(numba.from_dtype(arg.dtype), arg.ndim, layout)
    if isinstance(arg, (bool, np.bool_)):
        return nb_types.boolean
    if isinstance(arg, int):
        return nb_types.int64
    if isinstance(arg, float):
        return nb_types.float64
    if isinstance(arg, (np.integer, np.floating)):
        return numba.from_dtype(arg.dtype)
    if isinstance(arg, (complex, np.complexfloating)):
        raise TypeError(
            "Complex scalar kernel arguments are not supported on HIP: "
            f"{type(arg)}"
        )
    raise TypeError(f"Unsupported GPU kernel argument type: {type(arg)}")


def _fingerprint_callable(func, seen=None):
    """Hash a Python/Numba function plus its closures and called helpers."""
    if seen is None:
        seen = set()
    fid = id(func)
    if fid in seen:
        return ''
    seen.add(fid)

    h = hashlib.sha256()
    py_func = getattr(func, 'py_func', func)
    if not isinstance(py_func, py_types.FunctionType):
        h.update(repr(py_func).encode('utf-8', errors='replace'))
        return h.hexdigest()

    code = py_func.__code__
    h.update(py_func.__module__.encode('utf-8', errors='replace'))
    h.update(py_func.__qualname__.encode('utf-8', errors='replace'))
    h.update(code.co_code)
    for const in code.co_consts:
        if isinstance(const, py_types.CodeType):
            h.update(const.co_code)
        else:
            h.update(repr(const).encode('utf-8', errors='replace'))

    if py_func.__closure__:
        for cell in py_func.__closure__:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if callable(val) or hasattr(val, 'py_func'):
                h.update(_fingerprint_callable(val, seen).encode('ascii'))
            else:
                h.update(repr(val).encode('utf-8', errors='replace'))

    globs = py_func.__globals__
    disp_types = ('HIPDispatcher', 'CPUDispatcher', 'Intrinsic')
    for gname in code.co_names:
        if gname in globs:
            gval = globs[gname]
            # Include any device-jit functions / njit helpers called here
            if (
                hasattr(gval, 'py_func')
                or type(gval).__name__ in disp_types
            ):
                h.update(gname.encode('utf-8', errors='replace'))
                h.update(_fingerprint_callable(gval, seen).encode('ascii'))
    return h.hexdigest()


def _get_hip_cache_dir():
    default_dir = os.path.join(
        os.path.expanduser('~'), '.cache', 'fbpic-rocm', 'kernels'
    )
    return os.environ.get('FBPIC_CACHE_DIR', default_dir)


def _hip_compiler_fingerprint():
    import numba.hip as _nb_hip
    hip_ver = getattr(cupy.cuda.runtime, 'runtimeGetVersion', lambda: 0)()
    nb_hip_ver = getattr(_nb_hip, '__version__', 'unknown')
    return f"nb{numba.__version__}_nbhip{nb_hip_ver}_hip{hip_ver}_v2"


def compile_hip_kernel_to_codeobj(
    py_func, argtypes, amdgpu_arch, wave_size=None, extra_llvm_opts=None,
):
    """
    Compile ``py_func`` via ``numba.hip`` into an AMDGPU ELF code object.

    RDNA targets (e.g. ``gfx1201``) link wave32 HIP device bitcode inside
    ``numba-hip``, so hardware code objects remain wave32 on RDNA even when
    ``FBPIC_HIP_WAVE64=1`` tests wave64 launch-geometry alignment; CDNA
    (``gfx9*``) targets natively compile wave64 code objects via
    ``lib.get_codeobj``.

    Parameters
    ----------
    py_func : callable
        Python GPU kernel function.

    argtypes : tuple of numba.types.Type
        Numba argument types for kernel specialization.

    amdgpu_arch : str
        AMDGPU LLVM target name (e.g. ``'gfx1201'``, ``'gfx942'``).

    wave_size : int or None, optional
        Wavefront size when invoking external LLVM bitcode compilation.

    extra_llvm_opts : list of str or None, optional
        Optional extra flags forwarded to ``amdclang``.

    Returns
    -------
    entry_name : str
        Mangled kernel symbol name inside the ELF code object.

    codeobj : bytes
        Compiled HSACO ELF binary.
    """
    from numba.hip.compiler import compile_hip

    options = {'fastmath': False, 'opt': 3}
    cres = compile_hip(
        py_func, nb_types.void, argtypes,
        debug=False, lineinfo=False, inline=False, fastmath=False,
        options=options, amdgpu_arch=amdgpu_arch,
    )
    code = py_func.__code__
    lib, kern = cres.target_context.prepare_hip_kernel(
        cres.library, cres.fndesc, False, False, options,
        code.co_filename, code.co_firstlineno, None,
    )

    if extra_llvm_opts:
        ll_raw = lib.get_linked_llvm_ir(
            amdgpu_arch=amdgpu_arch,
            to_bc=False,
            link_in_hipdevicelib=True,
        )
        if isinstance(ll_raw, (bytes, bytearray)):
            ll_text = ll_raw.decode('utf-8')
        else:
            ll_text = ll_raw
        codeobj = _compile_llvm_ir_with_rocm_tools(
            ll_text,
            is_bc=False,
            amdgpu_arch=amdgpu_arch,
            wave_size=wave_size,
            extra_opts=extra_llvm_opts,
        )
        return kern.name, codeobj

    codeobj = lib.get_codeobj(amdgpu_arch=amdgpu_arch)
    return kern.name, bytes(codeobj)


def _compile_llvm_ir_with_rocm_tools(
    ir_data, is_bc, amdgpu_arch, wave_size=None, extra_opts=None,
):
    """Compile LLVM IR/bitcode to HSACO using ROCm `amdclang` / `comgr`."""
    import subprocess
    if not extra_opts and is_bc:
        try:
            from numba.hip.hipdrv import comgrutils
            if getattr(comgrutils, 'has_compile_bc', False):
                (codeobj, _, _) = comgrutils.compile_bc(
                    ir_data,
                    isa_name=f"amdgcn-amd-amdhsa--{amdgpu_arch}",
                    logging=True,
                )
                return bytes(codeobj)
        except Exception:
            pass

    rocm_home = os.environ.get('ROCM_HOME', '/opt/rocm')
    clang_candidates = [
        os.path.join(rocm_home, 'llvm', 'bin', 'amdclang'),
        os.path.join(rocm_home, 'llvm', 'bin', 'clang'),
        '/opt/rocm/llvm/bin/amdclang',
        '/opt/rocm/llvm/bin/clang',
        '/usr/bin/amdclang',
    ]
    clang_bin = next((p for p in clang_candidates if os.path.isfile(p)), None)
    if clang_bin is None:
        raise RuntimeError(
            "Could not locate amdclang/clang in ROCM_HOME/llvm/bin to "
            "compile modified LLVM IR."
        )

    suffix = '.bc' if is_bc else '.ll'
    with tempfile.TemporaryDirectory() as tmpd:
        in_path = os.path.join(tmpd, 'kernel' + suffix)
        out_path = os.path.join(tmpd, 'kernel.hsaco')
        mode = 'wb' if isinstance(ir_data, (bytes, bytearray)) else 'w'
        with open(in_path, mode) as f:
            f.write(ir_data)
        cmd = [
            clang_bin, '-x', 'ir',
            '--target=amdgcn-amd-amdhsa',
            f'-mcpu={amdgpu_arch}',
            '-O3',
            '-nogpulib',
        ]
        if wave_size == 64:
            cmd.append('-mwavefrontsize64')
        if extra_opts:
            cmd.extend(extra_opts)
        cmd.extend(['-o', out_path, in_path])
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"amdclang failed ({' '.join(cmd)}):\n"
                f"{res.stderr}\n{res.stdout}"
            )
        with open(out_path, 'rb') as f:
            return f.read()


def _flatten_kernel_args(args):
    kernel_args = []
    for a in args:
        if isinstance(a, cupy.ndarray):
            kernel_args.extend([0, 0, a.size, a.dtype.itemsize, a])
            kernel_args.extend(a.shape)
            kernel_args.extend(a.strides)
        elif gpu_backend == 'hip' and isinstance(a, (bool, np.bool_)):
            kernel_args.append(np.uint8(a))
        else:
            kernel_args.append(a)
    return kernel_args


if cuda_installed:

    class CompileCupy(object):
        """
        Decorator compiling Numba GPU kernels for launch via CuPy.

        On NVIDIA CUDA it extracts PTX; on AMD HIP it compiles an AMDGPU
        HSACO code object (with disk caching) and loads it into a
        ``cupy.cuda.function.Module`` on CuPy's active stream.

        Parameters
        ----------
        func : callable
            GPU kernel function written with ``cuda.grid`` primitives.
        """

        def __init__(self, func):
            self.python_func = func
            self.kernel_dict = {}
            self.is_specialized = False
            self.specialized_kernel = None
            _REGISTERED_KERNELS.append(
                (func.__module__, func.__qualname__, self)
            )

        # --- CUDA path (unchanged upstream behavior) ---
        def _make_cupy_kernel_cuda(self, numba_kernel):
            if numba_version < (0, 56):
                ptx_code = numba_kernel.ptx
            else:
                ptx_code = numba_kernel._codelibrary.get_asm_str()
            ptx_code = ptx_code.encode('ascii')
            module = cupy.cuda.function.Module()
            module.load(ptx_code)
            return module.get_function(numba_kernel.entry_name)

        # --- HIP path (K1 + disk HSACO cache) ---
        def _load_or_compile_hip(self, argtypes):
            arch_info = get_device_arch_info()
            amdgpu_arch = arch_info['arch']
            wave_size = arch_info['warp_size']

            use_cache = not _is_truthy_env('FBPIC_DISABLE_KERNEL_CACHE')
            cache_path = None
            if use_cache:
                cache_dir = _get_hip_cache_dir()
                func_fp = _fingerprint_callable(self.python_func)
                sig_str = ','.join(str(t) for t in argtypes)
                key_payload = (
                    f"{func_fp}|{sig_str}|{amdgpu_arch}|w{wave_size}|"
                    f"{_hip_compiler_fingerprint()}"
                )
                digest = hashlib.sha256(
                    key_payload.encode('utf-8')
                ).hexdigest()[:32]
                safe_name = self.python_func.__name__[:48]
                cache_path = os.path.join(
                    cache_dir,
                    f"{safe_name}_{amdgpu_arch}_w{wave_size}_{digest}.hsaco",
                )
                if os.path.isfile(cache_path):
                    t0 = time.perf_counter()
                    try:
                        with open(cache_path, 'rb') as f:
                            magic = f.readline()
                            if magic == b'FBPIC_HSACO_V1\n':
                                hdr = json.loads(f.readline().decode('utf-8'))
                                codeobj = f.read()
                                valid = (
                                    len(codeobj) == hdr['size']
                                    and len(codeobj) > 0
                                )
                                if valid:
                                    mod = cupy.cuda.function.Module()
                                    mod.load(codeobj)
                                    fn = mod.get_function(hdr['entry_name'])
                                    _KERNEL_CACHE_STATS['hits'] += 1
                                    _KERNEL_CACHE_STATS['load_time_s'] += (
                                        time.perf_counter() - t0
                                    )
                                    return fn
                    except Exception:
                        pass

            t0 = time.perf_counter()
            entry_name, codeobj = compile_hip_kernel_to_codeobj(
                self.python_func, tuple(argtypes),
                amdgpu_arch=amdgpu_arch, wave_size=wave_size,
            )
            mod = cupy.cuda.function.Module()
            mod.load(codeobj)
            fn = mod.get_function(entry_name)
            _KERNEL_CACHE_STATS['misses'] += 1
            _KERNEL_CACHE_STATS['compile_time_s'] += (time.perf_counter() - t0)

            if use_cache and cache_path is not None:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    fd, tmp_path = tempfile.mkstemp(
                        dir=os.path.dirname(cache_path), prefix='.tmp_hsaco_'
                    )
                    try:
                        with os.fdopen(fd, 'wb') as tf:
                            tf.write(b'FBPIC_HSACO_V1\n')
                            hdr = json.dumps({
                                'entry_name': entry_name,
                                'size': len(codeobj),
                            }).encode('utf-8')
                            tf.write(hdr + b'\n')
                            tf.write(codeobj)
                        os.replace(tmp_path, cache_path)
                    except Exception:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                        raise
                except Exception:
                    pass

            return fn

        def make_cupy_kernel(self, numba_kernel):
            """Compatibility helper for CUDA PTX loading."""
            if gpu_backend == 'hip':
                raise RuntimeError(
                    "make_cupy_kernel(numba_kernel) is CUDA-only; "
                    "HIP compiles code objects directly."
                )
            return self._make_cupy_kernel_cuda(numba_kernel)

        def specialize(self, signature):
            """
            Specialize a kernel immediately for an explicit Numba signature.

            Parameters
            ----------
            signature : numba.core.typing.templates.Signature
                Explicit Numba kernel signature.

            Returns
            -------
            self : CompileCupy
                Specialized kernel wrapper.
            """
            if gpu_backend == 'hip':
                argtypes, _ = sigutils.normalize_signature(signature)
                self.specialized_kernel = self._load_or_compile_hip(argtypes)
            else:
                numba_kernel = cuda.jit(signature)(self.python_func)
                self.specialized_kernel = self._make_cupy_kernel_cuda(
                    numba_kernel
                )
            self.is_specialized = True
            return self

        def compile_for_target(
            self, argtypes, amdgpu_arch, wave_size=None, extra_llvm_opts=None,
        ):
            """
            Compile this kernel to an HSACO binary for a given AMD target.

            Parameters
            ----------
            argtypes : tuple of numba.types.Type
                Kernel argument types.

            amdgpu_arch : str
                Target AMDGPU architecture (e.g. ``'gfx942'``).

            wave_size : int or None, optional
                Wavefront size override when invoking external LLVM tools.

            extra_llvm_opts : list of str or None, optional
                Optional flags passed to ``amdclang``.

            Returns
            -------
            entry_name : str
                Kernel symbol name.

            codeobj : bytes
                Compiled HSACO ELF bytes.
            """
            return compile_hip_kernel_to_codeobj(
                self.python_func, tuple(argtypes),
                amdgpu_arch=amdgpu_arch, wave_size=wave_size,
                extra_llvm_opts=extra_llvm_opts,
            )

        def __getitem__(self, bt):
            blocks_per_grid = bt[0]
            threads_per_block = bt[1]
            if not isinstance(blocks_per_grid, tuple):
                blocks_per_grid = (blocks_per_grid,)
            if not isinstance(threads_per_block, tuple):
                threads_per_block = (threads_per_block,)

            def call_kernel(*args):
                if self.is_specialized:
                    kernel = self.specialized_kernel
                else:
                    args_hash = get_args_hash(args)
                    if args_hash not in self.kernel_dict:
                        if gpu_backend == 'hip':
                            argtypes = tuple(
                                _arg_to_numba_type(a) for a in args
                            )
                            self.kernel_dict[args_hash] = (
                                self._load_or_compile_hip(argtypes)
                            )
                        else:
                            numba_kernel = cuda.jit()(
                                self.python_func
                            ).specialize(*args)
                            self.kernel_dict[args_hash] = (
                                self._make_cupy_kernel_cuda(numba_kernel)
                            )
                    kernel = self.kernel_dict[args_hash]

                kernel_args = _flatten_kernel_args(args)
                kernel(blocks_per_grid, threads_per_block, kernel_args)

            return call_kernel

else:

    class CompileCupy(object):
        """
        Placeholder decorator when no GPU backend is active.

        Parameters
        ----------
        func : callable
            GPU kernel function.
        """

        def __init__(self, func):
            self.python_func = func

        def specialize(self, signature):
            """No-op specialization in CPU-only environments."""
            return self

        def __getitem__(self, bt):
            raise RuntimeError(
                f"Cannot launch GPU kernel {self.python_func.__name__}: "
                f"no GPU backend is active (gpu_backend={gpu_backend!r})."
            )


compile_cupy = CompileCupy
