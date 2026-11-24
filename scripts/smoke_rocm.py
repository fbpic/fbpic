#!/usr/bin/env python3
"""Phase 1 smoke test for ROCm/HIP/CuPy/numba-hip/hip-python/mpi4py."""
import os
import sys
import numpy as np


def main():
    """Run smoke checks for HIP FFT, BLAS, Thrust sort, and RNG."""
    import cupy
    from cupy.cuda import cufft, cublas, device, thrust, runtime
    import numba
    from numba import hip
    import hip as hip_py
    import mpi4py
    from mpi4py import MPI

    hip_ver = getattr(hip, '__version__', 'installed')
    hip_avail = hip.is_available()
    mpi_lib = MPI.Get_library_version().splitlines()[0]
    rocr_vis = os.environ.get('ROCR_VISIBLE_DEVICES', '<unset>')
    hip_vis = os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')

    print("=== Versions ===")
    print(f"Python:      {sys.version.split()[0]}")
    print(f"CuPy:        {cupy.__version__} (is_hip={runtime.is_hip})")
    print(f"HIP runtime: {runtime.runtimeGetVersion()}")
    print(f"HIP driver:  {runtime.driverGetVersion()}")
    print(f"Numba:       {numba.__version__}")
    print(f"numba-hip:   {hip_ver} (available={hip_avail})")
    print(f"hip-python:  {getattr(hip_py, '__version__', 'installed')}")
    print(f"mpi4py:      {mpi4py.__version__} (MPI={mpi_lib})")
    print(f"ROCR_VISIBLE_DEVICES={rocr_vis}")
    print(f"HIP_VISIBLE_DEVICES={hip_vis}")

    n_dev = runtime.getDeviceCount()
    print(f"\n=== Visible HIP Devices ({n_dev}) ===")
    for i in range(n_dev):
        props = runtime.getDeviceProperties(i)
        name = props.get("name", b"")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        arch = props.get("gcnArchName", b"")
        if isinstance(arch, bytes):
            arch = arch.decode("utf-8", errors="replace")
        mem_gb = props.get('totalGlobalMem', 0) / 1e9
        pci = (
            props.get('pciDomainID'),
            props.get('pciBusID'),
            props.get('pciDeviceID'),
        )
        print(
            f"  [{i}] name={name!r}, gcnArchName={arch!r}, "
            f"warpSize={props.get('warpSize')}, "
            f"CUs={props.get('multiProcessorCount')}, "
            f"mem={mem_gb:.2f} GB, uuid={props.get('uuid')!r}, pci={pci}"
        )

    assert runtime.is_hip, "Expected CuPy ROCm/HIP backend"
    assert hip.is_available(), "Expected numba.hip.is_available() == True"

    # 1. Batched 1D complex128 FFT (matching fbpic fourier.py)
    rng = np.random.default_rng(42)
    nz, nr = 256, 64
    arr_np = (
        rng.standard_normal((nr, nz))
        + 1j * rng.standard_normal((nr, nz))
    )
    arr_np = np.ascontiguousarray(arr_np, dtype=np.complex128)
    d_in = cupy.asarray(arr_np.ravel())
    d_out = cupy.empty_like(d_in)
    plan = cufft.Plan1d(nz, cufft.CUFFT_Z2Z, nr)
    plan.fft(d_in, d_out, cufft.CUFFT_FORWARD)
    fft_gpu = d_out.get().reshape(nr, nz)
    fft_np = np.fft.fft(arr_np, axis=1)
    fft_fwd_err = np.linalg.norm(fft_gpu - fft_np) / np.linalg.norm(fft_np)

    d_back = cupy.empty_like(d_in)
    plan.fft(d_out, d_back, cufft.CUFFT_INVERSE)
    back_gpu = (d_back.get().reshape(nr, nz)) / nz
    fft_inv_err = np.linalg.norm(back_gpu - arr_np) / np.linalg.norm(arr_np)
    print(
        f"\n[FFT] forward rel_err={fft_fwd_err:.3e}, "
        f"inverse rel_err={fft_inv_err:.3e}"
    )
    assert fft_fwd_err < 1e-12, f"FFT forward error too high: {fft_fwd_err}"
    assert fft_inv_err < 1e-12, f"FFT inverse error too high: {fft_inv_err}"

    # 2. hipBLAS dgemm (matching fbpic hankel.py F-order & ptr alpha/beta)
    m, n, k = nr, 2 * nz, nr
    a_np = rng.standard_normal((m, k))
    b_np = rng.standard_normal((k, n))
    d_a = cupy.asarray(np.asfortranarray(a_np, dtype=np.float64))
    d_b = cupy.asarray(np.asfortranarray(b_np, dtype=np.float64))
    d_c = cupy.empty((m, n), dtype=np.float64, order="F")
    alpha = np.array(1.0, dtype=np.float64)
    beta = np.array(0.0, dtype=np.float64)
    handle = device.get_cublas_handle()
    cublas.dgemm(
        handle, 0, 0, m, n, k,
        alpha.ctypes.data, d_a.data.ptr, m,
        d_b.data.ptr, k,
        beta.ctypes.data, d_c.data.ptr, m,
    )
    c_gpu = d_c.get()
    c_np = a_np @ b_np
    dgemm_err = np.linalg.norm(c_gpu - c_np) / np.linalg.norm(c_np)
    print(f"[dgemm] rel_err={dgemm_err:.3e}")
    assert dgemm_err < 1e-12, f"dgemm error too high: {dgemm_err}"

    # 3. rocThrust argsort (matching fbpic cuda_sorting.py)
    n_ptcl = 20000
    keys_np = rng.integers(0, 5000, size=n_ptcl, dtype=np.int32)
    d_keys = cupy.asarray(keys_np)
    d_idx = cupy.arange(n_ptcl, dtype=np.intp)
    thrust.argsort(
        dtype=d_keys.dtype,
        idx_start=d_idx.data.ptr,
        data_start=d_keys.data.ptr,
        keys_start=0,
        shape=d_keys.shape,
    )
    keys_sorted = d_keys.get()
    idx_sorted = d_idx.get()
    assert idx_sorted.dtype == np.intp
    assert np.array_equal(keys_sorted, np.sort(keys_np))
    assert np.array_equal(keys_np[idx_sorted], keys_sorted)
    print(
        f"[thrust.argsort] OK (N={n_ptcl}, idx_dtype={idx_sorted.dtype})"
    )

    # 4. cupy.random, cumsum, pinned memory
    r32 = cupy.random.rand(1024, dtype=cupy.float32)
    r64 = cupy.random.uniform(0.0, 1.0, 1024, dtype=cupy.float64)
    cs = cupy.cumsum(cupy.ones(100, dtype=cupy.int64))
    pinned = cupy.cuda.alloc_pinned_memory(4096)
    assert float(r32.min()) >= 0.0 and float(r32.max()) <= 1.0
    assert float(r64.min()) >= 0.0 and float(r64.max()) <= 1.0
    assert int(cs[-1]) == 100
    assert pinned.ptr != 0
    print("[random / cumsum / pinned] OK")
    print("\nSMOKE TEST PASSED.")


if __name__ == "__main__":
    main()
