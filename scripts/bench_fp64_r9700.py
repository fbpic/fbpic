#!/usr/bin/env python3
"""Microbenchmark measuring FP64 vs FP32 throughput on AMD Radeon PRO R9700."""
import time
import numpy as np
import cupy
from cupy.cuda import runtime


FMA_KERNEL_SRC = r"""
extern "C" __global__
void fma_bench_f64(
    double* __restrict__ out, const double a, const double b, const int iters
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double x0 = a + 1e-7 * (double)(tid & 255);
    double x1 = x0 + 0.1;
    double x2 = x0 + 0.2;
    double x3 = x0 + 0.3;
    double x4 = x0 + 0.4;
    double x5 = x0 + 0.5;
    double x6 = x0 + 0.6;
    double x7 = x0 + 0.7;
    for (int i = 0; i < iters; ++i) {
        #pragma unroll 16
        for (int k = 0; k < 16; ++k) {
            x0 = fma(x0, a, b);
            x1 = fma(x1, a, b);
            x2 = fma(x2, a, b);
            x3 = fma(x3, a, b);
            x4 = fma(x4, a, b);
            x5 = fma(x5, a, b);
            x6 = fma(x6, a, b);
            x7 = fma(x7, a, b);
        }
    }
    if (tid == -1) {
        out[0] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
    }
}

extern "C" __global__
void fma_bench_f32(
    float* __restrict__ out, const float a, const float b, const int iters
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x0 = a + 1e-5f * (float)(tid & 255);
    float x1 = x0 + 0.1f;
    float x2 = x0 + 0.2f;
    float x3 = x0 + 0.3f;
    float x4 = x0 + 0.4f;
    float x5 = x0 + 0.5f;
    float x6 = x0 + 0.6f;
    float x7 = x0 + 0.7f;
    for (int i = 0; i < iters; ++i) {
        #pragma unroll 16
        for (int k = 0; k < 16; ++k) {
            x0 = fmaf(x0, a, b);
            x1 = fmaf(x1, a, b);
            x2 = fmaf(x2, a, b);
            x3 = fmaf(x3, a, b);
            x4 = fmaf(x4, a, b);
            x5 = fmaf(x5, a, b);
            x6 = fmaf(x6, a, b);
            x7 = fmaf(x7, a, b);
        }
    }
    if (tid == -1) {
        out[0] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
    }
}
"""


def bench_gemm(m, k, n, dtype, warmup=5, repeats=20):
    """
    Time hipBLAS GEMM ``(m, k) @ (k, n)`` in milliseconds and TFLOPS.

    Parameters
    ----------
    m, k, n : int
        Matrix multiplication dimensions.

    dtype : cupy.dtype
        Element floating-point dtype.

    warmup, repeats : int, optional
        Number of warm-up and timed iterations.

    Returns
    -------
    tuple of (float, float)
        Mean time in milliseconds and achieved TFLOPS.
    """
    a = cupy.ones((m, k), dtype=dtype)
    b = cupy.ones((k, n), dtype=dtype)
    for _ in range(warmup):
        c = a @ b
    cupy.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
    cupy.cuda.Stream.null.synchronize()
    dt = (time.perf_counter() - t0) / repeats
    flops = 2.0 * m * n * k
    tflops = flops / dt / 1e12
    del c
    return dt * 1e3, tflops


def bench_fma(
    kernel, out_buf, scalar_a, scalar_b, n_threads, iters,
    warmup=3, repeats=10,
):
    """
    Time pure FMA throughput kernel in milliseconds and TFLOPS.

    Parameters
    ----------
    kernel : cupy.RawKernel
        Compiled FMA microbenchmark kernel.

    out_buf : cupy.ndarray
        Dummy output buffer preventing dead-code elimination.

    scalar_a, scalar_b : float
        FMA operands passed to the kernel.

    n_threads, iters : int
        Total GPU threads and loop iteration count.

    warmup, repeats : int, optional
        Number of warm-up and timed iterations.

    Returns
    -------
    tuple of (float, float)
        Mean time in milliseconds and achieved TFLOPS.
    """
    tpb = 256
    bpg = n_threads // tpb
    args = (out_buf, scalar_a, scalar_b, np.int32(iters))
    for _ in range(warmup):
        kernel((bpg,), (tpb,), args)
    cupy.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        kernel((bpg,), (tpb,), args)
    cupy.cuda.Stream.null.synchronize()
    dt = (time.perf_counter() - t0) / repeats
    # 8 chains * 16 unroll * 2 FLOPs (mul+add) per inner iter
    flops = float(n_threads) * iters * 16 * 8 * 2
    tflops = flops / dt / 1e12
    return dt * 1e3, tflops


def main():
    """Run hipBLAS GEMM and VALU FMA benchmarks on the visible GPU."""
    props = runtime.getDeviceProperties(0)
    raw_name = props["name"]
    name = raw_name.decode() if isinstance(raw_name, bytes) else raw_name
    raw_arch = props["gcnArchName"]
    arch = raw_arch.decode() if isinstance(raw_arch, bytes) else raw_arch
    print(
        f"Device 0: {name} ({arch}), CUs={props['multiProcessorCount']}, "
        f"warpSize={props['warpSize']}"
    )

    print("\n=== hipBLAS GEMM Benchmark ===")
    for label, (m, k, n) in [
        ("Square 2048", (2048, 2048, 2048)),
        ("Square 4096", (4096, 4096, 4096)),
        ("Hankel (256x256 @ 256x2048)", (256, 256, 2048)),
    ]:
        ms32, tf32 = bench_gemm(m, k, n, cupy.float32, warmup=5, repeats=20)
        ms64, tf64 = bench_gemm(m, k, n, cupy.float64, warmup=5, repeats=20)
        print(
            f"  {label:30s} | FP32: {ms32:7.3f} ms ({tf32:6.3f} TFLOPS) | "
            f"FP64: {ms64:7.3f} ms ({tf64:6.3f} TFLOPS) | "
            f"ratio={tf32 / tf64:5.1f}x"
        )

    print("\n=== Pure VALU FMA Microbenchmark ===")
    mod = cupy.RawModule(code=FMA_KERNEL_SRC)
    k_f64 = mod.get_function("fma_bench_f64")
    k_f32 = mod.get_function("fma_bench_f32")
    out64 = cupy.zeros(1, dtype=cupy.float64)
    out32 = cupy.zeros(1, dtype=cupy.float32)

    n_threads = 65536 * 4
    ms32, tf32 = bench_fma(
        k_f32, out32, np.float32(1.0001), np.float32(0.0001),
        n_threads, iters=4000,
    )
    ms64, tf64 = bench_fma(
        k_f64, out64, np.float64(1.000001), np.float64(0.000001),
        n_threads, iters=500,
    )
    print(f"  FP32 FMA: {ms32:7.3f} ms ({tf32:6.3f} TFLOPS)")
    print(f"  FP64 FMA: {ms64:7.3f} ms ({tf64:6.3f} TFLOPS)")
    print(f"  FP32:FP64 FMA ratio = {tf32 / tf64:.2f}x")


if __name__ == "__main__":
    main()
