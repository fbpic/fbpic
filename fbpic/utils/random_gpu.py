# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""
Backend-neutral xoroshiro128+ PRNG for GPU kernels (CUDA & AMD ROCm/HIP).

Matches Numba's ``numba.cuda.random`` SplitMix64 seeding, 2**64-step
subsequence jumping, uint64-to-unit conversion, and Box-Muller transform.
States are stored as a plain 2D ``(n, 2)`` ``uint64`` array so they work
directly with ``@compile_cupy`` across both CUDA and HIP without relying
on ``numba.cuda.random`` (missing in ``numba-hip``).
"""
import math
import numpy as np
from numba import njit, float32, float64, uint32, uint64

from fbpic.utils.gpu_backend import cuda, cuda_installed, cupy

_TWO_PI_FLOAT64 = float64(2.0 * math.pi)
_INV_2_53 = float64(1.0) / float64(uint64(1) << uint64(53))


@njit
def _rotl64(x, k):
    return uint64((uint64(x) << uint32(k)) | (uint64(x) >> uint32(64 - k)))


@njit
def _init_state_cpu(states, index, seed):
    seed_u = uint64(seed)
    z = seed_u + uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> uint32(30))) * uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> uint32(27))) * uint64(0x94D049BB133111EB)
    z = z ^ (z >> uint32(31))
    states[index, 0] = z
    states[index, 1] = z


@njit
def _next_cpu(states, index):
    s0 = states[index, 0]
    s1 = states[index, 1]
    result = uint64(s0 + s1)
    s1 ^= s0
    states[index, 0] = _rotl64(s0, 55) ^ s1 ^ (s1 << uint32(14))
    states[index, 1] = _rotl64(s1, 36)
    return result


@njit
def _jump_cpu(states, index):
    jump0 = uint64(0xbeac0467eba5facb)
    jump1 = uint64(0xd86b048b86aa9922)
    ns0 = uint64(0)
    ns1 = uint64(0)
    for b in range(64):
        if jump0 & (uint64(1) << uint32(b)):
            ns0 ^= states[index, 0]
            ns1 ^= states[index, 1]
        _next_cpu(states, index)
    for b in range(64):
        if jump1 & (uint64(1) << uint32(b)):
            ns0 ^= states[index, 0]
            ns1 ^= states[index, 1]
        _next_cpu(states, index)
    states[index, 0] = ns0
    states[index, 1] = ns1


@njit
def init_xoroshiro128p_states_cpu(n, seed, subsequence_start=0):
    """
    Initialize ``(n, 2)`` ``uint64`` xoroshiro128+ states on CPU.

    Parameters
    ----------
    n : int
        Number of independent RNG streams to allocate.

    seed : int
        Initial scalar seed passed through SplitMix64.

    subsequence_start : int, optional
        Number of ``2**64``-step jumps applied before stream 0.

    Returns
    -------
    states : ndarray of uint64, shape (n, 2)
        Initialized xoroshiro128+ state array on CPU.
    """
    states = np.empty((n, 2), dtype=np.uint64)
    if n <= 0:
        return states
    _init_state_cpu(states, 0, seed)
    for _ in range(subsequence_start):
        _jump_cpu(states, 0)
    for i in range(1, n):
        states[i, 0] = states[i - 1, 0]
        states[i, 1] = states[i - 1, 1]
        _jump_cpu(states, i)
    return states


def create_xoroshiro128p_states(n, seed, subsequence_start=0, on_gpu=True):
    """
    Create and initialize ``(n, 2)`` ``uint64`` xoroshiro128+ states.

    Each row ``i`` is spaced by ``2**64`` generator steps using SplitMix64
    seeding + polynomial jump, matching
    ``numba.cuda.random.create_xoroshiro128p_states``.

    Parameters
    ----------
    n : int
        Number of independent RNG streams to allocate.

    seed : int
        Initial scalar seed passed through SplitMix64.

    subsequence_start : int, optional
        Number of ``2**64``-step jumps applied before stream 0.

    on_gpu : bool, optional
        When True and a GPU backend is active, transfer states to GPU.

    Returns
    -------
    states : ndarray or cupy.ndarray of uint64, shape (n, 2)
        Initialized xoroshiro128+ state array.
    """
    states_np = init_xoroshiro128p_states_cpu(
        int(n), np.int64(seed), int(subsequence_start)
    )
    if on_gpu and cuda_installed and cupy is not None:
        return cupy.asarray(states_np)
    return states_np


# ---------------------------------------------------------------------------
# Device / CPU inlineable draw primitives
# ---------------------------------------------------------------------------
def _xoroshiro128p_next_py(states, index):
    s0 = states[index, 0]
    s1 = states[index, 1]
    result = uint64(s0 + s1)
    s1 ^= s0
    rotl55 = uint64((s0 << uint32(55)) | (s0 >> uint32(9)))
    rotl36 = uint64((s1 << uint32(36)) | (s1 >> uint32(28)))
    states[index, 0] = rotl55 ^ s1 ^ (s1 << uint32(14))
    states[index, 1] = rotl36
    return result


def _xoroshiro128p_uniform_float64_py(states, index):
    s0 = states[index, 0]
    s1 = states[index, 1]
    result = uint64(s0 + s1)
    s1 ^= s0
    rotl55 = uint64((s0 << uint32(55)) | (s0 >> uint32(9)))
    rotl36 = uint64((s1 << uint32(36)) | (s1 >> uint32(28)))
    states[index, 0] = rotl55 ^ s1 ^ (s1 << uint32(14))
    states[index, 1] = rotl36
    return float64(result >> uint32(11)) * _INV_2_53


def _xoroshiro128p_uniform_float32_py(states, index):
    s0 = states[index, 0]
    s1 = states[index, 1]
    result = uint64(s0 + s1)
    s1 ^= s0
    rotl55 = uint64((s0 << uint32(55)) | (s0 >> uint32(9)))
    rotl36 = uint64((s1 << uint32(36)) | (s1 >> uint32(28)))
    states[index, 0] = rotl55 ^ s1 ^ (s1 << uint32(14))
    states[index, 1] = rotl36
    return float32(float64(result >> uint32(11)) * _INV_2_53)


def _xoroshiro128p_normal_float64_py(states, index):
    # First Box-Muller draw (float32, matching numba.cuda.random)
    s0 = states[index, 0]
    s1 = states[index, 1]
    res1 = uint64(s0 + s1)
    s1 ^= s0
    rotl55 = uint64((s0 << uint32(55)) | (s0 >> uint32(9)))
    rotl36 = uint64((s1 << uint32(36)) | (s1 >> uint32(28)))
    s0_mid = rotl55 ^ s1 ^ (s1 << uint32(14))
    s1_mid = rotl36

    # Second Box-Muller draw
    res2 = uint64(s0_mid + s1_mid)
    s1_mid ^= s0_mid
    rotl55_2 = uint64((s0_mid << uint32(55)) | (s0_mid >> uint32(9)))
    rotl36_2 = uint64((s1_mid << uint32(36)) | (s1_mid >> uint32(28)))
    states[index, 0] = rotl55_2 ^ s1_mid ^ (s1_mid << uint32(14))
    states[index, 1] = rotl36_2

    u1 = float32(float64(res1 >> uint32(11)) * _INV_2_53)
    u2 = float32(float64(res2 >> uint32(11)) * _INV_2_53)
    return math.sqrt(-float64(2.0) * math.log(float64(u1))) * math.cos(
        _TWO_PI_FLOAT64 * float64(u2)
    )


# CPU-compiled variants (usable in unit tests without any GPU)
xoroshiro128p_next_cpu = njit(_xoroshiro128p_next_py)
xoroshiro128p_uniform_float64_cpu = njit(_xoroshiro128p_uniform_float64_py)
xoroshiro128p_uniform_float32_cpu = njit(_xoroshiro128p_uniform_float32_py)
xoroshiro128p_normal_float64_cpu = njit(_xoroshiro128p_normal_float64_py)

# GPU device-compiled variants
if cuda_installed:
    xoroshiro128p_next = cuda.jit(
        _xoroshiro128p_next_py, device=True, inline=True
    )
    xoroshiro128p_uniform_float64 = cuda.jit(
        _xoroshiro128p_uniform_float64_py, device=True, inline=True
    )
    xoroshiro128p_uniform_float32 = cuda.jit(
        _xoroshiro128p_uniform_float32_py, device=True, inline=True
    )
    xoroshiro128p_normal_float64 = cuda.jit(
        _xoroshiro128p_normal_float64_py, device=True, inline=True
    )
else:
    xoroshiro128p_next = xoroshiro128p_next_cpu
    xoroshiro128p_uniform_float64 = xoroshiro128p_uniform_float64_cpu
    xoroshiro128p_uniform_float32 = xoroshiro128p_uniform_float32_cpu
    xoroshiro128p_normal_float64 = xoroshiro128p_normal_float64_cpu
