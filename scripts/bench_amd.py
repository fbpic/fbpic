#!/usr/bin/env python
# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""
LWFA benchmark (§6.6) comparing AMD ROCm/HIP against all-core CPU execution.

Default case follows `docs/source/example_input/lwfa_script.py` scaled to
`Nz = 1024, Nr = 256, Nm = 2` with diagnostics disabled, running 20 warm-up
steps followed by 200 timed steps in three modes:
  1. GPU (cold cache): fresh temporary FBPIC_CACHE_DIR + CUPY_CACHE_DIR
  2. GPU (cached): reusing the populated cache (with optional amd-smi sample)
  3. CPU (all logical cores): OMP_WAIT_POLICY=PASSIVE
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.constants import c, e, m_e


def _run_worker(args):
    from fbpic.main import Simulation
    from fbpic.lpa_utils.laser import add_laser_pulse, GaussianLaser
    from fbpic.utils.cuda import (
        cuda, cuda_installed, gpu_backend,
        get_device_arch_info, get_kernel_cache_stats,
        reset_kernel_cache_stats,
    )
    from fbpic.utils.random_seed import set_random_seed

    use_gpu = (args.backend != 'cpu')
    if use_gpu and not cuda_installed:
        raise RuntimeError(
            f"GPU mode requested (backend={args.backend}), "
            f"but gpu_backend={gpu_backend}"
        )

    set_random_seed(42)
    nz, nr, nm = args.nz, args.nr, args.nm
    zmin, zmax = -10.e-6, 30.e-6
    rmax = 80.e-6
    dt = (zmax - zmin) / nz / c

    reset_kernel_cache_stats()
    t_init0 = time.perf_counter()
    sim = Simulation(
        nz, zmax, nr, rmax, nm, dt,
        zmin=zmin, n_order=-1,
        boundaries={'z': 'open', 'r': 'reflective'},
        initialize_ions=False,
        use_cuda=use_gpu,
        verbose_level=0,
    )

    ramp_start = -5.e-6
    ramp_length = 20.e-6

    def dens_func(z, r):
        n = np.ones_like(z)
        ramp = (z - ramp_start) / ramp_length
        n = np.where(z < ramp_start + ramp_length, ramp, n)
        n = np.where(z < ramp_start, 0., n)
        return n

    sim.add_new_species(
        q=-e, m=m_e, n=4.e24, dens_func=dens_func,
        p_zmin=-5.e-6, p_zmax=500.e-6, p_rmax=70.e-6,
        p_nz=2, p_nr=2, p_nt=4,
    )
    add_laser_pulse(sim, GaussianLaser(4.0, 12.e-6, 16.e-15, 15.e-6))
    sim.set_moving_window(v=c)
    sim.diags = []

    # Warm-up (triggers all kernel JIT/load + moving-window paths)
    t_warm0 = time.perf_counter()
    sim.step(args.warmup, show_progress=False)
    if use_gpu:
        cuda.synchronize()
    t_warm1 = time.perf_counter()
    cache_stats = get_kernel_cache_stats()

    # Timed steady-state window; sync periodically for WDDM TDR safety
    chunk = 20
    remaining = args.steps
    smi_proc = None
    smi_out = None
    smi_bin = '/opt/rocm-wsl/bin/amd-smi'
    if use_gpu and args.sample_smi and os.path.exists(smi_bin):
        smi_out = tempfile.NamedTemporaryFile(
            prefix='fbpic_smi_', suffix='.txt', delete=False
        )
        smi_out.close()
        smi_proc = subprocess.Popen(
            ['bash', '-c', f'sleep 2 && {smi_bin} > {smi_out.name} 2>&1'],
        )

    t_run0 = time.perf_counter()
    while remaining > 0:
        n = min(chunk, remaining)
        sim.step(n, show_progress=False)
        if use_gpu:
            cuda.synchronize()
        remaining -= n
    t_run1 = time.perf_counter()

    smi_text = ''
    if smi_proc is not None:
        try:
            smi_proc.wait(timeout=15)
            with open(
                smi_out.name, 'r', encoding='utf-8', errors='replace'
            ) as f:
                smi_text = f.read()
        finally:
            if os.path.exists(smi_out.name):
                os.unlink(smi_out.name)

    arch = (
        get_device_arch_info()
        if use_gpu
        else {'name': 'CPU', 'arch': 'cpu'}
    )
    payload = {
        'mode': args.mode_label,
        'backend': gpu_backend,
        'device': arch['name'],
        'arch': arch['arch'],
        'nz': nz,
        'nr': nr,
        'nm': nm,
        'ntot_final': int(sim.ptcl[0].Ntot),
        'warmup_steps': args.warmup,
        'timed_steps': args.steps,
        'init_wall_s': round(t_warm0 - t_init0, 3),
        'warmup_wall_s': round(t_warm1 - t_warm0, 3),
        'kernel_compile_s': round(cache_stats['compile_time_s'], 3),
        'kernel_load_s': round(cache_stats['load_time_s'], 4),
        'cache_hits': int(cache_stats['hits']),
        'cache_misses': int(cache_stats['misses']),
        'steady_total_s': round(t_run1 - t_run0, 3),
        'steady_ms_per_step': round((t_run1 - t_run0) * 1e3 / args.steps, 3),
        'smi_snapshot': smi_text.strip(),
    }
    print('BENCH_JSON:' + json.dumps(payload), flush=True)


def _launch_sub(mode, extra_env, args, sample_smi=False):
    env = os.environ.copy()
    env.pop('HIP_VISIBLE_DEVICES', None)
    env.update(extra_env)
    cmd = [
        sys.executable, '-u', os.path.abspath(__file__),
        '--worker',
        '--mode-label', mode,
        '--backend', 'cpu' if mode == 'cpu' else 'hip',
        '--nz', str(args.nz),
        '--nr', str(args.nr),
        '--nm', str(args.nm),
        '--warmup', str(args.warmup),
        '--steps', str(args.steps),
    ]
    if sample_smi:
        cmd.append('--sample-smi')
    res = subprocess.run(
        cmd, env=env, capture_output=True, text=True, check=False
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"Benchmark mode '{mode}' failed (rc={res.returncode}):\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    for line in res.stdout.splitlines():
        if line.startswith('BENCH_JSON:'):
            return json.loads(line[len('BENCH_JSON:'):])
    raise RuntimeError(
        f"No BENCH_JSON line in output for '{mode}':\n"
        f"{res.stdout}\n{res.stderr}"
    )


def main():
    """Run cold-JIT GPU, warm GPU, and CPU LWFA benchmarks."""
    parser = argparse.ArgumentParser(
        description='FBPIC AMD ROCm vs CPU LWFA benchmark'
    )
    parser.add_argument('--nz', type=int, default=1024)
    parser.add_argument('--nr', type=int, default=256)
    parser.add_argument('--nm', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument(
        '--mode',
        choices=['all', 'gpu-cold', 'gpu-warm', 'cpu'],
        default='all',
    )
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--mode-label', default='gpu-warm')
    parser.add_argument('--backend', default='hip')
    parser.add_argument('--sample-smi', action='store_true')
    args = parser.parse_args()

    if args.worker:
        _run_worker(args)
        return

    cpu_count = os.cpu_count() or 32
    results = []

    if args.mode in ('all', 'gpu-cold', 'gpu-warm'):
        with tempfile.TemporaryDirectory(prefix='fbpic_bench_cache_') as cdir:
            fbpic_cache = os.path.join(cdir, 'kernels')
            cupy_cache = os.path.join(cdir, 'cupy')
            os.makedirs(fbpic_cache, exist_ok=True)
            os.makedirs(cupy_cache, exist_ok=True)
            gpu_env = {
                'ROCR_VISIBLE_DEVICES': os.environ.get(
                    'ROCR_VISIBLE_DEVICES', '1'
                ),
                'HSA_ENABLE_SDMA': '0',
                'FBPIC_GPU_BACKEND': 'hip',
                'FBPIC_CACHE_DIR': fbpic_cache,
                'CUPY_CACHE_DIR': cupy_cache,
                'OMP_WAIT_POLICY': 'PASSIVE',
                'OMP_NUM_THREADS': '8',
                'NUMBA_NUM_THREADS': '8',
                'OPENBLAS_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1',
            }
            if args.mode in ('all', 'gpu-cold'):
                print("Running GPU (cold cache)...", flush=True)
                results.append(
                    _launch_sub('gpu-cold', gpu_env, args, sample_smi=False)
                )
            if args.mode in ('all', 'gpu-warm'):
                if args.mode == 'gpu-warm':
                    # Use persistent cache when invoked directly
                    gpu_env.pop('FBPIC_CACHE_DIR', None)
                    gpu_env.pop('CUPY_CACHE_DIR', None)
                print("Running GPU (warm cache)...", flush=True)
                results.append(
                    _launch_sub('gpu-warm', gpu_env, args, sample_smi=True)
                )

    if args.mode in ('all', 'cpu'):
        cpu_env = {
            'ROCR_VISIBLE_DEVICES': '',
            'CUDA_VISIBLE_DEVICES': '',
            'FBPIC_GPU_BACKEND': 'cpu',
            'OMP_WAIT_POLICY': 'PASSIVE',
            'OMP_NUM_THREADS': str(cpu_count),
            'NUMBA_NUM_THREADS': str(cpu_count),
            'OPENBLAS_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
        }
        print(f"Running CPU ({cpu_count} threads, PASSIVE)...", flush=True)
        results.append(_launch_sub('cpu', cpu_env, args, sample_smi=False))

    print("\n=== FBPIC LWFA Benchmark Summary ===")
    print(
        f"Grid: Nz={args.nz}, Nr={args.nr}, Nm={args.nm} | "
        f"Warmup={args.warmup}, Timed={args.steps}"
    )
    print(
        f"{'Mode':12s} {'Device':28s} {'Init+Warmup(s)':>14s} "
        f"{'JIT/Load(s)':>12s} {'Hits/Miss':>10s} {'Steady(ms/step)':>16s}"
    )
    for r in results:
        init_warm = r['init_wall_s'] + r['warmup_wall_s']
        if r['cache_misses'] > 0:
            jit_load = r['kernel_compile_s']
        else:
            jit_load = r['kernel_load_s']
        hm = f"{r['cache_hits']}/{r['cache_misses']}"
        print(
            f"{r['mode']:12s} {r['device'][:28]:28s} {init_warm:14.2f} "
            f"{jit_load:12.3f} {hm:>10s} {r['steady_ms_per_step']:16.3f}"
        )

    warm = next((r for r in results if r['mode'] == 'gpu-warm'), None)
    cpu_r = next((r for r in results if r['mode'] == 'cpu'), None)
    if warm and warm.get('smi_snapshot'):
        print("\n=== Mid-Run amd-smi Snapshot (gpu-warm) ===")
        print(warm['smi_snapshot'])

    if warm and cpu_r:
        speedup = cpu_r['steady_ms_per_step'] / warm['steady_ms_per_step']
        print(
            f"\nSteady-state GPU speedup vs {cpu_count}-core CPU: "
            f"{speedup:.2f}x"
        )
        if warm['steady_ms_per_step'] >= cpu_r['steady_ms_per_step']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
