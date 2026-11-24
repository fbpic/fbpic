# Running FBPIC on AMD GPUs (ROCm / HIP)

FBPIC supports AMD GPUs via ROCm / HIP behind the standard `use_cuda=True` API, so existing simulation scripts work without modification while NVIDIA CUDA and multi-core CPU backends keep their upstream behavior intact.

---

## 1. Supported Hardware & Verification Scope

| Architecture | Target (`gcnArchName`) | Example GPUs | Wavefront | FP64 Global Atomics | Status |
|---|---|---|---|---|---|
| **RDNA4** | `gfx1201` | Radeon AI PRO R9700 (32 GB) | 32 | `global_atomic_cmpswap_b64` (`SCOPE_DEV`) | **Hardware-validated** (upstream suite 51/51, acceptance 14/14, LWFA benchmark) |
| **CDNA2** | `gfx90a` | Instinct MI210 / MI250 / MI250X | 64 | Native `global_atomic_add_f64` | Codegen, wave64 launch config & ISA verified offline |
| **CDNA3** | `gfx942` | Instinct MI300X / MI300A / MI325X | 64 | Native `global_atomic_add_f64` | Codegen, wave64 launch config & ISA verified offline |
| **CDNA4** | `gfx950` | Instinct MI350X / MI355X | 64 | Native `global_atomic_add_f64` | Codegen, wave64 launch config & ISA verified offline |
| **CDNA1** | `gfx908` | Instinct MI100 | 64 | `global_atomic_cmpswap_x2` | Best-effort (offline codegen & ISA verified) |

Importing `fbpic` on a node with no GPUs installed or no visible GPU devices safely falls back to CPU mode (`cuda_installed=False`, `gpu_backend='cpu'`).

---

## 2. Tested ROCm Software Matrix

Validated on Ubuntu 24.04 (WSL2 kernel `6.18.33.2-microsoft-standard-WSL2`, ROCm driver `10.0.0` / HIP `7.15`):

| Package / Component | Pinned Version | Notes |
|---|---|---|
| Python | `3.12.16` | Requires `>= 3.10` for `hip-python` |
| ROCm Core / HIP / Clang | `10.0.0` / HIP `7.15.26333` / `amdclang 23` | `rocm-sdk-{core,devel,libraries,device-gfx1201}==10.0.0` |
| `hip-python` / `hip-python-interop` | `10.0.0.0.1.2` | From `https://pypi.amd.com/rocm-10.0.0/simple` |
| `rocm-bindings-hip` / `rocm-bindings-compiler` | `10.0.0.0.1.2` | Compiler & HIP Python bindings |
| `numba` / `numba-hip` | `0.63.1` / `0.2.1` | Compiles `@compile_cupy` kernels to HSACO ELF code objects |
| `cupy` | `14.1.1` (`is_hip=True`) | **Built from source** with `HCC_AMDGPU_TARGET="gfx1201,gfx90a,gfx942,gfx950"` |
| `mpich` / `mpi4py` | `5.0.2` / `4.1.1` | Conda-forge MPI runtime |

---

## 3. Installation (`scripts/setup_rocm_env.sh`, No `sudo`)

### Why CuPy Is Built From Source
Upstream CuPy has not published ROCm wheels since v13.4, and tested `amd-cupy==14.1.1` wheels on ROCm 10.0.0 exhibited two blockers on RDNA4:
1. Prebuilt `cupy.cuda.thrust` libraries lacked `gfx1201` code objects (`hipErrorInvalidImage` in `argsort`).
2. `amd-cupy` hardcoded `DT_RPATH=/opt/rocm/lib`, loading mismatched RNG libraries when `gfx1201` Tensile/math libraries live inside the Python environment (`rocm-sdk-libraries`).

### Setup Steps (RDNA4 Workstation, WSL2, or Native Linux)
Run the automated setup script (no root/sudo required):

```bash
bash scripts/setup_rocm_env.sh
```

What `scripts/setup_rocm_env.sh` does:
1. Creates the `fbpic-rocm` conda environment (`python=3.12`, `mpich`, `mpi4py=4.1.1`, `scipy`, `h5py`, `pyfftw`, `matplotlib`, `pytest`, `pyflakes`).
2. Installs pinned `rocm-sdk-*==10.0.0`, `hip-python==10.0.0.0.1.2`, `numba==0.63.1`, and `numba-hip==0.2.1`.
3. Assembles a self-contained `$CONDA_PREFIX/rocm_home` merging host core runtime libs (`librocdxg`/`libhsa-runtime64`/`libamdhip64`) with `rocm-sdk-devel` headers and `gfx1201`-enabled math libraries (including symlinking `hipblaslt/library/gfx*` Tensile files so `TensileLibrary_lazy_gfx1201.dat.zlib` resolves cleanly).
4. Builds CuPy `14.1.1` from source (`CUPY_INSTALL_USE_HIP=1`, comma-separated `HCC_AMDGPU_TARGET="gfx1201,gfx90a,gfx942,gfx950"` — `hipcc` 7.15 splits target lists on commas rather than semicolons).
5. Installs conda activation hooks (`ROCM_HOME`, `ROCBLAS_TENSILE_LIBPATH`, `OMP_WAIT_POLICY=PASSIVE`) and installs `fbpic` in editable mode (`pip install --no-deps --no-build-isolation -e .`).
6. Verifies the installation with `scripts/smoke_rocm.py`.

---

## 4. Platform Runbooks

### 4.1 RDNA4 (`gfx1201`) on WSL2 (`/dev/dxg`)

```bash
conda activate fbpic-rocm
unset HIP_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=1   # Device index of discrete R9700 on dual-GPU APU+dGPU hosts
export HSA_ENABLE_SDMA=0
export OMP_WAIT_POLICY=PASSIVE
export FBPIC_CACHE_DIR="$HOME/.cache/fbpic-rocm/kernels"
```

Key operational notes on WSL2 + external GPUs:
- **Do NOT set `HSA_OVERRIDE_GFX_VERSION`**: `gfx1201` has native ROCm 10.0.0 support.
- **Use `ROCR_VISIBLE_DEVICES`, not `HIP_VISIBLE_DEVICES` on dual-GPU WSL2 hosts**: Filtering at the HIP layer after `librocdxg` opens multiple WDDM adapters can fail with `[CreateContext] fail 11`; filtering at the HSA layer (`ROCR_VISIBLE_DEVICES=1`) isolates the R9700 cleanly.
- **Avoid concurrent WSL GPU launches & external dock TDR timeouts**: Do not run multiple independent GPU processes concurrently through `/dev/dxg`, and keep `FBPIC_CACHE_DIR` enabled on Linux ext4 (`$HOME/.cache/...`, never `/mnt/c`) so subsequent runs skip the ~62 s cold JIT compile.
- **Monitoring GPU activity on Windows / WSL2**: Run `/opt/rocm-wsl/bin/amd-smi monitor` inside WSL2 (`GFX-Uti`, `Mem-Uti`, `Power`, `Mem-Usage`). In **Windows Task Manager**, WDDM compute work from WSL2 appears under the **`Compute 0`** and **`Copy`** engine graphs on the R9700 rather than the `3D` graph.

### 4.2 Bare-Metal CDNA Clusters (MI210 / MI250X / MI300X / MI350X)

On native Linux nodes with KFD (`/dev/kfd`), `conda activate fbpic-rocm` and launch normally (no `ROCR_VISIBLE_DEVICES` workaround or `HSA_ENABLE_SDMA=0` needed unless your scheduler manages GPU isolation):

```bash
export FBPIC_GPU_BACKEND=hip
export FBPIC_CACHE_DIR="${SCRATCH:-$HOME}/.cache/fbpic-rocm/kernels"
mpirun -np <N_GPUS> python your_simulation.py
```

FBPIC selects `rank % n_gpus` cleanly under `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` and detects oversubscription via `hipDeviceGetUuid`, `amdsmi`, or PCI BDF fallback (emitting a warning if two MPI ranks share the same physical GPU).

---

## 5. Backend Selection & Environment Variables

| Variable | Values / Default | Purpose |
|---|---|---|
| `FBPIC_GPU_BACKEND` | Auto-detected (`hip` / `cuda` / `cpu`) | Force a specific backend. Explicit `hip` or `cuda` raises `RuntimeError` if hardware/runtime is unavailable rather than silently falling back to CPU. |
| `FBPIC_CACHE_DIR` | `~/.cache/fbpic-rocm/kernels` | Disk cache directory for compiled HSACO ELF code objects (`_v2` fingerprint). |
| `FBPIC_DISABLE_KERNEL_CACHE` | unset (`0`) / `1` | Disable disk HSACO caching and force fresh JIT compilation. |
| `FBPIC_HIP_WAVE64` | unset (`0`) / `1` | Exercise wave64 (`warp_size=64`) launch-geometry selection on wave32 GPUs (used for CI testing of CDNA block rules on RDNA4). |
| `FBPIC_HIP_COPY_TPB` | Auto: `(4, 32)` | Override `"TPBx,TPBy"` for Fourier/Hankel 2D copy kernels (auto-aligned to device `warpSize`). |
| `FBPIC_HIP_SPECTRAL_TPB` | Auto: `(4, 32)` RDNA, `(1, 64)` CDNA | Override `"TPBx,TPBy"` for spectral push/filter kernels. |
| `FBPIC_HIP_DEPOSIT_TPB` | Auto: `64` | Override 1D deposition block size (multiple of `warpSize`). |
| `FBPIC_HIP_GATHER_TPB` | Auto: `256` | Override 1D gather block size (multiple of `warpSize`). |

---

## 6. Architecture & Performance Notes

- **K1 Compilation Path**: `@compile_cupy` kernels stay single-source Numba-Python. On HIP, `fbpic/utils/gpu_backend.py` compiles kernels once via `numba.hip.compiler.compile_hip` to an AMDGPU ELF code object (`.hsaco`) and loads/launches it through `cupy.cuda.function.Module` on CuPy's active stream — keeping zero host synchronizations inside `sim.step()` and reducing dispatch latency by ~10x versus direct `numba-hip` CAI launches. Cold LWFA JIT compilation (~61.6 s for 29 kernels) drops to **0.110 s** (`560x` faster) with the warm HSACO cache.
- **FP64 Throughput Profile on RDNA4 vs CDNA**:
  - On the Radeon AI PRO R9700 (`gfx1201`), measured FP64 hardware throughput is **0.87–0.93 TFLOPS** vs **14.2–15.9 TFLOPS** FP32 (~`16:1`), making the double-precision spectral transforms (`hipBLAS dgemm` Hankel + `rocFFT` Z2Z FFT) ~54% of steady-state step time (`32.45 ms/step` out of `~76 ms/step` at `Nz=1024, Nr=256, Nm=2`). Even so, steady-state GPU simulation (`76.288 ms/step`) is **`6.35x` faster** than the 32-core passive CPU backend (`484.560 ms/step`).
  - On CDNA2/CDNA3 (`gfx90a`/`gfx942`), hardware FP64 throughput and native `global_atomic_add_f64` eliminate both the 16:1 FP64 bottleneck and the RDNA4 FP64 atomic CAS loop.
- **Coarse-Grained `hipMalloc` Requirement for FP64 Atomics**:
  - `fbpic/utils/gpu_backend.py` implements `_AgentAtomicRMW` casting global pointers to `addrspace(1)` with `syncscope("agent")` and `amdgpu.no.fine.grained.memory` / `amdgpu.no.remote.memory` metadata, lowering directly to `global_atomic_add_f64` on `gfx90a`/`gfx942`/`gfx950` and device-scoped `global_atomic_cmpswap_b64` on `gfx1201`.
  - Deposition and synchrotron radiation buffers (`Jr`, `Jt`, `Jz`, `rho`, `radiation_data`) **must** remain in CuPy's default coarse-grained `hipMalloc` VRAM — never managed or host-pinned/mapped allocations, on which ROCm float atomics can NOP or corrupt values.

---

## 7. Known Limitations

1. **CDNA runtime testing**: `gfx90a`, `gfx942`, `gfx950`, and `gfx908` are verified across all 76 `@compile_cupy` kernels (80 signatures), ELF wavefront metadata, and atomic ISA output offline (`scripts/verify_cdna_targets.py`), and `FBPIC_HIP_WAVE64=1` is verified live on `gfx1201`; live end-to-end execution on physical MI2xx/MI3xx hardware has not yet been run here.
2. **WSL2 `rocprofv3` limitations**: Under WSL2 `/dev/dxg` (no `/dev/kfd`), `rocprofv3 --kernel-trace` writes no hardware AQL kernel CSV, and `rocprofv3 --hip-trace` collides on duplicate `spirv-expand-step` LLVM option registration when `numba-hip`/`rocm-bindings-compiler` loads. Use `cupy.cuda.Event` GPU timing and `AMD_LOG_LEVEL=3` on WSL2, or `rocprofv3` / `rocprof-compute` on bare-metal Linux CDNA nodes.
3. **Host-staged MPI**: Boundary field/particle halo exchanges copy via host buffers (`send_data_to_gpu`/`receive_data_from_gpu`); GPU-aware MPI (ROCm RDMA) is not implemented.
4. **Precision**: FBPIC runs exclusively in `float64`/`complex128`; a `float32`/mixed-precision mode is not implemented.
5. **Native Windows**: Supported via WSL2 Ubuntu; native Windows ROCm builds are untested and unsupported.
