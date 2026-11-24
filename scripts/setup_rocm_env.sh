#!/usr/bin/env bash
# Setup reproducible ROCm Python environment (fbpic-rocm) without sudo.
# Targets ROCm 10.0.0 (HIP 7.15) on WSL2 Ubuntu 24.04 or native Linux,
# and builds CuPy from source for gfx1201;gfx90a;gfx942;gfx950.
set -euo pipefail

ENV_NAME="${FBPIC_ROCM_ENV:-fbpic-rocm}"
CONDA_BIN="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
if [ ! -x "$CONDA_BIN" ]; then
    CONDA_BIN="$(command -v conda)"
fi
CONDA_ROOT="$(cd "$(dirname "$CONDA_BIN")/.." && pwd)"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL_CACHE="${FBPIC_WHEEL_CACHE:-$HOME/.cache/fbpic-rocm/wheels}"
ROCM_TARGETS="${HCC_AMDGPU_TARGET:-gfx1201,gfx90a,gfx942,gfx950}"

echo "[1/7] Using conda: $CONDA_BIN (root: $CONDA_ROOT)"
if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[2/7] Creating conda env '$ENV_NAME' (Python 3.12 + MPI + scientific stack)..."
    "$CONDA_BIN" create -y -n "$ENV_NAME" -c conda-forge \
        python=3.12 \
        pip \
        mpich \
        mpi4py=4.1.1 \
        scipy \
        h5py \
        pyfftw \
        matplotlib \
        pytest \
        pyflakes
else
    echo "[2/7] Conda env '$ENV_NAME' already exists; ensuring base packages..."
    "$CONDA_BIN" install -y -n "$ENV_NAME" -c conda-forge \
        python=3.12 pip mpich mpi4py=4.1.1 scipy h5py pyfftw matplotlib pytest pyflakes
fi

ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"
PIP_BIN="$ENV_PREFIX/bin/pip"
PY_BIN="$ENV_PREFIX/bin/python"

echo "[3/7] Installing ROCm 10.0.0 SDK, bindings, numba-hip, and CuPy build deps..."
"$PIP_BIN" install --upgrade \
    --extra-index-url https://pypi.amd.com/rocm-10.0.0/simple \
    --extra-index-url https://download.pytorch.org/whl/rocm10.0 \
    "hip-python==10.0.0.0.1.2" \
    "hip-python-interop==10.0.0.0.1.2" \
    "rocm-bindings-hip==10.0.0.0.1.2" \
    "rocm-bindings-compiler==10.0.0.0.1.2" \
    "numba-hip==0.2.1" \
    "numba==0.63.1" \
    "rocm-sdk-core==10.0.0" \
    "rocm-sdk-devel==10.0.0" \
    "rocm-sdk-libraries==10.0.0" \
    "rocm-sdk-device-gfx1201==10.0.0" \
    "Cython>=3.0,<3.2" \
    "fastrlock>=0.8.2"

echo "[4/7] Assembling unified user-space ROCM_HOME at $ENV_PREFIX/rocm_home..."
"$PY_BIN" - << 'PYEOF'
import os
import shutil
import site
import tarfile

env_prefix = os.environ.get("CONDA_PREFIX") or os.path.dirname(os.path.dirname(os.sys.executable))
rocm_home = os.path.join(env_prefix, "rocm_home")
if os.path.exists(rocm_home):
    shutil.rmtree(rocm_home)
os.makedirs(rocm_home, exist_ok=True)

site_dir = site.getsitepackages()[0]
sdk_core = os.path.join(site_dir, "_rocm_sdk_core")
sdk_devel = os.path.join(site_dir, "_rocm_sdk_devel")
sdk_libs = os.path.join(site_dir, "_rocm_sdk_libraries")

devel_tar = os.path.join(site_dir, "rocm_sdk_devel", "_devel.tar")
if os.path.isfile(devel_tar) and not os.path.isdir(os.path.join(sdk_devel, "include", "hipcub")):
    print(f"Unpacking {devel_tar} into {site_dir}...")
    with tarfile.open(devel_tar, "r:*") as tf:
        tf.extractall(site_dir)

for sub in ["bin", "llvm", "amdgcn", "libexec", "share", "core", "core-10", "core-10.0"]:
    src = os.path.join("/opt/rocm", sub)
    if not os.path.exists(src):
        src = os.path.join(sdk_core, sub)
    if os.path.exists(src):
        os.symlink(src, os.path.join(rocm_home, sub))

def merge_tree(src_root, dst_root, skip_prefixes=()):
    if not os.path.isdir(src_root):
        return
    for root, dirs, files in os.walk(src_root, followlinks=True):
        rel = os.path.relpath(root, src_root)
        dst_dir = dst_root if rel == "." else os.path.join(dst_root, rel)
        os.makedirs(dst_dir, exist_ok=True)
        for d in list(dirs):
            # Avoid infinite loops or unneeded huge python/site dirs
            if d == "__pycache__":
                dirs.remove(d)
        for f in files:
            if rel == "." and any(f.startswith(p) for p in skip_prefixes):
                continue
            src_f = os.path.realpath(os.path.join(root, f))
            dst_f = os.path.join(dst_dir, f)
            if os.path.lexists(dst_f):
                os.unlink(dst_f)
            os.symlink(src_f, dst_f)

inc_dst = os.path.join(rocm_home, "include")
for src_inc in [
    "/opt/rocm/include",
    os.path.join(sdk_core, "include"),
    os.path.join(sdk_libs, "include"),
    os.path.join(sdk_devel, "include"),
]:
    merge_tree(src_inc, inc_dst)

lib_dst = os.path.join(rocm_home, "lib")
# 1. System /opt/rocm/lib (provides WSL2 librocdxg, libhsa-runtime64, libamdhip64, libhiprtc)
merge_tree("/opt/rocm/lib", lib_dst)
# 2. _rocm_sdk_devel/lib (skipping runtime/HSA/HIP core libs that must match WSL2 librocdxg)
skip_core = ("libamdhip64", "libhsa-runtime64", "librocdxg", "libamd_comgr", "libhiprtc")
merge_tree(os.path.join(sdk_devel, "lib"), lib_dst, skip_prefixes=skip_core)
# 3. _rocm_sdk_libraries/lib (provides gfx1201 math libs, rocblas/hipblaslt Tensile, librocrand)
merge_tree(os.path.join(sdk_libs, "lib"), lib_dst)

# Ensure hipblaslt top-level library dir symlinks per-arch files (e.g. gfx1201)
hipblaslt_lib_dir = os.path.join(lib_dst, "hipblaslt", "library")
if os.path.isdir(hipblaslt_lib_dir):
    for arch_sub in sorted(os.listdir(hipblaslt_lib_dir)):
        arch_path = os.path.join(hipblaslt_lib_dir, arch_sub)
        if arch_sub.startswith("gfx") and os.path.isdir(arch_path):
            for fname in os.listdir(arch_path):
                dst_f = os.path.join(hipblaslt_lib_dir, fname)
                if not os.path.lexists(dst_f):
                    os.symlink(os.path.join(arch_path, fname), dst_f)

# Ensure unversioned lib*.so linker symlinks point to the latest .so.* in rocm_home/lib
for fname in sorted(os.listdir(lib_dst)):
    if ".so." in fname:
        base = fname.split(".so.")[0] + ".so"
        base_path = os.path.join(lib_dst, base)
        if not os.path.lexists(base_path):
            os.symlink(fname, base_path)

print("Assembled rocm_home at", rocm_home)
PYEOF

echo "[5/7] Building and installing CuPy from source for targets: $ROCM_TARGETS ..."
"$PIP_BIN" uninstall -y amd-cupy 2>/dev/null || true
mkdir -p "$WHEEL_CACHE"
CACHED_WHEEL="$(ls -1 "$WHEEL_CACHE"/cupy-*.whl 2>/dev/null | head -n 1 || true)"
if [ -z "$CACHED_WHEEL" ]; then
    export ROCM_HOME="$ENV_PREFIX/rocm_home"
    export ROCM_PATH="$ENV_PREFIX/rocm_home"
    export HIP_PATH="$ENV_PREFIX/rocm_home"
    export CPLUS_INCLUDE_PATH="$ENV_PREFIX/rocm_home/include"
    export LIBRARY_PATH="$ENV_PREFIX/rocm_home/lib"
    export LD_LIBRARY_PATH="/opt/rocm-wsl/lib:$ENV_PREFIX/rocm_home/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUPY_INSTALL_USE_HIP=1
    export CUPY_NUM_BUILD_JOBS="${CUPY_NUM_BUILD_JOBS:-16}"
    export HCC_AMDGPU_TARGET="$ROCM_TARGETS"
    if ! "$PIP_BIN" wheel --no-deps --no-build-isolation -w "$WHEEL_CACHE" "cupy==14.1.1"; then
        echo "PyPI cupy==14.1.1 sdist build failed; falling back to ROCm/cupy bump-rocm-7-14..."
        rm -f "$WHEEL_CACHE"/cupy-*.whl
        "$PIP_BIN" wheel --no-deps --no-build-isolation -w "$WHEEL_CACHE" \
            "git+https://github.com/ROCm/cupy.git@96e733d1c10b8f55df00e023824cd7a6cd33cba1"
    fi
    CACHED_WHEEL="$(ls -1 "$WHEEL_CACHE"/cupy-*.whl | head -n 1)"
fi
echo "Installing built CuPy wheel: $CACHED_WHEEL"
"$PIP_BIN" install --no-deps --force-reinstall "$CACHED_WHEEL"

echo "[6/7] Installing fbpic test/diagnostic dependencies and editable fbpic..."
"$PIP_BIN" install --upgrade \
    openPMD-viewer \
    lasy \
    "picmistandard<=0.34.0" \
    numexpr \
    periodictable
"$PIP_BIN" install --no-build-isolation --no-deps -e "$REPO_DIR"

echo "[7/7] Writing conda activation hook for ROCm / WSL2 / native caches..."
mkdir -p "$ENV_PREFIX/etc/conda/activate.d"
cat > "$ENV_PREFIX/etc/conda/activate.d/fbpic_rocm.sh" << 'EOF'
# WSL2 DXG environment (amdsmi)
if [ -f /opt/rocm-wsl/.env.sh ]; then
    # shellcheck disable=SC1091
    source /opt/rocm-wsl/.env.sh
fi
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/rocm_home" ]; then
    export ROCM_HOME="$CONDA_PREFIX/rocm_home"
    export ROCM_PATH="$CONDA_PREFIX/rocm_home"
    export HIP_PATH="$CONDA_PREFIX/rocm_home"
    export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/rocm_home/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export LD_LIBRARY_PATH="/opt/rocm-wsl/lib:$CONDA_PREFIX/rocm_home/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if [ -d "$CONDA_PREFIX/rocm_home/lib/rocblas/library" ]; then
        export ROCBLAS_TENSILE_LIBPATH="$CONDA_PREFIX/rocm_home/lib/rocblas/library"
    fi
    if [ -d "$CONDA_PREFIX/rocm_home/lib/hipblaslt/library" ]; then
        export HIPBLASLT_TENSILE_LIBPATH="$CONDA_PREFIX/rocm_home/lib/hipblaslt/library"
    fi
fi

# On WSL2 DXG, disable SDMA queues to avoid librocdxg multi-thread CreateContext issues
if [ -f /opt/rocm/lib/librocdxg.so ] || [ -f /opt/rocm/lib/librocdxg.so.1 ]; then
    export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"
fi

# Avoid 100% CPU OpenMP barrier spin-wait on multi-core hosts
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"

# Keep all JIT and FFT caches on native Linux filesystem (not /mnt/c)
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$HOME/.cache/fbpic-rocm/cupy}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$HOME/.cache/fbpic-rocm/numba}"
export ROCFFT_RTC_CACHE_PATH="${ROCFFT_RTC_CACHE_PATH:-$HOME/.cache/fbpic-rocm/rocfft/rocfft_kernel_cache.db}"
export FBPIC_CACHE_DIR="${FBPIC_CACHE_DIR:-$HOME/.cache/fbpic-rocm/kernels}"
mkdir -p "$CUPY_CACHE_DIR" "$NUMBA_CACHE_DIR" "$(dirname "$ROCFFT_RTC_CACHE_PATH")" "$FBPIC_CACHE_DIR"
EOF
chmod +x "$ENV_PREFIX/etc/conda/activate.d/fbpic_rocm.sh"

echo "Verifying embedded gfx arches in cupy.cuda.thrust:"
THRUST_SO="$("$PY_BIN" -c 'import cupy.cuda.thrust as t; print(t.__file__)')"
strings "$THRUST_SO" | grep -oE 'amdgcn-amd-amdhsa--gfx[0-9a-z]+' | sort -u || true

echo "Done. Activate with: conda activate $ENV_NAME"
echo "On dual-GPU systems where R9700 (gfx1201) is device 1, export ROCR_VISIBLE_DEVICES=1"
