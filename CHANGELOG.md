# Change Log / Release Log for fbpic

## 0.7.1

This is bug-fix release. It fixes two bugs that were introduced in version
0.7.0:
- The first bug **only affected simulations performed on CPU** (not on GPU), and
typically lead to spuriously high charge density on the axis
(see [#191](https://github.com/fbpic/fbpic/pull/191)).
- The second bug affected restarts from a simulation checkpoint, and typically
lead to incorrect continuous injection of plasma and/or simulations crashing at
restart time (see [#192](https://github.com/fbpic/fbpic/pull/192)).
These two bugs are fixed in version 0.7.1.

## 0.7.0

This version incorporates various new features, optimizations and bug fixes.
See below for a details.

New features:
- The messages printed by FBPIC to the terminal have been improved.
The `Simulation` class now supports a `verbose_level` argument, in order to
choose the desired level of information [#158](https://github.com/fbpic/fbpic/pull/158).
- More self-consistent initialization of the laser field [#150](https://github.com/fbpic/fbpic/pull/150).
The laser initialization now supports arbitrary laser profiles and is always exactly
divergence-free, even for MPI-decomposed simulations. More laser profiles
will be implemented and documented in the next release.

New optimizations:
- The code performs fewer Hankel transforms per iteration, and is thus faster [#161](https://github.com/fbpic/fbpic/pull/161).
- Faster functions for removal/addition of particles on GPU [#179](https://github.com/fbpic/fbpic/pull/179)

Bug fixes:
- The position where plasma starts to be injected (for simulations with moving window,
featuring no plasma initially in the box) has been corrected. This mainly affects boosted-frame
simulations. [#160](https://github.com/fbpic/fbpic/pull/160)
- When restarting simulations from checkpoints, there was a bug in the particle
weights, which is now fixed. [#178](https://github.com/fbpic/fbpic/pull/178)
- The current and charge density are now written in the fields diagnostics
for iteration 0, whereas they were previously set to 0 in the diagnostics for this iteration.
[#178](https://github.com/fbpic/fbpic/pull/178)
- The boosted-frame particle diagnostics used to fail in some cases on GPU
due to an out-of-bound access, which is now fixed. [#169](https://github.com/fbpic/fbpic/pull/169)

Changes related to the installation process:
- FBPIC can now use numba 0.36 with threading [#167](https://github.com/fbpic/fbpic/pull/167) and [#170](https://github.com/fbpic/fbpic/pull/170).
- FBPIC is now able to load MKL on Windows [#177](https://github.com/fbpic/fbpic/pull/177) and has better support when MKL fails to load [#154](https://github.com/fbpic/fbpic/pull/154).
- FBPIC can now run without having MPI installed (for single-GPU or single-CPU node simulations) [#143](https://github.com/fbpic/fbpic/pull/143)

## 0.6.2

This is a bug-fix release. It corrects an important bug that was introduced in version 0.6.1 for the Hankel transform on GPU.

## 0.6.1

This version allows FBPIC to run without `mpi4py` installed, in the case of
single-proc simulations.

In addition, the current deposition on CPU, as well as the Hankel transform
on CPU and GPU have been optimized and should have significantly faster
execution time.

Finally, FBPIC now prints a message during just-in-time compilation.

## 0.6.0

This version allows FBPIC to use the MKL library for FFTs, on CPU. In most cases,
this will result in faster code execution compared to the FFTW library, especially
on CPUs with a large number of cores. FFTW can still be used with FBPIC if MKL is unavailable.

In addition, this version optimizes the number of thread per block on GPU for
costly operations, which should also result faster code execution.

## 0.5.4

This is a bug-fix release. It fixes the initial space-charge calculation by ensuring that:
- this calculation does not erase any pre-existing field (e.g. laser field)
- this calculation gives correct results for multi-CPU/multi-GPU simulations as well

## 0.5.3

This is a bug-fix release. It ensures that threading is only used with the
proper numba version (numba 0.34). It also fixes some issues with the MPI
implementation (esp. in particle bunch initialization and charge conservation).

## 0.5.2

This is a bug-fix release, that solves an issue when using
openPMD-viewer >= 0.6.0 to read the checkpoint files (for a restart simulation).

## 0.5.1

This is a bug-fix release, that solves an issue with the particle
boosted-frame diagnostics, in the case where both the GPU and a moving window
were used.

## 0.5.0

This version brings two majors changes to FBPIC:

- The code now supports **multi-threading** on CPU (when using numba>=0.34),
and is therefore much faster then it used to be on multi-core CPU architectures.

- The code does not rely on the proprietary library `accelerate` anymore, and
uses the open-source library `pyculib` instead. As a consequence, **all** the
dependencies of FBPIC are now open-source.

In addition to these changes, several minor improvements have been made to the
GPU code, including faster sorting routines (`prefix_sum`) and shorter
compilation time (function signatures have been removed).

## 0.4.1

This is a bug-fix release, to solve an issue with the particle boosted-frame
diagnostics, in the case where the GPU is used. The particle
boosted-frame diagnostics now correctly run on the GPU.

## 0.4.0

This version incorporates ionization (ADK model). The implementation is
Lorentz-invariant and thus works in the boosted-frame. The implementation
is also fully compatible with GPU, MPI, openPMD diagnostics (including
boosted openPMD diagnostics), and tracking (e.g. of the produced electrons)

In addition, several improvements were made to the code in general:
- External bunches can now be loaded to the simulation from openPMD files,
or from numpy arrays.
- Particle tracking is now compatible with the boosted openPMD diagnostics.
- The laser can now be injected in the simulation with a temporal chirp.

## 0.3.0

This version incorporates the 3rd order particle shapes, in addition to the
pre-existing 1st order particle shapes. (2nd order particle shapes are not
implemented.)
In addition, particle tracking was implemented (i.e. particle can have unique
IDs which are then output in the openPMD files.)

## 0.2.0

This version incorporates the Galilean scheme, in order to prevent the
numerical Cherenkov instability for a plasma with uniform velocity. The
user can now choose to run the simulation with either the standard PSATD, or
with the Galilean PSATD scheme.

In addition, several improvements were made to the code:
- The user can now choose to have each MPI rank run an independent simulation (e.g. for parameter scans). A corresponding example has been added `docs/source/example_input`.
- The boosted-frame diagnostics can now be used in parallel simulations.
- `matplotlib` was removed from the code's dependencies, and is not imported anymore, as it was sometimes slow to load.
- The implementation of the particle periodic boundaries (for single-proc simulation)
  is more efficient. (The particle position is simply shifted.)
- The option `ptcl_feedback` has been removed since it was seldom used.
- The field push is now done with numba functions instead of numpy functions.
