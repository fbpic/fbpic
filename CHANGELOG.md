# Change Log / Release Log for fbpic

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
