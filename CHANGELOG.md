# Change Log / Release Log for fbpic

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
