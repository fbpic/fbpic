# Change Log / Release Log for fbpic

## 0.2.0

This version incorporates the Galilean scheme, in order to prevent the 
numerical Cherenkov instability for a plasma with uniform velocity. The
user can now choose to run the simulation with either the standard PSATD, or
with the Galilean PSATD scheme.

In addition, several improvements were made to the code:
- The user can now choose to have each MPI rank run an independent simulation (e.g. for parameter scans). A corresponding example has been added `docs/source/example_input`. 
- `matplotlib` was removed from the code's dependencies, and is not imported anymore, as it was sometimes slow to load.
- The option `ptcl_feedback` has been removed since it was seldom used.
- The field push is now done with numba functions instead of numpy functions.