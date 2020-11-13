# Change Log / Release Log for fbpic

## 0.19.1

This release incorporates a small fix that allows the code to compile with
Python 3.8. (see [#488](https://github.com/fbpic/fbpic/issues/488))

## 0.19.0

This release makes the computation of the laser profiles faster, in particular
in the case when the laser is emitted with the antenna and the profile
thus needs to be computed at every time step.

- When using the laser antenna, the laser profile can now be computed on
GPU, if the profile has the flag `gpu_capable=True`.
(see [#473](https://github.com/fbpic/fbpic/pull/473))
- The flattened Gaussian laser was refactored and is now much faster to
compute. (see [#486](https://github.com/fbpic/fbpic/pull/486))

## 0.18.0

This release allows FBPIC to run on GPU with the latest version
of `numba`, by resolving a minor compatibility issue
(see [#482](https://github.com/fbpic/fbpic/pull/482)).

It also makes the `ExternalField` faster on GPU (see [#470](https://github.com/fbpic/fbpic/pull/470)).

## 0.17.1

This minor release removes restrictions on the use of recent versions of
numba, when running on GPU.

## 0.17.0

This release introduces a major change to the treatment of particles close to
the axis (see [#347](https://github.com/fbpic/fbpic/pull/347)).
As a result, the code is much more robust when a large amount of
particles simultaneously cross the axis, and concentrate in the very first
cell, in the radial direction.

In particular, this avoids problems in PWFA simulations when particles of the
driver can periodically pinch on the axis. In addition, the details of the
fields at the very tip of the bubble (where sheath electrons cross the axis)
are more realistic. As a result of the new treatment of particles, users may
notice that the charge density deposited on the grid, for a uniform
distribution of particles, appears to have a slight non-uniformity near the
axis. This is a known and expected effect, and can be reduced by increasing
the number of macro-particles in the radial direction (p_nr).

In addition to the above major change, a set of minor changes were introduced:
- JIT functions are now cached when running on CPU, which reduces the
  compilation time ([#451](https://github.com/fbpic/fbpic/pull/451) and
  [#445](https://github.com/fbpic/fbpic/pull/445))
- The new release fixes a bug that prevented the code to run on CPU, when
a GPU is available ([#454](https://github.com/fbpic/fbpic/pull/454)).

## 0.16.1

This is minor release of FBPIC, with essentially two improvements:
- It fixes a bug with GPUDirect MPI communications that was introduced in
version 0.16.1, when switching to a more extensive use of `cupy`.
(see [#440](https://github.com/fbpic/fbpic/pull/440))
- The compilation for multi-threaded CPU runs is now cached between different
simulations, thereby making to the first step of a simulation much faster.
(see [#445](https://github.com/fbpic/fbpic/pull/445))

## 0.16.0

This release uses `cupy` much more extensively in FBPIC, when running on GPU.
As a result, kernel launch overheads are drastically reduced, and small-size
or mid-size simulations will see a significant speed-up. Another consequence
is that FBPIC now requires Python 3 in order to run on GPU.

See [#437](https://github.com/fbpic/fbpic/pull/437) for more details.

## 0.15.1

This is a minor release, that makes improvements to the PML and to the documentation.

- The PML should now be slightly faster on GPU (due to the suppression of
unnecessary host-device communications: see [#423](https://github.com/fbpic/fbpic/pull/423))
- The particles do not gather the fields in the PML anymore: see [#424](https://github.com/fbpic/fbpic/pull/424).
- The documentation has a new section on 3D rendering: see [#426](https://github.com/fbpic/fbpic/pull/426).
- The documentation and code of FBPIC is now compatible with `openPMD-viewer 1.0.0`: see [#428](https://github.com/fbpic/fbpic/pull/428)
and [#427](https://github.com/fbpic/fbpic/pull/427).

## 0.15.0

This release adds the possibility to add Perfectly-Matched Layers in the
radial dimension (see [#417](https://github.com/fbpic/fbpic/pull/417)).
As part of this change, the user is now expected to pass a dictionary
for the argument `boundaries` of the `Simulation` object (instead of a
single string) ; for instance: `boundaries={'z':'open', 'r':'reflective'}`.
The PML in the radial direction are activated by passing `'r':'open'`.

In addition, several others changes have been included in this release:

- It is now possible to use FBPIC with numba 0.46 and cupy 7
(see [#414](https://github.com/fbpic/fbpic/pull/414)).
- The default phase of the `FlattenedGaussianLaser` has been updated.
(see [411](https://github.com/fbpic/fbpic/pull/411))
- A new laser profile has been added: `FewCycleLaser` which is well-adapted
for tightly-focused, short laser pulses (see [403](https://github.com/fbpic/fbpic/pull/403)).

## 0.14.0

This release fixes two important issues in FBPIC:

- There was a bug on GPU, related to particle sorting (see
[#402](https://github.com/fbpic/fbpic/pull/402) for more details). This bug
mostly affected simulations with ionization, and typically results in additional
ionization occurring in arbitrary locations in the simulation box
(See [#396](https://github.com/fbpic/fbpic/issues/396)). In addition, this
bug could have also affected simulations without ionization, and would result
in an incorrect particle push, for a fraction of the particles. It is now fixed.

- Particles that exit the simulation box radially used to still feel the
fields from the last cell of the box. This could cause these particles to
be pushed back into the box (see [#369](https://github.com/fbpic/fbpic/issues/369)).
This is now fixed: with the new version of the code, particles that exit the box
radially do not feel any force from the grid anymore, and move in a straight line
(see [#398](https://github.com/fbpic/fbpic/pull/398)).

## 0.13.3

This is a bug-fix release ; it prevents an incompatibility between cupy and the latest version of numba (numba 0.46).
In addition, a new diagnostic was added, in order to save the input script.

## 0.13.2

This is a bug-fix release ; it fixes a minor bug with Python 2, for the
back-transformed particle diagnostic (see [#389](https://github.com/fbpic/fbpic/pull/389)).

## 0.13.1

This is a minor release. It introduces:

- A minor API change: When creating a `Simulation` object with no particles,
the user does not need to do `sim.ptcl = []` anymore: [#376](https://github.com/fbpic/fbpic/pull/376)
- FBPIC is now compatible with the upcoming version 1.0 of `openPMD-viewer`: [#387](https://github.com/fbpic/fbpic/pull/387)
- A bug-fix for the ballistic injection in the lab-frame: [#384](https://github.com/fbpic/fbpic/pull/384)
- Limited, rudimentary support for PICMI: [#350](https://github.com/fbpic/fbpic/pull/350) [#383](https://github.com/fbpic/fbpic/pull/383) [#384](https://github.com/fbpic/fbpic/pull/384) [#384](https://github.com/fbpic/fbpic/pull/384)

## 0.13.0

This release introduces an important change on GPU: FBPIC now uses the `cupy` package instead of the `pyculib` package (since `pyculib` is no longer supported); see [#356](https://github.com/fbpic/fbpic/pull/356) [#363](https://github.com/fbpic/fbpic/pull/363) [#367](https://github.com/fbpic/fbpic/pull/367). As a result, it is now possible to use FBPIC with Numba 0.43 (or higher).

It also introduces official support for MPI-decomposed simulation ;
see [#348](https://github.com/fbpic/fbpic/pull/348)

In addition, this release adds various improvements to FBPIC:

**New features:**

- FBPIC can now initialize beams with arbitrary charge [#364](https://github.com/fbpic/fbpic/pull/364)
- The external fields can be automatically transformed to the boosted frame
[#342](https://github.com/fbpic/fbpic/pull/342)
- The `ParticleChargeDiagnostic` is now smoothed, in the same way as the regular charge density diagnostic [#349](https://github.com/fbpic/fbpic/pull/349)

**Optimizations:**    

- The number of threads per block were optimized for modern GPUs [#365](https://github.com/fbpic/fbpic/pull/365)
- Certain arrays are now kept on GPU, and never copied to CPU [#361](https://github.com/fbpic/fbpic/pull/361)

**Bug-fix:**

- There was a bug when restarting a simulation involving ionization on GPU [#360](https://github.com/fbpic/fbpic/pull/360)

## 0.12

This release introduces a few minor changes:

- It is now necessary to use Numba 0.42 (or lower), due to limitations in `pyculib`.
- The `BoostedFrameDiagnostics` are now renamed as `BackTransformedDiagnostics`; see [#339](https://github.com/fbpic/fbpic/pull/339)
- The user can now pass a physical time (instead of a number of iterations) to the `ParticleDiagnostic` and `FieldDiagnostic`; see [336](https://github.com/fbpic/fbpic/pull/336)

## 0.11

This release makes several improvements to FBPIC:

New features:
- It is now possible for the `ParticleDiagnostic` to output the gathered
E and B field (on the macroparticles), as well as the particles' Lorentz factor
to the openPMD files ; see the docstring of `ParticleDiagnostic`, and
[#330](https://github.com/fbpic/fbpic/pull/330),
[#316](https://github.com/fbpic/fbpic/pull/316).

Miscellaneous:
- It is now possible to use FBPIC with Numba 0.42 ;
see [#319](https://github.com/fbpic/fbpic/pull/319)
- The installation documentation now includes information for Summit, as
well as updated information for Lawrencium ; see
[#322](https://github.com/fbpic/fbpic/pull/322)

Bug-fixes:
- There was a bug in MPI communications on GPU (in the rare case where all
particles of a given species are located in the guard cells), that typically
resulted in a `CudaAPIError`. This is now fixed ; see
[##333](https://github.com/fbpic/fbpic/pull/333).
- Calling `add_gaussian_bunch` with erroneous parameters (e.g. such that
the bunch is not located inside the simulation box) used to produce `NaN`
in the simulation. This is now fixed ; see
[#331](https://github.com/fbpic/fbpic/pull/331).
- When using custom smoothers for the charge/current, although the Maxwell
solver did take into this custom smoother, the initial space-charge calculator
still used the default one. This is now fixed ; see
[#326](https://github.com/fbpic/fbpic/pull/326).
- The function `add_gaussian_bunch` could potentially generate electrons
with negative `gamma` (when initializing a bunch with a large energy spread).
This is now fixed ; see [#325](https://github.com/fbpic/fbpic/pull/325)

## 0.10.1

This is a bug-fix release.

It deals with the fact that FBPIC happens to be incompatible with Numba version 0.42.
FBPIC will now raise an error when Numba 0.42 is detected, and the user
will be prompted to install Numba 0.41.

In addition, the code can now perform particle sub-sampling in the openPMD diagnostic.

## 0.10.0

This release introduces various improvements to FBPIC:

New features:
- It is now possible to tune the amount of smoothing applied on the
charge and currents produced by the macroparticles; see [#268](https://github.com/fbpic/fbpic/pull/268).
- The documentation of the boosted-frame technique has been expanded. A
new function allows to easily compute the number of PIC iterations that
need to be performed in the boosted frame ; see
[#294](https://github.com/fbpic/fbpic/pull/294).
- The user can now set the name of the folder where the checkpoints are saved ; see
[#305](https://github.com/fbpic/fbpic/pull/305).
- The continuous injection of plasma (with moving window) is now more robust.
This can in particular suppress noise that some users might have observed in the
up-stream, continuously injected plasma, for very long simulations.

Bug-fixes:
- There was a bug in the laser antenna when using multiple CPUs/GPUs. This bug
has been fixed ; see [#309](https://github.com/fbpic/fbpic/pull/309).
- The `FlattenedGaussianLaser` had bugs, esp. when initializing it out of focus.
This is now fixed ; see [#305](https://github.com/fbpic/fbpic/pull/305) and
[ #299](https://github.com/fbpic/fbpic/pull/299).

## 0.9.5

This is bug-fix release.
It corrects a bug that was happening exclusively for *cubic shape* deposition,
when using the CPU (i.e. the bug does not occur on GPU) ; see [#297](https://github.com/fbpic/fbpic/pull/297)
In addition, this release adds a safe-guard for the sign of the charge, for Gaussian beams [#295](https://github.com/fbpic/fbpic/pull/295), and for the sign of the Galilean velocity [#293](https://github.com/fbpic/fbpic/pull/293).

## 0.9.4

This release introduces various improvements to FBPIC:

- The initial space-charge calculation produced unphysical fields for
high-energy beams with space-charge. This is now fixed ; see
[#289](https://github.com/fbpic/fbpic/pull/289)

- More flexible ionization API: The user can now store the electrons from
different ionization levels into different species. In addition, the user
can now set a maximum level of ionization ; see
[#288](https://github.com/fbpic/fbpic/pull/288) and
[#283](https://github.com/fbpic/fbpic/pull/283)

- Improved restart from checkpoints: for safer use, the restart now requires
that the number of species is the same in the simulation
(when calling `restart_from_checkpoint`) and in the checkpoint file.
(New species can nonetheless be added after calling `restart_from_checkpoint`.)
; see [#278](https://github.com/fbpic/fbpic/pull/278)

- The plasma can now be initialized with a non-zero temperature
(see the documentation of the method `add_new_species`) ; see
[#277](https://github.com/fbpic/fbpic/pull/277)

- There is a new diagnostic (`SpeciesChargeDensityDiagnostic`), which allows
to have the charge density of *one given* species ; see
[#287](https://github.com/fbpic/fbpic/pull/287)

Minor fixes:
- More efficient CPU execution by treading the `erase` function ;
see [#276](https://github.com/fbpic/fbpic/pull/276)
- When using `FBPIC_DISABLE_THREADING=1`, the code will not need to compile
for each execution, thereby allowing faster turnaround ; see
[#279](https://github.com/fbpic/fbpic/pull/279)



## 0.9.3

This is a minor release.

It fixes a few bugs:
- the restart mechanism for an arbitrary number of azimuthal modes ; see
[#275](https://github.com/fbpic/fbpic/pull/275). Before this, restarting
simulations worked only when using up to 2 azimuthal modes.
- the diagnostics of `rho` for multi-GPU simulations ; see [#265](https://github.com/fbpic/fbpic/pull/265)

In addition, this release introduces the possibility to inject a laser from
a moving antenna ; see [#262](https://github.com/fbpic/fbpic/pull/262).

It also introduces a minor change in the `LaserProfile` API:
the user should now pass the propagation direction (i.e. forward-propagating
or backward propagating) to the laser profile directly ; see [#260](https://github.com/fbpic/fbpic/pull/260).

## 0.9.2

This is a bug-fix release. It fixes a bug in the initial space-charge
calculation, that was introduced in version 0.9.1. (Previous versions do
not have this bug ; see [#254](https://github.com/fbpic/fbpic/pull/254) for
more details.)

In additional more laser profiles were added. This includes:
- Donut-like Laguerre-Gauss laser profile [#257](https://github.com/fbpic/fbpic/pull/257)
- Flattened Gaussian laser profile [#244](https://github.com/fbpic/fbpic/pull/244)

## 0.9.1

This is a minor release, which includes miscellaneous improvements to fbpic:

Bug fixes:
- The restart mechanism now works with ionizable particles. (See [#237](https://github.com/fbpic/fbpic/pull/237))
- For multi-CPU/GPU simulations using the Galilean scheme, the stencil extent and corresponding number of guard cells is now properly calculated. (See [#240](https://github.com/fbpic/fbpic/pull/241))

In addition, various improvements have been made to the documentation. (See [#239](https://github.com/fbpic/fbpic/pull/239) and [#240](https://github.com/fbpic/fbpic/pull/240).)

Finally, the charge/current deposition is now more efficient on CPU, when multiple species are used (See [#234](https://github.com/fbpic/fbpic/pull/234))

## 0.9.0

This version improves several features related to creation and continuous injection of particles.

New features:
- New species can now be created with the new method `add_new_species` of the
`Simulation` object. (See [228](https://github.com/fbpic/fbpic/pull/228))
This is particularly useful for simulations involving ionization.
(See [231](https://github.com/fbpic/fbpic/pull/231))
- Macroparticles with zero weight will not be created anymore. This saves time
and memory when running simulations where the plasma `dens_func` is 0 over
long distances ([223](https://github.com/fbpic/fbpic/pull/223)). In addition,
a warning is now printed if the `dens_func` returns negative values
([227](https://github.com/fbpic/fbpic/pull/227)).
- Electron bunches can now be injected through a plane, in order to avoid
space charge effects over long distances in the boosted frame
([186](https://github.com/fbpic/fbpic/pull/186)).
- MPI communications involve less synchronization
([#217](https://github.com/fbpic/fbpic/pull/217)) and include support for
GPUDirect ([226](https://github.com/fbpic/fbpic/pull/226)). Note that this is
for testing purposes for the moment. Multi-GPU/multi-CPU are still not
officially supported.
- The default number of guard cells and damp cells has been increased.
([229](https://github.com/fbpic/fbpic/pull/229))

Miscellaneous:
- The style of warnings has been changed ([227](https://github.com/fbpic/fbpic/pull/227)).
- This new release includes the cross-deposition scheme ([202](https://github.com/fbpic/fbpic/pull/202)).
This is for testing purposes for the moment.
- The arrays `z` and `r` of the interpolation grid are now calculated on the fly.
([215](https://github.com/fbpic/fbpic/pull/215))

Bug fixes:
- The restart from checkpoint were slightly incorrect, due to an incorrect
size of the grid, by one cell size ([224](https://github.com/fbpic/fbpic/pull/224)).
This is now fixed.
- Continuous injection (with moving window) used to be incorrect for particles
with different number of particles per cell in z. This is now fixed.
([230](https://github.com/fbpic/fbpic/pull/230))
- Injection of a Gaussian bunch in the boosted frame did not involve checking
that the particles are in the correct local subdomain. This is now fixed.
([232](https://github.com/fbpic/fbpic/pull/232))

## 0.8.0

This version allows FBPIC to run with an arbitrary number of azimuthal modes.
(The previous versions only worked with 2 modes: m=0 and m=1.)

This version also includes various improvements:

New features:
- Threading (on CPU) was only activated for numba 0.34 and numba 0.36. It
is now also activated for versions above 0.36.
(See [#199](https://github.com/fbpic/fbpic/pull/199))
- The code can now run in parallel with the `use_true_rho` option
(See [#210](https://github.com/fbpic/fbpic/pull/210))
- When using `verbose_level=2`, multi-CPU simulations now print the node
on which each MPI rank is running.
(See [#209](https://github.com/fbpic/fbpic/pull/209))

Bug fixes:
- The moving window used to only work in one direction. It can now work in both.
(See [#206](https://github.com/fbpic/fbpic/pull/206))
- In a multi-GPU run, the out-of-memory errors will stop the whole simulation,
instead of having other GPUs waiting indefinitely for the one GPU that crashed.
(See [#212](https://github.com/fbpic/fbpic/pull/212))
- Version 0.7.1 introduced a minor bug in the GPU current deposition, which
lead to slight differences between the CPU and GPU results.
This has now been fixed. (See [#216](https://github.com/fbpic/fbpic/pull/216))
- There was a minor bug in the restart code, which did not update the arrays
of positions of the grid. This was only likely to affect restarts followed
by the initialization of a laser (a rare case). This now fixed by calculating
these arrays on the fly. (See [#215](https://github.com/fbpic/fbpic/pull/215))
- The field and particle diagnostics will now automatically convert the input
`period` to an integer, which avoids erratic behavior if a float was provided.
(See [#198](https://github.com/fbpic/fbpic/pull/198))

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
