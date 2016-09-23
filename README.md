# Fourier-Bessel Particle-In-Cell code (FBPIC)

## Overview

FBPIC is a **[Particle-In-Cell (PIC) code](https://en.wikipedia.org/wiki/Particle-in-cell)** for plasma physics.  

It is especially well-suited for physical simulations of
**laser-wakefield acceleration** and **plasma-wakefield acceleration**, with close-to-cylindrical symmetry.

**Algorithm:**  
The distinctive feature of FBPIC is to use
a **spectral decomposition in
cylindrical geometry** (Fourier-Bessel
decomposition) for the fields. This combines the advantages of **spectral 3D Cartesian** PIC codes (high accuracy and stability) and
those of **finite-difference cylindrical** PIC codes
(orders-of-magnitude speedup when compared to 3D simulations).  
For more details on the algorithm, its advantages and limitations, see
the [documentation](http://fbpic.github.io).


**Language and harware:**  
FBPIC is written entirely in Python, but uses 
**[Numba](http://numba.pydata.org/)** Just-In-Time compiler for high
performance. In addition, the code was designed to be run
either on **CPU or GPU**. For large
simulations, running the code on GPU can be up to **40 times faster**
than on CPU.

## Installation

The installation instructions below are for a local computer. For more
details, or for instructions specific to a particular HPC cluster, see
the [documentation](http://fbpic.github.io).

The recommended installation is through the
[Anaconda](https://www.continuum.io/why-anaconda) distribution.
If Anaconda is not your default Python installation, download and install
it from [here](https://www.continuum.io/downloads).

**Installation steps**:

- Install the dependencies of FBPIC. This can be done with a single line:  
```
conda install -c conda-forge numba matplotlib scipy h5py mpi4py pyfftw
```
- Download and install FBPIC:
```
pip install fbpic
```

- **Optional:** in order to run on GPU, install the additional package
`accelerate`:
```
conda install accelerate
```
(The `accelerate` package is not free, but there is a 30-day free trial period,
  which starts when the above command is entered. For further use beyond 30
  days, one option is to obtain an academic license, which is also free. To do
  so, please visit [this link](https://www.continuum.io/anaconda-academic-subscriptions-available).)

## Running simulations

Simulations are run with a user-written python script, which calls the
FBPIC structures. An example script (called `lpa_sim.py`) can be found in
`docs/example_input`. The simulation can be run simply by entering
`python -i lpa_sim.py`.

The code outputs HDF5 files, that comply with the
[OpenPMD standard](http://www.openpmd.org/#/start),
 and which can thus be read as such (e.g. by using the [openPMD-viewer](https://github.com/openPMD/openPMD-viewer)).
