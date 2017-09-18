# Fourier-Bessel Particle-In-Cell code (FBPIC)

[![Build Status master](https://img.shields.io/travis/fbpic/fbpic/master.svg?label=master)](https://travis-ci.org/fbpic/fbpic/branches)
[![Build Status dev](https://img.shields.io/travis/fbpic/fbpic/dev.svg?label=dev)](https://travis-ci.org/fbpic/fbpic/branches)
[![pypi version](https://img.shields.io/pypi/v/fbpic.svg)](https://pypi.python.org/pypi/fbpic)
[![License](https://img.shields.io/pypi/l/fbpic.svg)](LICENSE.txt)
[![DOI](https://zenodo.org/badge/69215997.svg)](https://zenodo.org/badge/latestdoi/69215997)

## Overview

FBPIC is a
[Particle-In-Cell (PIC) code](https://en.wikipedia.org/wiki/Particle-in-cell)
for relativistic plasma physics.  

It is especially well-suited for physical simulations of
**laser-wakefield acceleration** and **plasma-wakefield acceleration**, with close-to-cylindrical symmetry.

### Algorithm

The distinctive feature of FBPIC is to use
a **spectral decomposition in
cylindrical geometry** (Fourier-Bessel
decomposition) for the fields. This combines the advantages of **spectral 3D** PIC codes (high accuracy and stability) and
those of **finite-difference cylindrical** PIC codes
(orders-of-magnitude speedup when compared to 3D simulations).  
For more details on the algorithm, its advantages and limitations, see
the [documentation](http://fbpic.github.io).

### Language and harware

FBPIC is written entirely in Python, but uses
[Numba](http://numba.pydata.org/) Just-In-Time compiler for high
performance. In addition, the code can run on **CPU** (with multi-threading) 
and on **GPU**. For large simulations, running the 
code on GPU can be much faster than on CPU.

## Installation

The installation instructions below are for a local computer. For more
details, or for instructions specific to a particular HPC cluster, see
the [documentation](http://fbpic.github.io).

The recommended installation is through the
[Anaconda](https://www.continuum.io/why-anaconda) distribution.
If Anaconda is not your default Python installation, download and install
it from [here](https://www.continuum.io/downloads).

**Installation steps**:

- Install the dependencies of FBPIC. This can be done in two lines:
```
conda install numba scipy h5py
conda install -c conda-forge mpi4py pyfftw
```
- Download and install FBPIC:
```
pip install fbpic
```

- **Optional:** in order to run on GPU, install the additional package
`pyculib`:
```
conda install -c numba pyculib
```

## Running simulations

Once installed, FBPIC is available as a **Python module** on your
system.

Therefore, in order to run a physical simulation, you will need a **Python
script** that imports FBPIC's functionalities and use them to setup the
simulation. You can find examples of such scripts in the
[documentation](http://fbpic.github.io) or in this repository, in `docs/source/example_input/`.

Once your script is ready, the simulation is run simply by typing:
```
python fbpic_script.py
```
The code outputs HDF5 files, that comply with the
[OpenPMD standard](http://www.openpmd.org/#/start),
 and which can thus be read as such (e.g. by using the
 [openPMD-viewer](https://github.com/openPMD/openPMD-viewer)).

## Contributing

We welcome contributions to the code! Please read [this page](https://github.com/fbpic/fbpic/blob/master/CONTRIBUTING.md) for guidelines on how to contribute.

## Attribution

FBPIC was originally developed by Remi Lehe at [Berkeley Lab](http://www.lbl.gov/),
and Manuel Kirchen at
[CFEL, Hamburg University](http://lux.cfel.de/). The code also
benefitted from the contributions of Soeren Jalas, Kevin Peters and
Irene Dornmair (CFEL).

If you use FBPIC for your research project: that's great! We are
very pleased that the code is useful to you!

If your project even leads to a scientific publication, please
consider citing FBPIC's original paper, which can be found
[here](http://www.sciencedirect.com/science/article/pii/S0010465516300224)
(see [this link](https://arxiv.org/abs/1507.04790) for the arxiv version).
