# Fourier-Bessel Particle-In-Cell code (FBPIC)

[![Build Status main](https://img.shields.io/travis/fbpic/fbpic/main.svg?label=main)](https://travis-ci.com/fbpic/fbpic/branches)
[![Build Status dev](https://img.shields.io/travis/fbpic/fbpic/dev.svg?label=dev)](https://travis-ci.com/fbpic/fbpic/branches)
[![pypi version](https://img.shields.io/pypi/v/fbpic.svg)](https://pypi.python.org/pypi/fbpic)
[![License](https://img.shields.io/pypi/l/fbpic.svg)](LICENSE.txt)
[![DOI](https://zenodo.org/badge/69215997.svg)](https://zenodo.org/badge/latestdoi/69215997)

Online documentation: [http://fbpic.github.io](http://fbpic.github.io)<br/>
Support: [Join slack](https://slack-fbpic.herokuapp.com)

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

### Language and hardware

FBPIC is written entirely in Python, but uses
[Numba](http://numba.pydata.org/) Just-In-Time compiler for high
performance. In addition, the code can run on **CPU** (with multi-threading)
and on **GPU**. For large simulations, running the
code on GPU can be much faster than on CPU.

### Advanced features of laser-plasma acceleration

FBPIC implements several useful features for laser-plasma acceleration, including:
- Moving window
- Cylindrical geometry (with azimuthal mode decomposition)
- Calculation of space-charge fields at the beginning of the simulation
- Intrinsic mitigation of Numerical Cherenkov Radiation (NCR) from relativistic bunches
- Field ionization module (ADK model)

In addition, FBPIC supports the **boosted-frame** technique (which can
dramatically speed up simulations), and includes:
- Utilities to convert input parameters from the lab frame to the boosted frame
- On-the-fly conversion of simulation results from the boosted frame back to the lab frame
- Suppression of the Numerical Cherenkov Instability (NCI) using the Galilean technique

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
conda install numba scipy h5py mkl
conda install -c conda-forge mpi4py
```
- Download and install FBPIC:
```
pip install fbpic
```

- **Optional:** in order to run on GPU, install the additional package
`cudatoolkit` and `cupy` -- e.g. using CUDA version 10.0.
```
conda install cudatoolkit=10.0
pip install cupy-cuda100
```
(In the above command, you should choose a CUDA version that is compatible with your GPU driver ; see [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#major-components__table-cuda-toolkit-driver-versions) for more info.)

- **Optional:** in order to run on a CPU which is **not** an Intel model, you
need to install `pyfftw`, in order to replace the MKL FFT:
```
conda install -c conda-forge pyfftw
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

We welcome contributions to the code! Please read [this page](https://github.com/fbpic/fbpic/blob/main/CONTRIBUTING.md) for guidelines on how to contribute.

## Research & Attribution

FBPIC was originally developed by Remi Lehe at [Berkeley Lab](http://www.lbl.gov/),
and Manuel Kirchen at
[CFEL, Hamburg University](http://lux.cfel.de/). The code also
benefitted from the contributions of Soeren Jalas (CFEL), Kevin Peters (CFEL),
Irene Dornmair (CFEL), Laurids Jeppe (CFEL), Igor Andriyash (Laboratoire d’Optique Appliquée),
Omri Seemann (Weizmann Institute), Daniel Seipt (University of Michigan)
and Samuel Yoffe (University of Strathclyde).

FBPIC's algorithms are documented in following scientific publications:

* General description of FBPIC and it's algorithm (original paper): [R. Lehe et al., CPC, 2016](http://www.sciencedirect.com/science/article/pii/S0010465516300224) ([arXiv](https://arxiv.org/abs/1507.04790))
* Boosted-frame technique with Galilean algorithm: [M. Kirchen et al., PoP, 2016](https://aip.scitation.org/doi/10.1063/1.4964770) ([arXiv](https://arxiv.org/abs/1608.00215)) and [Lehe et al., PRE, 2016](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.94.053305) ([arXiv](https://arxiv.org/abs/1608.00227))
* Parallel finite-order solver for multi-CPU/GPU simulations: [S. Jalas et al., PoP, 2017](https://aip.scitation.org/doi/abs/10.1063/1.4978569) ([arXiv](https://arxiv.org/abs/1611.05712))
* Parallel finite-order boosted-frame simulations for multi-CPU/GPU simulations: [M. Kirchen et al., PRE, 2020](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.102.013202)

If you use FBPIC for your research project: that's great! We are
very pleased that the code is useful to you!

If your project even leads to a scientific publication, please consider citing at least FBPIC's original paper. If your project uses the more advanced algorithms, please consider citing the respective publications in addition.
