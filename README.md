# Fourier-Bessel Particle-In-Cell code (FBPIC)

## Overview

This program is a Particle-In-Cell (PIC) code,
whose distinctive feature is to use a **spectral decomposition in
cylindrical geometry** for the fields (Fourier-Bessel
decomposition).

This decomposition allows to combine the advantages of
**spectral 3D Cartesian** PIC codes (high accuracy and stability) and
those of **finite-difference cylindrical** PIC codes with azimuthal
decomposition (orders-of-magnitude speedup when compared to 3D simulations).
Here are some of the specific features of this code :  

* The Maxwell solver uses a Pseudo-Spectral Analytical Time-Domain
  algorithm (PSATD), and is therefore **dispersion-free in all
  directions**, in vacuum.
* The fields *E* and *B* are defined at the **same points in space** and at
  the **same time** (i.e. they are not staggered). This avoids some
  interpolation errors.
* The particle pusher uses the Vay algorithm.
* The moving window is supported.
* The initialization of charged bunch (with its space-charge fields)
  is supported.

For more details on the algorithm, see the `docs/article` folder.

Implementation details:

* The code is written in Python and calls BLAS and FFTW for computationally
intensive parts. Moroever, it will also use Numba, if availabe
* Single-CPU (with partial use of multi-threading) or single-GPU

## Installation

The installation instructions below are for a local computer. For instructions
specific to a particular HPC cluster (e.g. Titan, Jureca, Lawrencium, etc.), please
see `docs/install`.

The recommended installation is through the
[Anaconda](https://www.continuum.io/why-anaconda) distribution:

- If Anaconda is not your default Python installation, download and install
it from [here](https://www.continuum.io/downloads).

- Clone the `fbpic` repository using git.

- `cd` into the top folder of `fbpic` and install the dependencies:  
```
conda install -c conda-forge --file requirements.txt
```
- **Optional:** In order to be able to run the code on a GPU:
```
conda install accelerate
conda install accelerate_cudalib
```
(The `accelerate` package is not free, but there is a 30-day free trial period,
  which starts when the above command is entered. For further use beyond 30
  days, one option is to obtain an academic license, which is also free. To do
  so, please visit [this link](https://www.continuum.io/anaconda-academic-subscriptions-available).)

- Install `fbpic`  
```
python setup.py install
```

The installation can be tested by running:
```
python setup.py test
```

If you encounter issues with the installation, you may find a
documented solution in `docs/install`.

## Running simulations

Simulations are run with a user-written python script, which calls the
FBPIC structures. An example script (called `lpa_sim.py`) can be found in
`docs/example_input`. The simulation can be run simply by entering
`python -i lpa_sim.py`.

The code outputs HDF5 files, that comply with the
[OpenPMD standard](http://www.openpmd.org/#/start),
 and which can thus be read as such (e.g. by using the [openPMD-viewer](https://github.com/openPMD/openPMD-viewer)).
