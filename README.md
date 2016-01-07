Fourier-Bessel Particle-In-Cell code (FBPIC)
=============================

Overview
--------

This program is a proof-of-principle Particle-In-Cell (PIC) code,
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

Since this is only a proof of principle, the implementation has
important shortcomings :

* Sub-optimized code written in Python.   
(However, the code does call BLAS and FFTW for computationally intensive parts. 
If available, it will also use Numba.)
* Single-CPU (with partial use of multi-threading) or single-GPU only  

Installation
---------

The recommended installation is through the
[Anaconda](https://www.continuum.io/why-anaconda) distribution:

- If Anaconda is not your default Python installation, download and install it from
  [here](https://www.continuum.io/downloads).
- `cd` into the top folder of `fbpic` and install the dependencies:  
```
conda install --file requirements.txt
```
- Install `pyfftw` (not in the standard Anaconda channels, and thus it
requires a special command):  
```
conda install -c https://conda.anaconda.org/mforbes pyfftw
conda upgrade numpy
```
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

Running simulations
----------------

Simulations are run with a user-written python script, which calls the
FBPIC structures. An example script (called `lpa_sim.py`) can be found in
`docs/example_input`. The simulation can be run simply by entering
`python -i lpa_sim.py`.

The code outputs HDF5 files, that comply with the
[OpenPMD standard](http://www.openpmd.org/#/start),
 and which can thus be read as such (e.g. by using the [openPMD-viewer](https://github.com/openPMD/openPMD-viewer)).
