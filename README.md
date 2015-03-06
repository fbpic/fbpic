Fourier-Bessel Particle-In-Cell code (FBPIC)
=============================

Overview
--------

This program is a proof-of-principle Particle-In-Cell (PIC) code,
whose distinctive feature is to use a **spectral decomposition in
cylindrical geometry** for the fields (Fourier-Bessel
decomposition). This decomposition allows to combine the advantages of
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

For more details on the algorithm, see the `docs/article` folder.

Since this is only a proof of principle, the implementation has
important shortcomings :

* Unoptimized code written entirely in Python. (However, the code does
  call BLAS and FFTW for computationally intensive parts. If
  available, it will also use Numba)
* Single-processor only

Installation
---------

Standard installation with `setup.py` (i.e. `python setup.py install`).

Numba is not required for the code to run, since it is somewhat
difficult to install on MacOS. However, if it is installed, the code
will automatically detect it and use it. This can result in up to an order
of magnitude speedup.

Instructions for installing Numba on MacOS can be found in `docs/install`
