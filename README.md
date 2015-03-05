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

For more details on the algorithm, see the `docs` folder.

Since this is only a proof of principle, the implementation has
important shortcomings :

* Unoptimized code written entirely in Python (using Numpy for matrix
operations)
* Single-processor only   

Installation
---------

Standard installation with `setup.py` (i.e. `python setup.py install`).