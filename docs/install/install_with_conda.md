Installation of FBPIC  on the Anaconda Python distribution
=======================================

Overview
-------

Anaconda is a convenient distribution of Python. However, it does not
have the packages `pyfftw` and moreover the package `mpi4py` is buggy.

This file describes how to install `pyfftw` in `mpi4py` in this case.

Installation of `pyfftw`
-------------------

For OSX :
`conda install -c https://conda.binstar.org/asmeurer pyfftw`

For Linux :
`conda install -c https://conda.binstar.org/richli pyfftw`

(See
[this link](https://groups.google.com/a/continuum.io/forum/#!topic/anaconda/_nDXdAYweCI)
for reference)

Bug fix  of `mpi4py` on OSX
------------------------

If the following command crashes
`>>> from mpi4py import MPI`
it can be fixed by typing
`sudo ln -s ~/anaconda /opt/anaconda1anaconda2anaconda3`
