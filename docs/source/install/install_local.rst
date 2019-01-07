Installation on a local computer
==================================

Installing FBPIC
------------------

The installation requires the
`Anaconda <https://www.continuum.io/why-anaconda>`__ distribution of
Python. If Anaconda is not your default Python distribution, download and install it from `here <https://www.continuum.io/downloads>`__.

**Installation steps**:

- Install the dependencies of FBPIC. This can be done in two lines:

  ::

     conda install numba scipy h5py mkl
     conda install -c conda-forge mpi4py

-  Install ``fbpic``

   ::

       pip install fbpic

   Alternatively, instead of using ``pip``, you can also install FBPIC
   from the souces, by cloning the `code from Github
   <https://github.com/fbpic/fbpic>`_, and typing ``python setup.py
   install``.


-  **Optional:** In order to be able to run the code on a GPU:

   ::

       conda install cudatoolkit=8.0 pyculib

   (If your GPU driver supports a newer version of CUDA than 8.0, you can
   modify the above number accordingly.)

-  **Optional:** In order to run on a CPU which is **not** an Intel model, you need to install `pyfftw`, in order to replace the MKL FFT:

   ::

      conda install -c conda-forge pyfftw


Potential issues
--------------------------------

On Mac OSX, the package ``mpi4py`` can sometimes cause
issues. If you observe that the code crashes with an
MPI-related error, try installing ``mpi4py`` using MacPorts and
``pip``. To do so, first install `MacPorts <https://www.macports.org/>`_. Then execute the following commands:

::

   conda uninstall mpi4py
   sudo port install openmpi-gcc48
   sudo port select --set mpi openmpi-gcc48-fortran
   pip install mpi4py

Running simulations
-------------------

See the section :doc:`../how_to_run`, for instructions on how to run a simulation.
