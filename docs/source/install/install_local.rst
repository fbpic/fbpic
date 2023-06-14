Installation on a local computer
==================================

Installing FBPIC
------------------

The installation requires the
`Anaconda <https://docs.anaconda.com/anaconda/>`__ distribution of
Python. If Anaconda is not your default Python distribution, download and install it from `here <https://docs.anaconda.com/anaconda/install/>`__.

**Installation steps**:

- Install the dependencies of FBPIC:

  ::

     conda install -c conda-forge numba scipy h5py mkl mpi4py

-  Install ``fbpic``

   ::

       pip install fbpic

   .. note::

       If you want to run FBPIC through the
       `PICMI interface <https://picmi-standard.github.io/>`__, you can instead
       use

       ::

           pip install fbpic[picmi]

   .. note::
       Instead of using a release, you can also install FBPIC from the sources,
       by cloning the `code from Github <https://github.com/fbpic/fbpic>`_,
       and executing ``python3 -m pip install .`` from the main directory.
       A shortcut for this is: ``python3 -m pip install git+https://github.com/fbpic/fbpic.git``.

-  **Optional:** In order to be able to run the code on a GPU,
   install the additional package ``cudatoolkit`` and ``cupy`` --
   e.g. using CUDA version 11.0:

   ::


       conda install -c conda-forge cudatoolkit=11.0
       pip install cupy-cuda110

   .. warning::

       In the above command, you should choose a CUDA version that is **compatible
       with your GPU driver**. You can see the version of your GPU driver by typing
       the command ``nvidia-smi``. You can then find the compatible CUDA
       versions using `this table <https://docs.nvidia.com/deploy/cuda-compatibility/index.html#use-the-right-compat-package>`__.

-  **Optional:** In order to run on a CPU which is **not** an Intel model, you need to install `pyfftw`, in order to replace the MKL FFT:

   ::

      conda install -c conda-forge pyfftw


Potential issues
----------------

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
