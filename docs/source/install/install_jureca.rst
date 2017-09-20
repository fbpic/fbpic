Installation on JURECA (JSC)
=================================================

`JURECA
<http://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JURECA/JURECA_node.html>`__
is a supercomputer at the `Juelich Supercomputing Centre <http://www.fz-juelich.de/ias/jsc/EN/Home/home_node.html>`__ (JSC).

Installation and usage of FBPIC requires the following steps:

-  Loading the cluster modules
-  Installation of Anaconda
-  Installation of FBPIC
-  Allocation of resources and running simulations

Loading the cluster modules
---------------------------

On the JURECA cluster, the correct modules to use a fast CUDA-aware MPI
distribution need to be loaded.

Therefore, the ``.bashrc`` should contain the following:

::

    module load Intel
    module load ParaStationMPI
    module load CUDA
    module load HDF5
    module load mpi4py/2.0.0-Python-2.7.12
    module load h5py/2.6.0-Python-2.7.12

Please note that the exact versions may change in the future.

Installation of Anaconda
------------------------------------------------

In order to download and install `Anaconda <https://www.continuum.io/downloads>`__, type:

::

    wget https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh
    bash Anaconda2-4.2.0-Linux-x86_64.sh

Then install the dependencies of FBPIC:
::

   conda install numba=0.34
   conda install -c numba pyculib
   conda install -c conda-forge pyfftw

It is important that the following packages are **NOT** installed
directly with Anaconda: ``mpich``, ``mpi4py``, ``hdf5`` and ``h5py``

One can check if the correct MPI is linked by opening a ``python`` shell
and checking:

::

    from mpi4py import MPI
    MPI.Get_library_version()

Note that the ``PATH`` and ``PYTHONPATH`` environment variables have to be set
after the ``module loads ...`` in your ``.bashrc`` to work with the conda environment.

Installation of FBPIC
---------------------

Finally, clone FBPIC using ``git``, ``cd`` into the folder ``fbpic/``
and type
::

   python setup.py install

Running simulations
------------------------------------------

In the following, it is explained how to allocate and use
**interactive** jobs on JURECA. For the usage of normal jobs, one can
use the similar commands in a job script. More information can be found
here:

``http://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JURECA/UserInfo/UserInfo_node.html``

**Allocation of ressources**

**CPU:** CPU nodes consist of 24 cores. Allocation of two nodes for 60
minutes:

``salloc --nodes=2 --time=00:60:00``

**GPU:** GPU nodes consist of 2 Nvidia K80 Devices, i.e. 4 GPUs.
Allocation of 4 GPUs (2 nodes) for 60 minutes:

``salloc --nodes=2 --partition=gpus --time=00:60:00 --gres=gpu:4``

**Starting an interactive run**

The following command starts an interactive run (run\_file.py) with 8
tasks (e.g. 8 GPUs). ``--pty`` activates continuous console output and
``--forward-x``\ enables X-forwarding if the connection to JURECA was
established with ``ssh -Y username@jureca.fz-juelich.de``.

``srun --ntasks=8 --forward-x --pty python run_file.py``
