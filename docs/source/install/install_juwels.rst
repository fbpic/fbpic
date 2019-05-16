Installation on JUWELS (JSC)
=================================================

`JUWELS
<https://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUWELS/JUWELS_node.html>`__
is a supercomputer at the `Juelich Supercomputing Centre <http://www.fz-juelich.de/ias/jsc/EN/Home/home_node.html>`__ (JSC).

Installation and usage of FBPIC requires the following steps:

-  Loading the cluster modules
-  Installation of Anaconda
-  Installation of FBPIC
-  Allocation of resources and running simulations

Loading the cluster modules
---------------------------

On the JUWELS cluster to run on CPU and GPU with MPI, the standard MPI
implementation can be loaded. To get potentially higher performance with
GPUDIRECT, follow the alternative steps below.

To load the standard modules the ``.bashrc`` should contain the following:

::

    module load CUDA
    module load GCC
    module load ParaStationMPI


If you want to use a fast CUDA-aware MPI library, a different module
needs to be loaded. The ``.bashrc`` should contain the following:

::

    module load CUDA
    module load GCC
    module load MVAPICH2

Warning: This MPI library will currently only work on GPU nodes!!
(Please also note that GPUDIRECT is currently not available on the cluster,
but should become available at some point - the unavailability is only a
software issue)

Installation of Anaconda
------------------------------------------------

In order to download and install `Anaconda <https://www.continuum.io/downloads>`__, type:

::

    wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
    bash Anaconda3-2019.03-Linux-x86_64.sh

Then install the dependencies of FBPIC:

::

   conda install numba==0.42 scipy h5py mkl
   conda install cudatoolkit=9.0 pyculib

It is advised that the following packages are **NOT** installed
directly with Anaconda: ``mpich`` and ``mpi4py``

You can install mpi4py directly with pip and it will be built against the MPI
library that is loaded on the cluster via the modules (so either the standard
ParaStationMPI or MVAPICH2 for GPUs).

::

   pip install mpi4py

You can check if the correct MPI is linked by opening a ``python`` shell
and checking:

::

    from mpi4py import MPI
    MPI.Get_library_version()

Please note that this might not work on the login node if mpi4py is built
against MVAPICH2. You need to allocate a GPU node (see below) and then open
an interactive python shell on the GPU node via ``srun --pty python``. Then
you can import the mpi4py MPI module as shown above and check that the library
version is correct.


Note that sometimes it is also required that you add the Anaconda folders to
your ``PATH`` and ``PYTHONPATH`` environment variables after the
``module loads ...`` in your ``.bashrc``. For example:

::

    export PATH="/p/home/jusers/USERNAME/juwels/anaconda3/bin":$PATH
    export PYTHONPATH="/p/home/jusers/USERNAME/juwels/anaconda3/lib/python3.7/site-packages":$PYTHONPATH

Installation of FBPIC
---------------------

Finally, clone FBPIC using ``git``, ``cd`` into the folder ``fbpic/``
and type
::

   python setup.py install

Running simulations
------------------------------------------

In the following, it is explained how to allocate and use
**interactive** jobs on JUWELS. For the usage of normal jobs, one can
use the similar commands in a job script. More information can be found
here:

``https://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUWELS/UserInfo/UserInfo_node.html``

**Allocation of ressources**

**CPU:** CPU nodes consist of 2x24 cores. Allocation of two nodes for 60
minutes:

``salloc --nodes=2 --time=00:60:00``

**CPU multithreading:** Best performance is expected if the FBPIC
(and NUMBA/MKL) threading settings are set to 24 threads, while using one MPI
process per socket. As a single JUWELS node has two sockets with each 24 cores,
this means that ideally 2 MPI processes should be used when running on a
single node.

**GPU:** GPU nodes consist of 4 Nvidia V100 Devices, i.e. 4 GPUs.
Allocation of 8 GPUs (2 nodes) for 60 minutes:

``salloc --nodes=2 --partition=gpus --time=00:60:00 --gres=gpu:4``

**Starting an interactive run**

The following command starts an interactive run (run_file.py) with 8
tasks (e.g. 8 GPUs). ``--pty`` activates continuous console output and
``--forward-x``\ enables X-forwarding if the connection to JURECA was
established with ``ssh -Y username@juwels.fz-juelich.de``.

``srun --ntasks=8 --forward-x --pty python run_file.py``
