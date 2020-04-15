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

To load the standard modules the ``.bashrc`` should contain the following:

::

    module load GCC/8.3.0
    module load MVAPICH2/2.3.3-GDR

Installation of Anaconda
------------------------------------------------

In order to download and install `Anaconda <https://www.continuum.io/downloads>`__, type:

::

    wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
    bash Anaconda3-2019.03-Linux-x86_64.sh

Then install the dependencies of FBPIC:

::

    conda install numba scipy h5py mkl
    conda install cudatoolkit=10.0
    pip install cupy-cuda100

It is advised that the following packages are **NOT** installed
directly with Anaconda: ``mpich`` and ``mpi4py``

You can install mpi4py directly with pip and it will be built against the MPI
library that is loaded on the cluster via the modules.

::

    pip install mpi4py --no-cache-dir

If you want to use GPUdirect, you instead need the development version:

::

    pip install git+https://bitbucket.org/mpi4py/mpi4py.git --no-cache-dir

You can check if the correct MPI is linked by opening a ``python`` shell
and checking:

::

    from mpi4py import MPI
    MPI.Get_library_version()

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
`here <https://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUWELS/UserInfo/UserInfo_node.html>`__.

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

**Using GPUdirect**

To take advantage of direct communication between MPI and the GPUs, run:

``export FBPIC_ENABLE_GPUDIRECT=1``

Note that this will only work with the correct MPI implementation (i.e. the
cluster modules above) and requires the development version of ``mpi4py``.
