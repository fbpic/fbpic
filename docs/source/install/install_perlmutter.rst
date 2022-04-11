Installation on Perlmutter (NERSC)
==================================

`Perlmutter
<https://docs.nersc.gov/systems/perlmutter/>`__
is a high-performance cluster at `NERSC
<http://www.nersc.gov/>`__.

Its compute nodes are equipped with 4 A100 NVIDIA GPUs.

Installation of FBPIC
---------------------

Setting up Anaconda
~~~~~~~~~~~~~~~~~~~

- Type the following lines to prepare your environment.

    ::

        module load PrgEnv-gnu cpe-cuda cudatoolkit python

- Create a new conda environment and activate it.

    ::

        conda create -n fbpic
        source activate fbpic

Installation of FBPIC and its dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Install the dependencies of ``fbpic``

   ::

       conda install -c conda-forge python numba scipy h5py mkl cudatoolkit=11.5
       pip install cupy-cuda115
       MPICC="cc -shared -target-accel=nvidia80" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py

-  Install ``fbpic``

   ::

       pip install fbpic

Running simulations
-------------------

Interactive jobs
~~~~~~~~~~~~~~~~

In order to request a node with a GPU:

::

    salloc -t 00:30:00 -N 1 -C gpu --ntasks-per-node=4 --gpus-per-task=1 -A <account_number>

Then ``cd`` to the directory where you prepared your input script and type

::

    module load python cudatoolkit
    source activate fbpic
    python <fbpic_script.py>

Batch job
~~~~~~~~~

Create a new file named ``submission_file`` in the same directory as
your input script (typically this directory is a subdirectory of
``/global/scratch/<yourUsername>``). Within this new file, copy the
following text (and replace the bracketed text by the proper values).

::

    #!/bin/bash
    #SBATCH -J my_job
    #SBATCH -A <account_number>
    #SBATCH -C gpu
    #SBATCH --time <requestedTime>
    #SBATCH --ntasks <requestedRanks>
    #SBATCH --ntasks-per-node=4
    #SBATCH --gpus-per-task=1

    module load python cudatoolkit
    source activate fbpic

    export MPICH_GPU_SUPPORT_ENABLED=0
    export FBPIC_ENABLE_GPUDIRECT=0

    srun -n <requestedRanks> python fbpic_script.py

Then run:

::

    sbatch submission_file
