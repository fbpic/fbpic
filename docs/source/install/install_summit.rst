Installation on Summit (OLCF)
=============================

`Summit <https://www.olcf.ornl.gov/olcf-resources/compute-systems/summit/>`__
is a GPU cluster at the `Oakridge Leadership Computing Facility
<https://www.olcf.ornl.gov/>`__ (OLCF).

Each node has of 6 Nvidia V100 GPUs.

Installation of FBPIC
---------------------

Preparing the Anaconda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First create a new `conda` environment

   ::

        module load python/3.7.0-anaconda3-5.3.0
        conda create -n fbpic python=3

Then add the following lines in your `.bashrc`

   ::

        module purge
        module load gcc/4.8.5
        module load spectrum-mpi/10.2.0.10-20181214
        module load python/3.7.0-anaconda3-5.3.0
        module load py-mpi4py/3.0.0-py3
        source activate fbpic

Then type

    ::

        . .bashrc

Installation of FBPIC and its dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Install the dependencies of ``fbpic``

   ::

        conda install -c conda-forge numba scipy h5py cudatoolkit=8.0

- Install ``pyfftw``

    ::

        module load fftw
        pip install pyfftw

- Install ``pyculib``

   ::

        git clone https://github.com/dzhoshkun/pyculib_sorting.git
        cd pyculib_sorting
        git submodule update --init
        module load cuda
        python build_sorting_libs.py
        module unload cuda
        cp lib/*.so ~/.conda/envs/fbpic/lib/
        pip install pyculib
        cd ..

-  Install ``fbpic``

   ::

        pip install fbpic

Running simulations
-------------------

Interactive jobs
~~~~~~~~~~~~~~~~

In order to request one node for 20 minutes:

::

    bsub -W 00:20 -nnodes 1 -P <account_number> -Is /bin/bash

Then ``cd`` to the directory where you prepared your input script and type

::

    jsrun -n 1 -a 1 -c 1 -g 1 python <fbpic_script.py>

Batch job
~~~~~~~~~

Create a new file named ``submission_file`` in the same directory as
your input script. Within this new file, copy the
following text (and replace the bracketed text by the proper values).

::

    #!/bin/bash
    #SBATCH -J my_job
    #SBATCH --nodes <requestedNode>
    #SBATCH --time <requestedTime>
    #SBATCH --export=ALL
    #SBATCH -p gpu
    #SBATCH --gres=gpu:<gpuType>:4
    #SBATCH --ntasks-per-node=<coresPerGPU>

    srun --mpi=pmi2 -n <nMPI> python fbpic_script.py

where ``<nMPI>`` should be 4 times ``<requestedNode>``
(since there are 4 GPUs per node on Comet), and where:

    - For a K80 node: ``<gpuType>`` should be ``k80`` and ``coresPerGPU`` should be ``6``
    - For a P100 node: ``<gpuType>`` should be ``p100`` and ``coresPerGPU`` should be ``7``

Then run:

::

    sbatch submission_file
