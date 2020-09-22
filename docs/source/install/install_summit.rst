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

First load the relevant modules:

::

    module purge
    module load gcc
    module load fftw
    module load python/3.7.0-anaconda3-5.3.0

Then create a new `conda` environment

::

    conda create -n fbpic --clone base
    source activate fbpic

Installation of FBPIC and its dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Install the dependencies of ``fbpic``

    ::

        conda install cython numba=0.49 cudatoolkit=9.0

- Install ``pyfftw``

    ::

        pip install pyfftw

-  Install ``fbpic``

    ::

        pip install fbpic

Running simulations
-------------------

In order to create a new simulation, create a new directory in
``$MEMBERWORK/`` and copy your input script there:

::

    mkdir $MEMBERWORK/<project_id>/<simulation name>
    cp fbpic_script.py $MEMBERWORK/<project_id>/<simulation name>

Also, before launching any new job, please make sure that the `conda`
environment ``fbpic`` is **not** loaded, for instance by using

::

    source deactivate fbpic

Interactive jobs
~~~~~~~~~~~~~~~~

In order to request one node for 20 minutes:

::

    bsub -W 00:20 -nnodes 1 -P <account_number> -Is /bin/bash

Then ``cd`` to the directory where you prepared your input script and type

::

    module purge
    module load gcc
    module load cuda/9.1.85
    module load fftw
    module load python/3.7.0-anaconda3-5.3.0
    source activate fbpic

    export CUPY_CACHE_DIR=$MEMBERWORK/<project_id>/.cupy/kernel_cache
    export FBPIC_DISABLE_CACHING=1
    export NUMBA_NUM_THREADS=1
    export OMP_NUM_THREADS=1

    jsrun -n 1 -a 1 -c 1 -g 1 python <fbpic_script.py>

where ``<project_id>`` should be replaced by your project account number.

Batch job
~~~~~~~~~

Create a new file named ``submission_script`` in the same directory as
your input script. Within this new file, copy the
following text (and replace the bracketed text by the proper values).

::

    #!/bin/bash
    #BSUB -J my_job
    #BSUB -W <requestedTime>
    #BSUB -nnodes <requestedNodes>
    #BSUB -P <accountNumber>

    module purge
    module load gcc
    module load cuda/9.1.85
    module load fftw
    module load python/3.7.0-anaconda3-5.3.0
    source activate fbpic

    export CUPY_CACHE_DIR=$MEMBERWORK/<project_id>/.cupy/kernel_cache
    export FBPIC_ENABLE_GPUDIRECT=0
    export FBPIC_DISABLE_CACHING=1
    export NUMBA_NUM_THREADS=1
    export OMP_NUM_THREADS=1

    jsrun -n <requestedMPIRanks> -a 1 -c 1 -g 1 --smpiargs="-gpu" python fbpic_script.py > cpu.log

where ``<project_id>`` should be replaced by your project account number, and
``<requestedNodes`` and ``<requestedMPIRanks>`` should be replaced by the
number of nodes and MPI ranks (use 6 MPI ranks per Summit node).

Then run:

::

    bsub submission_script


.. note::

    Note that, in the above script, ``--smpiargs="-gpu"`
    is in fact only needed when ``export FBPIC_ENABLE_GPUDIRECT=1``,
    i.e. when attempting to use the **cuda-aware** MPI.

Use ``bjobs`` to monitor the job.
