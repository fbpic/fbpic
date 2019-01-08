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
    module load gcc/4.8.5
    module load spectrum-mpi/10.2.0.10-20181214
    module load fftw/3.3.8
    module load python/3.7.0-anaconda3-5.3.0

Then create a new `conda` environment

::

    conda create -n fbpic python=3
    source activate fbpic

Installation of FBPIC and its dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Install the dependencies of ``fbpic``

    ::

        conda install -c conda-forge numba=0.42 scipy h5py cython cudatoolkit=8.0

- Install ``pyfftw``

    ::

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
    module load gcc/4.8.5
    module load spectrum-mpi/10.2.0.10-20181214
    module load fftw/3.3.8
    module load python/3.7.0-anaconda3-5.3.0
    module load py-mpi4py/3.0.0-py3
    source activate fbpic
    export NUMBA_NUM_THREADS=7
    export OMP_NUM_THREADS=7

    jsrun -n 1 -a 1 -c 1 -g 1 python <fbpic_script.py>

Batch job
~~~~~~~~~

Create a new file named ``submission_script`` in the same directory as
your input script. Within this new file, copy the
following text (and replace the bracketed text by the proper values).

::

    #!/bin/bash
    #BSUB -J my_job
    #BSUB -W <requestedTime>
    #BSUB -nnodes <requestedNode>
    #BSUB -P <accountNumber>

    module purge
    module load gcc/4.8.5
    module load cuda/9.1.85
    module load spectrum-mpi/10.2.0.10-20181214
    module load fftw/3.3.8
    module load python/3.7.0-anaconda3-5.3.0
    module load py-mpi4py/3.0.0-py3
    source activate fbpic

    export NUMBA_NUM_THREADS=7
    export OMP_NUM_THREADS=7
    export FBPIC_ENABLE_GPUDIRECT=1

    jsrun -n <requestedNode> -a 6 -c 42 -g 6 --smpiargs="-gpu" python fbpic_script.py > cpu.log

Then run:

::

    bsub submission_script


.. note::

    Note that, in the above script, ``module load cuda/9.1.85``,
    ``export FBPIC_ENABLE_GPUDIRECT=1`` and ``--smpiargs="-gpu"``
    are only needed if you wish to use the **cuda-aware** MPI.

Use ``bjobs`` to monitor the job.
