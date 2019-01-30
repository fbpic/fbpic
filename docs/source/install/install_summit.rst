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
    module load fftw/3.3.8
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

        conda install -c conda-forge numba scipy h5py cython cudatoolkit=8.0

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
    module load spectrum-mpi/10.2.0.10-20181214
    module load fftw/3.3.8
    module load python/3.7.0-anaconda3-5.3.0
    module load py-mpi4py/3.0.0-py3
    source activate fbpic

    jsrun -n <requestedNode> -a 6 -c 6 -g 6 python fbpic_script.py > cpu.log

Then run:

::

    bsub submission_script

Use ``bjobs`` to monitor the job.
