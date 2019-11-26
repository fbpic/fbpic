Installation on Cori (NERSC)
============================

`Cori
<http://www.nersc.gov/users/computational-systems/cori/>`__ is
a high-performance cluster at `NERSC
<http://www.nersc.gov/>`__.

Cori has two types of nodes:

- Haswell (32-core Xeon processor)
- KNL (68-core Xeon-Phi processor)

.. warning::

    FBPIC has not been optimized for KNL, and thus its performance on these
    nodes is poor. It is strongly recommended to use the Haswell nodes
    when running FBPIC on Cori.

Installation of FBPIC and its dependencies
------------------------------------------

In order to install FBPIC, follow the steps below:

-  Set your environment to use the Anaconda distribution:

   ::

    module load python/2.7-anaconda

-  Install the missing dependencies of FBPIC

   ::

       pip install --upgrade numba llvmlite tbb --user

-  Install FBPIC

   ::

       pip install fbpic --user

Running simulations
-------------------

Preparing a new simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~

It is adviced to use the directory ``$SCRATCH`` for faster I/O.

In order to prepare a new simulation, create a new subdirectory within
the above-mentioned directory, and copy your input script there.

Interactive jobs
~~~~~~~~~~~~~~~~

In order to request a Haswell node on Cori, use the following command.

::

    salloc --time=00:30:00 --nodes=1 --partition debug  -C haswell

Once the job has started, you will directly be logged into the node. Then
``cd`` to the directory where you prepared your input script and type

::

    module load python/2.7-anaconda
    python <fbpic_script.py>

Batch job
~~~~~~~~~

Create a new file named ``submission_file`` in the same directory as
your input script (typically this directory is a subdirectory of
``$SCRATCH``). Within this new file, copy the following text,
and replace the bracketed text by the proper values.

::

    #!/bin/bash
    #SBATCH -J my_job
    #SBATCH --partition=regular
    #SBATCH -C haswell
    #SBATCH --time <requested time>
    #SBATCH --nodes <n_nodes>

    module load python/2.7-anaconda
    export NUMBA_THREADING_LAYER=tbb
    export NUMBA_NUM_THREADS=16
    export MKL_NUM_THREADS=16

    srun -n <n_mpi> -c 32 --cpu-bind=cores python <fbpic_script.py>

Then run:

    ::

        sbatch submission_file
 

Visualizing the results through Jupyter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cori provides access to the cluster via Jupyter, at
`https://jupyter-dev.nersc.gov <https://jupyter-dev.nersc.gov>`__.
Once you logged in and opened a Jupyter notebook, you can type in a cell:

::

	!pip install openPMD-viewer --user

in order to install `openPMD-viewer <https://github.com/openPMD/openPMD-viewer>`__.
