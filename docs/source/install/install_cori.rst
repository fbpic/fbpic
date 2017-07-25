Installation on Cori (LBNL)
===========================

`Cori
<http://www.nersc.gov/users/computational-systems/cori/>`__ is a supercomputer
at LBNL. It has two types of nodes:

- Haswell (32-core Xeon processor)
- KNL (68-core Xeon-Phi processor)

.. warning::

    FBPIC has not been optimized for KNL, and thus its performance on these
    nodes is poor. It is strongly recommended to use the Haswell nodes
    when running FBPIC.

Installation of FBPIC
---------------------

Installation of Anaconda
~~~~~~~~~~~~~~~~~~~~~~~~

In order to download and install Anaconda and FBPIC, follow the steps below:

-  Download Miniconda:

   ::

       wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

-  Execute the Miniconda installation script

   ::

       bash Miniconda-latest3-Linux-x86_64.sh

  Then type

  ::

     source .bashrc

Installation of FBPIC and its dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Install the dependencies of ``fbpic``

   ::

       conda install -c conda-forge numba scipy h5py pyfftw mpi4py

-  Install ``fbpic``

   ::

      pip install fbpic

Running simulations
-------------------

Preparing a new simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~

It is adviced to use the directory ``$SCRATCH`` for faster I/O.

In order to prepare a new simulation, create a new subdirectory within
the above-mentioned directory, and copy your input script there.

Interactive jobs
~~~~~~~~~~~~~~~~

In order to request a node with a GPU:

::

    salloc --time=00:30:00 --nodes=1 --partition debug  -C haswell

Once the job has started, you will directly be logged into the node. Then
``cd`` to the directory where you prepared your input script and type

::

    python <fbpic_script.py>

Batch job
~~~~~~~~~

Create a new file named ``submission_file`` in the same directory as
your input script (typically this directory is a subdirectory of
``$SCRATCH``). Within this new file, copy the following text
(and replace the bracketed text by the proper values).

::

    #!/bin/bash
    #SBATCH -J my_job
    #SBATCH --partition=regular
    #SBATCH -C haswell
    #SBATCH --time <requestedTime>
    #SBATCH --nodes 1

    export NUMBA_NUM_THREADS=<Number of threads per MPI rank>
    mpirun -np <Number of MPI ranks> python <fbpic_script.py>

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
