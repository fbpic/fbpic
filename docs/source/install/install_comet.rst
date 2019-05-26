Installation on Comet (SDSC)
============================

`Comet <https://portal.xsede.org/sdsc-comet>`__ is an HPC cluster at the
`San Diego Supercomputer Center <http://www.sdsc.edu/>`__ (SDSC).

It provides both NVIDIA K80 and P100 GPU-based resources.

Installation of FBPIC
---------------------

Installation of Anaconda
~~~~~~~~~~~~~~~~~~~~~~~~

In order to download and install Anaconda and FBPIC, follow the steps
below:

-  Download Miniconda:

   ::

       wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

-  Execute the Miniconda installation script

   ::

       bash Miniconda3-latest-Linux-x86_64.sh

   when asked whether to append the path of ``miniconda``
   to your ``.bashrc``, answer yes.


Installation of FBPIC and its dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Install the dependencies of ``fbpic`` (except ``mpi4py`` and ``cupy``)

   ::

      conda install -c conda-forge numba scipy h5py mkl cudatoolkit=9.2


-  Install ``mpi4py`` and ``cupy``

   ::

      module purge
      module load gnutools
      module load gnu openmpi_ib
      pip install cupy-cuda92
      env MPICC=/opt/openmpi/gnu/ib/bin/mpicc pip install mpi4py --user


-  Install ``fbpic``

   ::

      pip install fbpic

Running simulations
-------------------

This section briefly describes how to submit simulations. For more information,
see `Comet's User Guide <http://www.sdsc.edu/support/user_guides/comet.html>`__.

Preparing a new simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to prepare a new simulation, create a new subdirectory within
the above-mentioned directory, and copy your input script there.

Interactive jobs
~~~~~~~~~~~~~~~~

In order to request a node with 4 K80 GPUs:

::

    salloc -p gpu --gres=gpu:k80:4 -t 00:30:00

Once the job has started, type

::

    srun --pty /bin/bash

in order to connect to the node that has been allocated. Then ``cd`` to
the directory where you prepared your input script and type

::

    python <fbpic_script.py>

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
