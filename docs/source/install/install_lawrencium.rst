Installation on Lawrencium (LBNL)
=================================

`Lawrencium
<https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/lbnl-supercluster/lawrencium>`__
is a local cluster at the `Lawrence Berkeley National Lab <http://www.lbl.gov/>`__
(LBNL).

It has 24 nodes with GPUs:

    - 12 nodes with four GTX 1080Ti GPUs each
    - 12 nodes with two V100 GPUs each

Connecting to Lawrencium
------------------------

Lawrencium uses a one-time password (OTP) system. Before being able to
connect to Lawrencium via ssh, you need to configure an OTP Token, using
`these
instructions <https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/getting-started/new-user-information>`__.

Once your OTP token is configured, you can connect by using

::

    ssh <username>@lrc-login.lbl.gov

Installation of FBPIC
---------------------

Installation of Anaconda
~~~~~~~~~~~~~~~~~~~~~~~~

In order to download and install Anaconda and FBPIC, follow the steps
below:

-  Download Miniconda:

   ::

       wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

-  Execute the Miniconda installation script, and use ``/global/scratch/<yourUsername>`` as an install directory, for faster disk access.

   ::

       bash Miniconda3-latest-Linux-x86_64.sh -p /global/scratch/<yourUsername>/miniconda3

   where the bracketed text should be replaced by your username. Then type

  ::

       source .bashrc

Installation of FBPIC and its dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Install the dependencies of ``fbpic``

   ::

       conda install -c conda-forge numba==0.42 scipy h5py mkl mpi4py
       conda install -c conda-forge cudatoolkit=8 pyculib

-  Install ``fbpic``

   ::

	pip install fbpic

Running simulations
-------------------

Preparing a new simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~

It is adviced to use the directory ``/global/scratch/<yourUsername>``
for faster I/O access, where ``<yourUsername>`` should be replaced by
your username.

In order to prepare a new simulation, create a new subdirectory within
the above-mentioned directory, and copy your input script there.

Interactive jobs
~~~~~~~~~~~~~~~~

In order to request a node with a GPU:

::

    salloc --time=00:30:00 --nodes=1 --partition es1  --constraint=es1_1080ti --qos=es_normal

Once the job has started, type

::

    srun --pty -u bash -i

in order to connect to the node that has been allocated. Then ``cd`` to
the directory where you prepared your input script and type

::

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
    #SBATCH --partition es1
    #SBATCH --qos es_normal
    #SBATCH --constraint <gpuConstraint>
    #SBATCH --time <requestedTime>
    #SBATCH --ntasks <requestedRanks>
    #SBATCH --ntasks-per-node <gpuPerNode>

    mpirun -np <requestedRanks> python fbpic_script.py

where ``<gpuConstraint>`` and ``<gpuPerNode>`` should be:

    - For the nodes with four GTX 1080Ti GPUs, ``gpuConstraint=es1_1080ti`` and ``gpuPerNode=4``
    - For the nodes with two V100 GPUs, ``gpuConstraint=es1_v100`` and ``gpuPerNode=2``

for more information on the available nodes, see
`this page <https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/lbnl-supercluster/lawrencium>`__.

Then run:

::

    sbatch submission_file

In order to see the queue:

::

    squeue -p es1

Visualizing the results through Jupyter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lawrencium provides access to the cluster via Jupyter, at `https://lrc-jupyter.lbl.gov <https://lrc-jupyter.lbl.gov>`__. Once you logged in and opened a Jupyter notebook, you can type in a cell:

::

	!pip install openPMD-viewer --user

in order to install `openPMD-viewer <https://github.com/openPMD/openPMD-viewer>`__.
