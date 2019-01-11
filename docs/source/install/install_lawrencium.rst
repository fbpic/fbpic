Installation on Lawrencium (LBNL)
=================================

`Lawrencium
<https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/lbnl-supercluster/lawrencium>`__
is a local cluster at the `Lawrence Berkeley National Lab <http://www.lbl.gov/>`__
(LBNL).

It has a few nodes with GPUs (Tesla K20 and K80, as well as GeForce GTX 1080 Ti).

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

       conda install numba==0.41 scipy h5py mkl
       conda install -c conda-forge mpi4py
       conda install cudatoolkit=8 pyculib


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

    salloc --time=00:30:00 --nodes=1 --partition lr_manycore  --constraint=lr_kepler --qos=lr_normal

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
    #SBATCH --partition=lr_manycore
    #SBATCH --constraint <gpuConstraint>
    #SBATCH --time <requestedTime>
    #SBATCH --nodes 1
    #SBATCH --qos lr_normal

    python <fbpic_script.py>

where ```<gpuConstraint>`` should be either:

    - ``lr_k20`` for a node with a single K20 GPU
    - ``lr_k80`` for a node with four K80 GPUs
    - ``lr_pascal`` for a node with four GTX 1080Ti GPUs

for more information on the available nodes, see
`this page <https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/lbnl-supercluster/lawrencium>`__.

Then run:

::

    sbatch submission_file

In order to see the queue:

::

    squeue -p lr_manycore

Visualizing the results through Jupyter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lawrencium provides access to the cluster via Jupyter, at `https://lrc-jupyter.lbl.gov <https://lrc-jupyter.lbl.gov>`__. Once you logged in and opened a Jupyter notebook, you can type in a cell:

::

	!pip install openPMD-viewer --user

in order to install `openPMD-viewer <https://github.com/openPMD/openPMD-viewer>`__.


Transfering data to your local computer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to transfer your data to your local machine, you need to
connect to the transfer node. From a Lawrencium login node, type:

::

    ssh lrc-xfer.scs00

You can then use for instance ``rsync`` to transfer data to your local
computer.
