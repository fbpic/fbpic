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

Setting up Anaconda
~~~~~~~~~~~~~~~~~~~

- Add the following lines in your ``~/.bashrc``

    ::

        module load python/3.6

  Then log off and log in again in order for these changes to be active.

- Create a new conda environment and activate it.

    ::

        conda create -p $SCRATCH/fbpic_env python=3.6
        source activate $SCRATCH/fbpic_env

Installation of FBPIC and its dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Install the dependencies of ``fbpic``

   ::

       conda install numba scipy h5py mkl cudatoolkit=10.0
       conda install -c conda-forge mpi4py=*=*mpich*
       pip install cupy-cuda100

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

    salloc --time=00:30:00 --nodes=1 --partition es1  --constraint=es1_1080ti --qos=es_normal --gres=gpu:4 --cpus-per-task=2

Once the job has started, type

::

    srun --pty -u bash -i

in order to connect to the node that has been allocated. Then ``cd`` to
the directory where you prepared your input script and type

::

    source activate $SCRATCH/fbpic_env
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
    #SBATCH --gres=gpu:<gpuPerNode> --cpus-per-task=<cpuPerTask>

    module load python/3.6
    source activate $SCRATCH/fbpic_env

    mpirun -np <requestedRanks> python fbpic_script.py

where ``<gpuConstraint>`` and ``<gpuPerNode>`` should be:

    - For the nodes with four GTX 1080Ti GPUs, ``gpuConstraint=es1_1080ti``, ``gpuPerNode=4`` and ``cpuPerTask=8``
    - For the nodes with two V100 GPUs, ``gpuConstraint=es1_v100``, ``gpuPerNode=2`` and ``cpuPerTask=4``

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
