Installation on Perlmutter (NERSC)
==================================

`Perlmutter
<https://docs.nersc.gov/systems/perlmutter/>`__
is a high-performance cluster at `NERSC
<http://www.nersc.gov/>`__.

Its compute nodes are equipped with 4 A100 NVIDIA GPUs.

Installation of FBPIC
---------------------

Setting up Anaconda
~~~~~~~~~~~~~~~~~~~

- Add the following lines in your ``~/.bashrc``

    ::

        module load python

  Then log off and log in again in order for these changes to be active.

- Create a new conda environment and activate it.

    ::

        conda create -n fbpic
        source activate fbpic

Installation of FBPIC and its dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Install the dependencies of ``fbpic``

   ::

       conda install -c conda-forge python numba scipy h5py mkl cudatoolkit=11.3
       pip install cupy-cuda113

-  Install ``fbpic``

   ::

       pip install fbpic

Running simulations
-------------------

Interactive jobs
~~~~~~~~~~~~~~~~

In order to request a node with a GPU:

::

    salloc -t 00:30:00 -N 1 -C gpu --ntasks-per-node=4 -A <account_number>

Then ``cd`` to the directory where you prepared your input script and type

::

    source activate fbpic
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
    #SBATCH -A <account_number>
    #SBATCH -C gpu
    #SBATCH --time <requestedTime>
    #SBATCH --ntasks-per-node=4

    module load python
    source activate fbpic

    python fbpic_script.py

Then run:

::

    sbatch submission_file
