Installation on Lawrencium (LBNL)
=================================

`Lawrencium
<https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/lbnl-supercluster/lawrencium>`__
is a local cluster at the `Lawrence Berkeley National Lab <http://www.lbl.gov/>`__
(LBNL).

It has a few nodes with K20 and K80 Nvidia GPUs.

Connecting to Lawrencium
------------------------

Lawrencium uses a one-time password (OTP) system. Before being able to
connect to Lawrencium via ssh, you need to configure an OTP Token, using
`these
instructions <https://commons.lbl.gov/display/itfaq/Installing+and+Configuring+the+OTP+Token>`__.

Once your OTP token is configured, you can connect by using

::

    ssh <username>@lrc-login.lbl.gov

When prompted for the password, generate a new one-time password with
the Pledge application, and enter it at the prompt.

Installation of FBPIC
---------------------

Installation of Anaconda
~~~~~~~~~~~~~~~~~~~~~~~~

In order to download and install Anaconda and FBPIC, follow the steps
below:

-  Download Miniconda:

   ::

       wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh

-  Execute the Miniconda installation script

   ::

       bash Miniconda-latest-Linux-x86_64.sh

   Accept the default location of the installation, and answer yes
   when the installer proposes to modify your ``PATH`` inside your ``.bashrc``.

-  Add the following lines at the end of your ``.bashrc``

   ::

       module load glib/2.32.4
       module load cuda

  and type

  ::

     source .bashrc

Installation of FBPIC and its dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Install the dependencies of ``fbpic``

   ::

       conda install -c conda-forge numba scipy h5py pyfftw mpi4py accelerate

   (NB: The ``accelerate`` package is not free, but there is a 30-day free
   trial period, which starts when the above command is entered. For
   further use beyond 30 days, one option is to obtain an academic
   license, which is also free. To do so, please visit `this
   link <https://www.continuum.io/anaconda-academic-subscriptions-available>`__.)


-  Install ``fbpic``

   ::

      git clone git://github.com/fbpic/fbpic.git
      cd fbpic
      python setup.py install

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
    #SBATCH --constraint lr_kepler
    #SBATCH --time <requestedTime>
    #SBATCH --nodes 1
    #SBATCH --qos lr_normal
    #SBATCH -e my_job.%j.err
    #SBATCH -o my_job.%j.out

    python <fbpic_script.py>

Then run:

::

    sbatch submission_file

In order to see the queue:

::

    squeue -p lr_manycore

Transfering data to your local computer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to transfer your data to your local machine, you need to
connect to the transfer node. From a Lawrencium login node, type:

::

    ssh lrc-xfer.scs00

You can then use for instance ``rsync`` to transfer data to your local
computer.
