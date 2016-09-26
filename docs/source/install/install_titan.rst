Installation on Titan (OLCF)
================================================

`Titan <https://www.olcf.ornl.gov/titan/>`__ is a GPU supercomputer at the
`Oakridge Leadership Computing Facility
<https://www.olcf.ornl.gov/>`__ (OLCF).

Each node consists of 1 Nvidia K20 device.

Installing FBPIC
----------------

Installation of Anaconda
~~~~~~~~~~~~~~~~~~~~~~~~

In order to install FBPIC, you need to first install `Anaconda <https://www.continuum.io/why-anaconda>`__:

-  Download Miniconda:

   ::

       wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh

-  Execute the installation script, and use ``/ccs/proj/<project_id>``
   as an install directory, so that the installation is accessible to
   the compute nodes.

   ::

       bash miniconda.sh -b -p /ccs/proj/<project_id>/miniconda2

   where the bracketed text should be replaced by the values for your
   account.

-  Add the following lines at the end of your .bashrc

   ::

       module load python_anaconda
       export PATH=/ccs/proj/<project_id>/miniconda2/bin:$PATH
       export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ccs/proj/<project_id>/miniconda2/lib
       export PYTHONPATH=$PYTHONPATH:/ccs/proj/<project_id>/miniconda2/lib/python2.7/site-packages/
       export PYTHONPATH=$PYTHONPATH:/sw/xk6/python_anaconda/2.3.0/sles11.3_gnu4.8.2/lib/python2.7/site-packages/

   where again the bracketed text should be replaced by the values for
   your account. The first line gives access to the ``mpi4py``
   installation of Titan (which is contained in the module
   ``python_anaconda``) while the other lines allow you to use packages
   that you install locally.

Then execute the modified .bashrc file: ``source .bashrc``.

Installation of FBPIC and its dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Clone the ``fbpic`` repository using git.

-  ``cd`` into the top folder of ``fbpic`` and install the dependencies:

   ::

       conda install -c conda-forge --file requirements.txt

-  Uninstall mpi4py (in order to use the ``mpi4py`` from Titan instead):

   ::

       conda uninstall mpi4py

       
-  Install the ``accelerate`` package in order to be able to run on GPUs

   ::

       conda install cudatoolkit=7.0
       conda install accelerate

   (The ``accelerate`` package is not free, but there is a 30-day free
   trial period, which starts when the above command is entered. For
   further use beyond 30 days, one option is to obtain an academic
   license, which is also free. To do so, please visit `this
   page <https://www.continuum.io/anaconda-academic-subscriptions-available>`__.)

-  Install ``fbpic`` into your local set of packages

   ::

       python setup.py install


Running simulations
------------------------------------------

In order to create a new simulation, create a new directory in
``$MEMBERWORK/`` and copy your input script there:

::

    mkdir $MEMBERWORK/<project_id>/<simulation name>
    cp fbpic_script.py $MEMBERWORK/<project_id>/<simulation name>

Interactive jobs
~~~~~~~~~~~~~~~~

In order to request an interactive job:

::

    qsub -I -A <project_id> -l nodes=1,walltime=00:30:00 -q debug

Once the job has started, switch to your simulation directory

::

    cd $MEMBERWORK/<project_id>/<simulation name>

Then use ``aprun`` to launch the job on a GPU (even for single-node job)

::

    aprun -n 1 -N 1 python <fbpic_script.py>

Batch jobs
~~~~~~~~~~

Create a file ``submission_script`` with contains the following text:

::

    #!/bin/bash
    #PBS -A <project_id>
    #PBS -l walltime=<your walltime>
    #PBS -l nodes=<number of nodes>

    cd  $MEMBERWORK/<project_id>/<simulation name>

    aprun -n <number of nodes> -N 1 python fbpic_script.py

Then submit it with:

::

   qsub submission_script
