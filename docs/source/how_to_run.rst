How to run the code
===================

Once installed (see :doc:`install/installation`), FBPIC is available as a **Python
module** on your system. Thus, a simulation is setup by creating a
**Python script** that imports and uses FBPIC's functionalities.

Script examples
----------------

You can download examples of FBPIC scripts below (which you can then modify
to suit your needs):

.. toctree::
    :maxdepth: 1

    example_input/lwfa_script.rst
    example_input/ionization_script.rst
    example_input/boosted_frame_script.rst

(See the documentation of :any:`Particles.make_ionizable` for more information on ionization,
and the section :doc:`advanced/boosted_frame` for more information on the boosted frame.)

The different FBPIC objects that are used in the above simulation scripts are
defined in the section :doc:`api_reference/api_reference`.

Running the simulation
----------------------

The simulation is then run by typing

::

   python fbpic_script.py

where ``fbpic_script.py`` should be replaced by the name of your
Python script: either ``lwfa_script.py`` or
``boosted_frame_script.py`` for the above examples.

If an MPI implementation is available within the compute environment and
the ``mpi4py`` package is installed, the computation can be scaled to multiple
processes (e.g. 4) by running

::

  mpirun -n 4 python fbpic_script.py

.. warning::

    Note that, depending on the size of the simulation, running with
    multiple MPI processes is not necessarily faster. In addition,
    MPI simulations require using a finite order (e.g. ``n_order=32``)
    for the field solver. Please read the
    documentation on the :doc:`parallelisation of FBPIC
    <overview/parallelisation>` before using this feature.

.. note::

   When running on CPU, **multi-threading** is enabled by default, and the
   default number of threads is the number of (virtual) cores on your system.
   You can modify this with environment variables:

   - To modify the number of threads (e.g. set it to 8 threads):

   ::

    export MKL_NUM_THREADS=8
    export NUMBA_NUM_THREADS=8
    python fbpic_script.py

   - To disable multi-threading altogether:

   ::

    export FBPIC_DISABLE_THREADING=1
    export MKL_NUM_THREADS=1
    export NUMBA_NUM_THREADS=1
    python fbpic_script.py

   It can also happen that an alternative threading backend is selected by Numba
   during installation. It is therefore sometimes required to set
   ``OMP_NUM_THREADS`` in addition to (or instead of) ``MKL_NUM_THREADS``.

   When running in a Jupyter notebook, environment variables can be set by
   executing the following command at the beginning of the notebook:

   ::

    import os
    os.environ['MKL_NUM_THREADS']='1'

.. note::

  On systems with more than one CPU socket per node, multi-threading
  can become inefficient if the threads are distributed across sockets.
  It can be advantageous to use one MPI process per socket and to limit the
  number of threads to the number of physical cores of each socket. In addition,
  it can be necessary to explicitly bind all threads of an MPI process to the
  same socket.

  On a machine with 2 sockets and 12 physical cores per socket, the following
  commands spawn 2 MPI processes each with 12 threads bound to a single socket:

  - Using the SLURM workload manager:

    ::

      export MKL_NUM_THREADS=12
      export NUMBA_NUM_THREADS=12
      srun -n 2 --cpu_bind=socket python fbpic_script.py

  - Using the ``mpirun`` executable:

    ::

      export MKL_NUM_THREADS=12
      export NUMBA_NUM_THREADS=12
      mpirun -n 2 --bind-to socket python fbpic_script.py

.. note::

  When running on GPU with MPI domain decomposition, it is possible to enable
  the CUDA GPUDirect technology. GPUDirect enables direct communication of
  CUDA device arrays between GPUs over MPI without explicitly copying the data
  to CPU first, resulting in reduced latencies and increased bandwidth. As this
  feature requires a CUDA-aware MPI implementation that supports GPUDirect,
  it is disabled by default and should be used with care.

  To activate this feature, the user needs to set the following
  environment variable:

  ::

    export FBPIC_ENABLE_GPUDIRECT=1


Visualizing the simulation results
----------------------------------

The code outputs HDF5 files, that comply with the
`openPMD standard <http://www.openpmd.org/#/start>`_. As such, they
can be visualized for instance with the `openPMD-viewer
<https://github.com/openPMD/openPMD-viewer>`_). To do so, first
install the openPMD-viewer by typing

::

   pip install openpmd-viewer

or

::

   conda install -c conda-forge openpmd-viewer

And then type

::

   openPMD_notebook

and follow the instructions in the notebook that pops up. (NB: the
notebook only shows some of the capabilities of the openPMD-viewer. To
learn more, see the tutorial notebook on the  `Github repository
<https://github.com/openPMD/openPMD-viewer>`_ of openPMD-viewer).

If you want to render your simulation results in 3D, see the section
:doc:`advanced/3d_visualization`.
