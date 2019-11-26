Profiling the code
==================

Profiling the code consists in finding which parts of the algorithm **dominate
the computational time**, for your particular simulation setup.

Profiling the code executed on CPU
----------------------------------

Getting the results in a simple text file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dumping the profiling results in a text file allows you
to quickly profile the execution of a simulation.

Run the code with
`cProfile <http://docs.python.org/2/library/profile.html>`__ :

::

   python -m cProfile -s time fbpic_script.py > cpu.log

and then open the file ``cpu.log`` with a text editor.

Using a visual profiler
~~~~~~~~~~~~~~~~~~~~~~~

For a more detailed analysis, you can use a visual profiler (i.e. profilers with
a graphical user interface).

Run the code with
`cProfile <http://docs.python.org/2/library/profile.html>`__, using binary output:

::

   python -m cProfile -o cpu.prof fbpic_script.py

and then open the file ``cpu.prof`` with `snakeviz <https://jiffyclub.github.io/snakeviz/>`__

::

   snakeviz cpu.prof

Profiling the code executed on GPU
----------------------------------

Two profiling tools exists for GPU:

    - `nvprof <http://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview>`__
    - `Nsight Systems <https://docs.nvidia.com/nsight-systems/>`__ (which can be installed from `this page <https://developer.nvidia.com/gameworksdownload#?dn=nsight-systems-2019-5>`__)

Instructions here are given for both tools.

Getting the results in a simple text file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- For **nvprof**: First run the code with ``nvprof``

    ::

        nvprof --log-file gpu.log python fbpic_script.py

    and then open the file ``gpu.log`` with a standard text editor.

- For **Nsight Systems**: Run the code with ``nsys profile``

    ::

        nsys profile --stats=true python fbpic_script.py

    The profiling information is printed directly in the Terminal output.

.. note::

    In order to simultaneously profile the device-side (i.e. GPU-side)
    and host-side (i.e. CPU-side) code, you can use:

    ::

        nvprof --log-file gpu.log python -m cProfile -s time fbpic_script.py > cpu.log

Using a visual profiler
~~~~~~~~~~~~~~~~~~~~~~~

- For **nvprof**: First run the code with ``nvprof``

    ::

        nvprof -o gpu.prof python fbpic_script.py

    and then launch
    `nvvp <http://docs.nvidia.com/cuda/profiler-users-guide/index.html#visual>`__

    ::

        nvvp

    And click ``File > Open``, in order to select the file ``gpu.prof``.

- For **Nsight System**: First run the code with ``nsys profile``

    ::

        nsys profile python fbpic_script.py

    and then launch ``nsight-sys``:

    ::

        nsight-sys

    Click ``File > Open``, navigate to the folder in which you ran the simulation,
    and open the file that ends in ``.qdrep``.

.. note::

    You do not need to run **snakeviz** or **nvvp**/**nsight-sys** on the same machine on
    which the simulation was run. (In particular, **nvvp**/**nsight-sys** does not need to
    have access to a GPU.) This means for example that, if your simulation
    was run on a remote cluster, you can simply transfer the
    files **cpu.prof** and/or **gpu.prof** to your local computer, and run
    **snakeviz** or **nvvp** locally.

    You can install **nvvp** on your local computer by installing the
    `cuda toolkit <http://developer.nvidia.com/cuda-downloads>`__.

.. note::

    When profiling the code with **nvprof** or **nsys**, the profiling data can
    quickly become very large. Therefore we recommend to profile the code only
    on a small number of PIC iterations (<1000).

Profiling MPI simulations
-------------------------

One way to profile MPI simulations is to write **one file per MPI rank**. In
this case, each file will contain only the profiling data of the corresponding
MPI process.

Profiling the CPU code
~~~~~~~~~~~~~~~~~~~~~~

One way to create one file per MPI rank is to **modify your FBPIC script**,
in the following way:

- Add the following lines at the beginning of the file:

    ::

        import cProfile, sys
        from mpi4py.MPI import COMM_WORLD as comm

- Replace the line:

    ::

        sim.step( N_step )

  by the following set of lines:

    ::

        # First step: do not profile (includes just-in-time compilation)
        sim.step(1)

        # Profile the next N_step
        pr = cProfile.Profile()
        pr.enable()
        sim.step( N_step )
        pr.disable()

        # Dump results:
        # - for binary dump
        pr.dump_stats('cpu_%d.prof' %comm.rank)
        # - for text dump
        with open( 'cpu_%d.txt' %comm.rank, 'w') as output_file:
            sys.stdout = output_file
            pr.print_stats( sort='time' )
            sys.stdout = sys.__stdout__

Then run your FBPIC script with MPI as usual, e.g. with 4 MPI ranks:

    ::

        mpirun -np 4 python fbpic_script.py


Profiling the GPU code
~~~~~~~~~~~~~~~~~~~~~~

Use the following command:

::

    mpirun -np 2 nvprof -o gpu_%q{<RANK>}.prof python fbpic_script.py

where ``<RANK>`` should be replaced by the following name depending on
your MPI distribution:

    - For openmpi: ``OMPI_COMM_WORLD_RANK``
    - For mpich: ``PMI_RANK``

(If you are unsure which name to use, type ``mpirun -np 2 printenv | grep RANK``.)

``nvprof`` will then create one profile file per MPI rank. You can load these
files on the same timeline within ``nvvp`` by clicking
``File > Import > Nvprof > Multiple processes > Browse``. For more information,
see `this page <https://devblogs.nvidia.com/cuda-pro-tip-profiling-mpi-applications/>`__.
