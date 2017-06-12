Profiling the code
==================

Profiling the code consists in finding which parts of the algorithm **dominate
the computational time**, for your particular simulation setup.

Quick text-based profiling
--------------------------

The tools below allow to dump timing statistics into a simple text file, so
as to quickly profile the execution of a simulation.

On CPU
~~~~~~

Run the code with
`cProfile <http://docs.python.org/2/library/profile.html>`__ :

::

   python -m cProfile -s time fbpic_script.py > cpu.log

and then open the file ``cpu.log`` with a text editor.


On GPU
~~~~~~

Run the code with
`nvprof <http://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview>`__ :

::

    nvprof --log-file gpu.log python fbpic_script.py

in order to profile the device-side (i.e. GPU-side) code alone, or

::

    nvprof --log-file gpu.log python -m cProfile -s time fbpic_script.py > cpu.log

in order to simultaneously profile the device-side (i.e. GPU-side)
and host-side (i.e. CPU-side) code.
Then open the files ``gpu.log`` and/or ``cpu.log`` with a text editor.


Using a visual profiler
-----------------------

For a more detailed analysis, you can use visual profilers (i.e. profilers with
a graphical user interface).

On CPU
~~~~~~
Run the code with
`cProfile <http://docs.python.org/2/library/profile.html>`__, using binary output:

::

   python -m cProfile -o cpu.prof fbpic_script.py

and then open the file ``cpu.prof`` with `snakeviz <https://jiffyclub.github.io/snakeviz/>`__

::

   snakeviz cpu.prof

On GPU
~~~~~~
Run the code with
`nvprof <http://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview>`__,
using binary output:

::

    nvprof -o gpu.prof python fbpic_script.py

and then launch
`nvvp <http://docs.nvidia.com/cuda/profiler-users-guide/index.html#visual>`__:

::

    nvvp

And click ``File > Open``, in order to select the file ``gpu.prof``.

.. note::

    You do not need to run **snakeviz** or **nvvp** on the same machine on
    which the simulation was run. (In particular, **nvvp** does not need to
    have access to a GPU.) This means for example that, if your simulation
    was run on a remote cluster, you can simply transfer the
    files **cpu.prof** and/or **gpu.prof** to your local computer, and run
    **snakeviz** or **nvvp** locally.

    You can install **nvvp** on your local computer by installing the
    `cuda toolkit <http://developer.nvidia.com/cuda-downloads>`__.

.. note::

    When profiling the code with **nvprof**, the profiling data can quickly
    become very large. Therefore we recommend to profile the code only
    on a small number of PIC iterations (<1000).
