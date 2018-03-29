Performing parameter scans in parallel
======================================

On some clusters, a single compute node may contain several GPUs.
In this case, it can sometimes be useful to run a separate PIC simulation on
each GPU, as part of a parameter scan (where e.g. each separate PIC
simulation is run with a different value of the laser intensity).

However, launching these separate simulations with separate Python scripts may
not be efficient, since it is possible that all simulations will be run on
the same GPU, while the other GPUs will be left idle.

Therefore, FBPIC provides the possibility to launch these different simulations
as a single command with ``mpirun``, whereby each MPI rank will perform a separate
PIC simulation, and FBPIC will make sure that each simulation runs on a separate
GPU. This feature is activated by setting ``use_all_mpi_ranks=False`` in the
:any:`Simulation` object, in order to instruct FBPIC to use only one MPI rank per
simulation instead of *all* MPI ranks for a single simulation.

Here is an example on how to structure the input script and specify the parameter
to be varied between the separate simulations:
:download:`parametric_script.py <../example_input/parametric_script.py>`
