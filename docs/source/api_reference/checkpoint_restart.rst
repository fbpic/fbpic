Checkpoints and restarts
=========================

For very long simulations, it is good to set
**checkpoints**. Checkpoints are files that contain all the simulation
data at one given iteration, so that the simulation can be later
**restarted** from this iteration.

Checkpoints are useful when there is a risk that the simulation
crashes before the end (e.g. because of the finite walltime on HPC
clusters). In this case, thanks to checkpoints, the simulation can be restarted
without having to run it again from the beginning.

Setting checkpoints
-----------------------

.. autofunction:: fbpic.openpmd_diag.set_periodic_checkpoint

Restarting a simulation
--------------------------
		  
.. autofunction:: fbpic.openpmd_diag.restart_from_checkpoint

