The Simulation class
======================

The `Simulation` class is top-level class in FBPIC. It contains all
the simulation data, and has the high-level methods to perform the PIC cycle.

    The `Simulation` class has several important attributes:

    - `fld`, a `Fields` object which contains the field information
    - `ptcl`, a list of `Particles` objects (one per species)
    - `diags`, a list of diagnostics to be run during the simulation
    - `comm`, a `BoundaryCommunicator`, which contains the MPI decomposition

.. autoclass:: fbpic.main.Simulation
   :members: step, set_moving_window
