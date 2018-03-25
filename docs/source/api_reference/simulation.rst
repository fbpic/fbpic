The Simulation class
======================

The :any:`Simulation` class is top-level class in FBPIC. It contains all
the simulation data, and has the high-level method :any:`step`
that performs the PIC cycle.

In addition, its method :any:`add_new_species` allows to create new particle
species, and its method :any:`set_moving_window` activates the moving window.

.. autoclass:: fbpic.main.Simulation
   :members: step, add_new_species, set_moving_window
