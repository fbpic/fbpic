Beam initialization
===================

The functions below allow to **initialize a relativistic electron beam**
with various distributions in space (e.g. Gaussian, flat-top, from a file,
etc.). In each case, the **self-consistent space-charge fields** are
automatically calculated by the code, and added to the simulation.

.. autofunction:: fbpic.lpa_utils.bunch.add_elec_bunch

.. autofunction:: fbpic.lpa_utils.bunch.add_elec_bunch_gaussian

.. autofunction:: fbpic.lpa_utils.bunch.add_elec_bunch_from_arrays

.. autofunction:: fbpic.lpa_utils.bunch.add_elec_bunch_openPMD

.. autofunction:: fbpic.lpa_utils.bunch.add_elec_bunch_file
