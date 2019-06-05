Beam initialization
===================

The functions below allow to **initialize a relativistic particle beam**
with various distributions in space (e.g. Gaussian, flat-top, from a file,
etc.). In each case, the **self-consistent space-charge fields** are
automatically calculated by the code, and added to the simulation.

.. autofunction:: fbpic.lpa_utils.bunch.add_particle_bunch

.. autofunction:: fbpic.lpa_utils.bunch.add_particle_bunch_gaussian

.. autofunction:: fbpic.lpa_utils.bunch.add_particle_bunch_from_arrays

.. autofunction:: fbpic.lpa_utils.bunch.add_particle_bunch_openPMD

.. autofunction:: fbpic.lpa_utils.bunch.add_particle_bunch_file

.. warning::

    The former methods to initialize electron beams have been updated to allow
    the use of arbitrary particle beams.

    Although the old methods (e.g. ``add_elec_bunch``) can still be used
    (for backward compatibility), we recommend using
    the new more flexible methods instead.