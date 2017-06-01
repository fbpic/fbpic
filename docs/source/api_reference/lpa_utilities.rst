Laser-plasma utilities
======================

FBPIC was designed specifically for simulations of **laser-wakefield acceleration** and **plasma-wakefield acceleration**. Therefore it provides helper functions and classes that allow to easily:

- Inject a laser pulse in the simulation.
- Initialize a relativistic electron beam with its space charge field.
- Add external fields (e.g. focusing fields), which are applied to the particles at each timestep but are not self-consistently evolved by the simulation.

.. toctree::

Laser utility
-------------

.. autofunction:: fbpic.lpa_utils.laser.add_laser

Beam utilities
--------------
		  
.. autofunction:: fbpic.lpa_utils.bunch.add_elec_bunch

.. autofunction:: fbpic.lpa_utils.bunch.add_elec_bunch_gaussian

.. autofunction:: fbpic.lpa_utils.bunch.add_elec_bunch_from_arrays

.. autofunction:: fbpic.lpa_utils.bunch.add_elec_bunch_file

External fields
---------------

.. autoclass:: fbpic.lpa_utils.external_fields.ExternalField
