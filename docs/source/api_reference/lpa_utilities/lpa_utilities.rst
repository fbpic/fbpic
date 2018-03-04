Laser-plasma utilities
======================

FBPIC was designed specifically for simulations of **laser-wakefield acceleration** and **plasma-wakefield acceleration**. Therefore it provides helper functions and classes that allow to easily:

- Inject a laser pulse in the simulation.
- Initialize a relativistic electron beam with its space charge field.
- Add external fields (e.g. focusing fields), which are applied to the particles at each timestep but are not self-consistently evolved by the simulation.

See the sections below for the corresponding documentation:

.. toctree::
	:maxdepth: 1

	laser
	beam
	external_fields
