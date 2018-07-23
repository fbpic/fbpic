The openPMD diagnostics
=======================

During a simulation, FBPIC outputs diagnostics of fields and
particles in the form of `openPMD files <https://github.com/openPMD>`__.

A diagnostic can be directly added to simulation (or removed) by
adding or removing a corresponding diagnostic object in the list of diagnostics `diags`, which is an attribute of the `Simulation` class.

Below are the different available diagnostics objects:

.. toctree::
   :maxdepth: 2

Regular diagnostics
-------------------

Field diagnostic
~~~~~~~~~~~~~~~~

.. autoclass:: fbpic.openpmd_diag.FieldDiagnostic

Particle diagnostic
~~~~~~~~~~~~~~~~~~~

.. autoclass:: fbpic.openpmd_diag.ParticleDiagnostic

Particle density diagnostic
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fbpic.openpmd_diag.ParticleDensityDiagnostic

Back-transformed diagnostics (boosted-frame simulations)
--------------------------------------------------------

These diagnostics recontruct and output the lab-frame quantities on the fly,
during a boosted-frame simulation.

Field diagnostic
~~~~~~~~~~~~~~~~

.. autoclass:: fbpic.openpmd_diag.BoostedFieldDiagnostic

Particle diagnostic
~~~~~~~~~~~~~~~~~~~

.. autoclass:: fbpic.openpmd_diag.BoostedParticleDiagnostic
