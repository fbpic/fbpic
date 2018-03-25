The Particles class
===================

A :any:`Simulation` object may contain several particle species. Each species is
represented by an instance of the :any:`Particles` class, and is stored in
the list ``Simulation.ptcl``.

.. note::

    It is **not** recommended to initialize a :any:`Particles` object directly.
    Instead, use the method :any:`Simulation.add_new_species`. This will
    append a new instance of :any:`Particles` to the list ``Simulation.ptcl``.
    This new instance will also be directly returned by the function, so
    that you can use its methods below.

.. py:class:: fbpic.particles.Particles

    An instance of `Particles` stores a set of macroparticles that have the
    same charge and mass. (Note that for ionizable species, ions in different
    states of ionization - i.e. different levels - are stored in
    the same `Particles` object.)

   .. automethod:: track
   .. automethod:: make_ionizable
