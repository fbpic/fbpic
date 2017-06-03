.. fbpic documentation master file, created by
   sphinx-quickstart on Sun Sep 11 12:04:39 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FBPIC documentation
======================================================

FBPIC (Fourier-Bessel Particle-In-Cell) is a `Particle-In-Cell (PIC)
code <http://en.wikipedia.org/wiki/Particle-in-cell)>`_ for
relativistic plasma
physics. It is especially well-suited for physical simulations of **laser-wakefield acceleration** and **plasma-wakefield acceleration**.

The distinctive feature of FBPIC, compared to *most* other PIC codes, is to use
a **spectral cylindrical representation.** This makes the code both **fast**
and **accurate**, for situations with **close-to-cylindrical
symmetry**. Partly because of this representation, the code is
**dispersion-free in all directions**, and avoids some **major interpolation
artifacts** that other PIC codes may have. For a brief overview of the
algorithm, its advantages and limitations, see the section :doc:`overview`.

FBPIC can run either on CPU or GPU. For large
simulations, running the code on GPU can be up to **40 times faster**
than on CPU.

.. note::

   This is an Alpha release, and some parts of the code are not yet fully
   functional. In particular, for **multi**-CPU and **multi**-GPU runs
   (using MPI), the code is not yet production-ready.



Contents of the documentation
-------------------------------------

If you are new to FBPIC, we **strongly recommend** that you read the
section :doc:`overview` first, so as to have a basic understanding of
what the code does.

You can then see the section :doc:`install/installation` and
:doc:`how_to_run`, to get started with using FBPIC. For more
information, the section :doc:`api_reference/api_reference` lists the main objects
that are accessible through FBPIC.

.. toctree::
   :maxdepth: 1

   overview
   install/installation
   how_to_run
   api_reference/api_reference.rst
   advanced/advanced.rst

Contributing to FBPIC
---------------------

FBPIC is open-source, and the source code is hosted `here <http://github.com/fbpic/fbpic>`_, on
Github.

We welcome contributions to the code! If you wish to contribute,
please read `this page <https://github.com/fbpic/fbpic/blob/master/CONTRIBUTING.md>`_ .

Attribution
---------------

FBPIC was originally developed by Remi Lehe at `Berkeley Lab <http://www.lbl.gov/>`_,
and Manuel Kirchen at
`CFEL, Hamburg University <http://lux.cfel.de/>`_. The code also
benefitted from the contributions of Soeren Jalas, Kevin Peters and
Irene Dornmair (CFEL).

If you use FBPIC for your research project: that's great! We are
very pleased that the code is useful to you! If your project even leads
to a scientific publication, please consider citing FBPIC's original
paper, which can be found `on this page <http://www.sciencedirect.com/science/article/pii/S0010465516300224>`_
(see `this link <https://arxiv.org/abs/1507.04790>`_ for the arxiv version).
