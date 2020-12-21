.. fbpic documentation main file, created by
   sphinx-quickstart on Sun Sep 11 12:04:39 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FBPIC documentation
======================================================

FBPIC (Fourier-Bessel Particle-In-Cell) is a `Particle-In-Cell (PIC)
code <http://en.wikipedia.org/wiki/Particle-in-cell>`_ for
relativistic plasma
physics. It is especially well-suited for physical simulations of **laser-wakefield acceleration** and **plasma-wakefield acceleration**.

The distinctive feature of FBPIC, compared to *most* other PIC codes, is to use
a **spectral cylindrical representation.** This makes the code both **fast**
and **accurate**, for situations with **close-to-cylindrical
symmetry**. For a brief overview of the
algorithm, its advantages and limitations, see the section :doc:`overview/overview`.

In addition, FBPIC implements several **useful features for laser-plasma acceleration**, including:

    * Moving window
    * Calculation of **space-charge fields** at the beginning of the simulation
    * Intrinsic **mitigation of Numerical Cherenkov Radiation** (NCR) from relativistic bunches
    * **Field ionization** module (ADK model)
    * Support for **boosted-frame simulations** (see :doc:`advanced/boosted_frame`)


FBPIC can run on **multi-core CPU** (with multi-threading) or **GPU**. For large
simulations, running the code on GPU can be much faster than on CPU.

Contents of the documentation
-----------------------------

If you are new to FBPIC, we **strongly recommend** that you read the
section :doc:`overview/overview` first, so as to have a basic understanding of
what the code does.

You can then see the section :doc:`install/installation` and
:doc:`how_to_run`, to get started with using FBPIC. For more
information, the section :doc:`api_reference/api_reference` lists the main objects
that are accessible through FBPIC.

.. toctree::
   :maxdepth: 1

   overview/overview
   install/installation
   how_to_run
   api_reference/api_reference
   advanced/advanced

Contributing to FBPIC
---------------------

FBPIC is open-source, and the source code is hosted `here <http://github.com/fbpic/fbpic>`_, on
Github.

We welcome contributions to the code! If you wish to contribute,
please read `this page <https://github.com/fbpic/fbpic/blob/main/CONTRIBUTING.md>`_ .

Research & Attribution
----------------------

FBPIC was originally developed by Remi Lehe at `Berkeley Lab <http://www.lbl.gov/>`_,
and Manuel Kirchen at
`CFEL, Hamburg University <http://lux.cfel.de/>`_. The code also
benefitted from the contributions of Soeren Jalas (CFEL), Kevin Peters (CFEL),
Irene Dornmair (CFEL), Laurids Jeppe (CFEL), Igor Andriyash (Laboratoire d’Optique Appliquée),
Omri Seemann (Weizmann Institute), Daniel Seipt (University of Michigan)
and Samuel Yoffe (University of Strathclyde).

FBPIC's algorithms are documented in following scientific publications:

    * General description of FBPIC and it's algorithm (original paper):
      `R. Lehe et al., CPC, 2016 <http://www.sciencedirect.com/science/article/pii/S0010465516300224>`_ (`arXiv <https://arxiv.org/abs/1507.04790>`__)
    * Boosted-frame technique with Galilean algorithm:
      `M. Kirchen et al., PoP, 2016 <https://aip.scitation.org/doi/10.1063/1.4964770>`_ (`arXiv <https://arxiv.org/abs/1608.00215>`__) and
      `R. Lehe et al., PRE, 2016 <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.94.053305>`_ (`arXiv <https://arxiv.org/abs/1608.00227>`__)
    * Parallel finite-order solver for multi-CPU/GPU simulations:
      `S. Jalas et al., PoP, 2017 <https://aip.scitation.org/doi/abs/10.1063/1.4978569>`_ (`arXiv <https://arxiv.org/abs/1611.05712>`__)
    * Parallel finite-order boosted-frame simulations for multi-CPU/GPU simulations:
      `M. Kirchen et al., PRE, 2020 <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.102.013202>`_

If you use FBPIC for your research project: that's great! We are
very pleased that the code is useful to you!

If your project even leads to a scientific publication, please consider citing at least FBPIC's original paper.
If your project uses the more advanced algorithms, please consider citing the respective publications in addition.
