Overview of the code
====================

Introduction to the PIC cycle
--------------------------------

Like any electromagnetic PIC code, FBPIC simulates the
**self-consistent interaction** of **charged particles** and
**electromagnetic fields**.

The charged particles are represented by **macroparticles** (which lump
together several physical particles), while the fields are represented
on a **grid**. The time evolution of the system is simulated by taking **discrete
time steps**. At each timestep:

  - The values of E and B are gathered from the grid onto the macroparticles.
  - The particles are pushed in time.
  - The charge and current of the macroparticles are deposited onto the grid.
  - The fields E and B are pushed in time.

.. image:: images/pic_loop.png
	   
The distinctive features of FBPIC
-------------------------------------

Cylindrical grid with azimuthal decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the *standard* PIC algorithm, the fields are represented on a 3D
Cartesian grid. This is very generic, but also very computational
expensive. However, for physical situations 

.. image:: images/3d_vs_cylindrical.png

The above is only a schematic view. In fact, instead of using a **3D
Cartesian grid**, FBPIC uses **a set of 2D radial grids**. Each 2D radial grid represents **an azimuthal mode** (labeled by an integer :math:`m`):

- The grid for :math:`m=0` represents the fields that are **independent of**
  :math:`\theta`. (In idealized laser-wakefield acceleration, this is
  for instance the case for the plasma wakefield.)

- The grid for :math:`m=1` represents the fields that **vary
  proportionally to** :math:`\cos(\theta)` and :math:`\sin(\theta)`. (In
  idealized laser-wakefield acceleration, this is the case of the
  linearly-polarized laser field, when expressed in cylindrical coordinates,
  as :math:`E_r` and :math:`E_{\theta}`.)

- The grids for higher values of :math:`m`

Explain decomposition in modes

The laser stuff

.. image:: images/cylindrical_grid.png

.. note::
   In practice, FBPIC only uses 2 modes: allows to model fields that vary

.. note::

   Deposition in the first cell
  
Analytical integration in spectral space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mention artifacts that it avoids.

.. note::

   scales badly with number of radial cells

For more details on the algorithm, see the article

Centering in time and space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   

Guard cells and moving window
------------------------------------------

.. note::

  large number of guard cells:
  automatically added in the simulation, and not included in the
  output, so most invisible to the user. Yet, important to be aware

  Moving window is only moved every so often.
