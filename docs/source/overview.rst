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

Cylindrical grid (with azimuthal decomposition)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explain decomposition in modes

The laser stuff

.. image:: images/cylindrical_grid.png

FBPIC only uses 2 modes: allows to model fields that vary

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
