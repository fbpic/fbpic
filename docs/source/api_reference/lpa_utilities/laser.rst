Laser initialization
====================

There are two functions that allow to introduce a laser pulse in an FBPIC simulation:

- The generic function ``add_laser_pulse``, which can introduce an arbitrary laser profile.
- The compact function ``add_laser``, which can only introduce a Gaussian profile.

The generic function is presented first here. This page also explains how
to define laser profiles that differ from a standard Gaussian profile.
Finally the compact function is described at the end of this page.

Both of the above functions support two types of methods, in order to introduce
the laser in the simulation box:

- The ``direct`` method:
    In this case, the entire laser pulse is initialized in the simulation box,
    immediately when the function is called.

    .. note::

        When using this method, FBPIC will always initialize the laser field
        on the grid in a **self-consistent manner** (e.g. such that the equations
        :math:`\boldsymbol{\nabla} \cdot \boldsymbol{E} = 0`
        and :math:`\boldsymbol{\nabla} \cdot \boldsymbol{B} = 0`
        are automatically satisfied, for the added laser field).

- The ``antenna`` method:
    In this case, the laser pulse is progressively emitted by a virtual antenna
    during the simulation. This can be advantageous when the simulation box is
    too small to contain the initial laser pulse entirely.

    .. note::

        Due to the details of its implementation, the laser antenna always
        emits **two laser pulses** that propagate in opposite directions (one
        on each side of the plane of the antenna). One of these two pulses
        is usually unwanted. However, when using a moving window that follows
        the only pulse of interest, the other (unwanted) pulse rapidly exits
        the simulation box.

Generic function for arbitrary laser profile
---------------------------------------------

.. autofunction:: fbpic.lpa_utils.laser.add_laser_pulse

Laser profiles
---------------

Mention that laser profiles can be summed.
Mention how to create new laser profiles.

Gaussian profile
****************

.. autoclass:: fbpic.lpa_utils.laser.GaussianLaser

Laguerre-Gauss profile
**********************

.. autoclass:: fbpic.lpa_utils.laser.LaguerreGaussLaser

Creating your own laser profile
*******************************


Compact function for a Gaussian pulse
-------------------------------------

.. autofunction:: fbpic.lpa_utils.laser.add_laser
