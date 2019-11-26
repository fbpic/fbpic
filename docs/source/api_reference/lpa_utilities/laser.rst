Laser initialization
====================

There are two functions that allow to introduce a laser pulse in an FBPIC simulation:

- The generic function ``add_laser_pulse``, which can introduce an arbitrary laser profile.
- The compact function ``add_laser``, which can only introduce a Gaussian profile.

The generic function is presented first here. This page also explains how
to define laser profiles in FBPIC. Finally the compact function is described
at the end of this page.

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

This section lists the available laser profiles in FBPIC. Note that you can
also combine these laser profiles (by summing them), or even create your
own custom laser profiles.

.. toctree::
    :maxdepth: 1

    laser_profiles/gaussian
    laser_profiles/laguerre
    laser_profiles/donut_laguerre
    laser_profiles/flattened
    laser_profiles/few_cycle

Combining (summing) laser profiles
**********************************

Laser profiles can be combined by **summing them together**, so as to create
new laser profiles.

For instance, one might want to use a **circularly-polarized
laser pulse**, but FBPIC only provides linearly-polarized laser profiles.
However, it turns out that a circularly-polarized pulse can be decomposed
as the sum of two linearly-polarized pulses, with orthogonal polarization
and a :math:`\pi/2` phase shift. In FBPIC, this can be done in the following
way:

::

    import math
    from fbpic.lpa_utils.laser import add_laser_pulse, GaussianLaser
    w0 = 5.e-6; z0 = 3.e-6; tau = 30.e-15; a0 = 1

    linear_profile1 = GaussianLaser( a0/math.sqrt(2), waist, tau, z0,
                                theta_pol=0., cep_phase=0. )
    linear_profile2 = GaussianLaser( a0/math.sqrt(2), waist, tau, z0,
                                theta_pol=math.pi/2, cep_phase=math.pi/2 )

    circular_profile = linear_profile1 + linear_profile2
    add_laser_pulse( sim, circular_profile )

Creating your own custom laser profile
**************************************

If you have some experience with Python programming, you can also define
your own laser profile, inside your input script. To do so, you can have a
look at how the existing laser profiles that are implemented `here
<https://github.com/fbpic/fbpic/blob/dev/fbpic/lpa_utils/laser/laser_profiles.py>`_.


Compact function for a Gaussian pulse
-------------------------------------

Using this function is equivalent to using the more generic ``add_laser_pulse``
function, with a ``GaussianLaser`` profile.

.. autofunction:: fbpic.lpa_utils.laser.add_laser
