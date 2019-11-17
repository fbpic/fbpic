# Copyright 2019, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It tests the few-cycle laser profile, by initializing a laser pulse
out of focus, letting propagate to the focus, and checking that the
fields at focus are the expected ones.

Usage :
-------
In order to show the profile out of focus, and then at focus:
$ python tests/test_fewcycle_laser.py
(except when setting show to False in the parameters below)

In order to let Python check the agreement automatically
$ py.test -q tests/test_fewcycle_laser.py
or
$ python setup.py test
"""
import numpy as np
from scipy.constants import c
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser_pulse, FewCycleLaser

# Parameters
# ----------

show = True  # Whether to show the plots, and check them manually

use_cuda = True

# The simulation box
Nz = 200         # Number of gridpoints along z
zmax = 5.e-6     # Right end of the simulation box (meters)
zmin = -5.e-6    # Left end of the simulation box (meters)
Nr = 200         # Number of gridpoints along r
rmax = 17.e-6    # Length of the box along r (meters)
Nm = 2           # Number of modes used
# The laser
a0 = 4.           # Laser amplitude
w0 = 1.5e-6       # Laser waist
tau_fwhm = 3.e-15 # Laser duration
z0 = 0.e-6        # Laser centroid
zfoc = 30e-6

# Checking the results
rtol = 5.e-2

def compare_fields( grid, t, profile, show ):
    """
    Compare the Er field in `grid` to the theoretical fields given
    by `profile`.

    Parameters
    ----------
    grid: an InterpolationGrid object
        Contains the simulated fields
    t: float
        Time at which to calculate the fields given by `profile`
    profile: a LaserProfile object
        Has the method `E_field` that allows to calculate the theoretical profile
    show: bool
        Whether to show the fields with matplotlib
    """
    # Select the field to plot
    Er = 2*getattr( grid, 'Er' ).real
    # Factor 2 comes from definition of field in FBPIC
    # Calculate the theoretical profile
    z, r = np.meshgrid( grid.z, grid.r, indexing='ij' )
    Er_th, _ = profile.E_field( r, 0, z + c*t, t )

    if show:
        import matplotlib.pyplot as plt

        # Show the field also below the axis for a more realistic picture
        extent = 1.e6*np.array([grid.zmin, grid.zmax, -grid.rmax, grid.rmax])

        # Plot the real part
        plt.subplot(121)
        plt.imshow( np.hstack( (Er[:,::-1],Er) ).T, aspect='auto',
                    interpolation='nearest', extent=extent, cmap='RdBu' )
        plt.xlabel('z')
        plt.ylabel('r')
        plt.colorbar()
        plt.title('Simulation')

        plt.subplot(122)
        plt.imshow( np.hstack( (Er_th[:,::-1],Er_th) ).T, aspect='auto',
                    interpolation='nearest', extent=extent, cmap='RdBu' )
        plt.xlabel('z')
        plt.ylabel('r')
        plt.colorbar()
        plt.title('Theoretical')

        plt.tight_layout()
        plt.show()

    # Check that the fields agree to the required precision
    assert np.allclose( Er, Er_th, atol=rtol*Er_th.max() )


def test_laser_periodic(show=False):
    """
    Function that is run by py.test, when doing `python setup.py test`
    Test the propagation of a laser in a periodic box.
    """
    # Propagate the pulse in a single step
    dt = zfoc*1./c

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
                      zmin=zmin, boundaries='periodic' )

    # Initialize the laser fields
    profile = FewCycleLaser(a0=a0, waist=w0, tau_fwhm=tau_fwhm, z0=0, zf=zfoc)
    add_laser_pulse( sim, profile )

    # Propagate the pulse
    compare_fields(sim.fld.interp[1], sim.time, profile, show)
    sim.step(1)
    compare_fields(sim.fld.interp[1], sim.time, profile, show)

if __name__ == '__main__' :

    # Run the testing function
    test_laser_periodic(show=show)
