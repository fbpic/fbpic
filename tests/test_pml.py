# Copyright 2019, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Soeren Jalas
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It tests the Perfectly-Matched-Layers, by initializing a laser with
a small waist and letting it diffract into the radial PML.
The test then checks that the profile of the laser (inside the
physical domain) is identical to the theoretical profile
(i.e. that the reflections are negligible).

Usage :
-------
In order to show the images of the laser, and manually check the
agreement between the simulation and the theory:
$ python tests/test_pml.py
(except when setting show to False in the parameters below)

In order to let Python check the agreement between the curve without
having to look at the plots
$ py.test -q tests/test_pml.py
or
$ python setup.py test
"""
import numpy as np
from scipy.constants import c
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser_pulse, \
    GaussianLaser, LaguerreGaussLaser

# Parameters
# ----------
# (See the documentation of the function propagate_pulse
# below for their definition)

show = True  # Whether to show the plots, and check them manually

use_cuda = True

# Simulation box
Nz = 360
zmin = -6.e-6
zmax = 6.e-6
Nr = 50
Lr = 4.e-6
Nm = 2
res_r  = Lr/Nr
n_order = -1
# Laser pulse
w0 = 1.5e-6
lambda0 = 0.8e-6
tau = 10.e-15
a0 = 1.
zf = 0.
# Propagation
L_prop = 40.e-6
dt = (zmax-zmin)*1./c/Nz
N_diag = 4 # Number of diagnostic points along the propagation
# Checking the results
rtol0 = 9.e-2 # Tolerance for mode 0
rtol1 = 5.e-2 # Tolerance for mode 1

def test_laser_periodic(show=False):
    """
    Function that is run by py.test, when doing `python setup.py test`
    Test the propagation of a laser in a periodic box.
    """
    propagate_pulse( boundaries='periodic', show=show )

def test_laser_galilean(show=False):
    """
    Function that is run by py.test, when doing `python setup.py test`
    Test the propagation of a laser with a galilean change of frame
    """
    propagate_pulse( boundaries='open', use_galilean=True,
                     v_comoving=0.999*c, show=show )

def propagate_pulse( boundaries, v_window=0, use_galilean=False,
                        v_comoving=None, show=False ):
    """
    Propagate the beam over a distance L_prop and compare with the
    theoretical profile

    Parameters
    ----------
    boundaries : string
        Type of boundary condition
        Either 'open' or 'periodic'

    v_window : float
        Speed of the moving window

    use_galilean: bool
        Whether to use a galilean frame that moves at the speed v_comoving

    v_comoving : float
        Velocity at which the currents are assumed to move

    show : bool
       Wether to show the fields, so that the user can manually check
       the agreement with the theory.
    """
    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, Lr, Nm, dt, r_boundary='open',
                      n_order=n_order, zmin=zmin, use_cuda=use_cuda,
                      boundaries=boundaries, v_comoving=v_comoving,
                      use_galilean=use_galilean )

    # Set the moving window object
    if v_window !=0:
        sim.set_moving_window( v=v_window )

    # Initialize the laser fields
    z0 = (zmax+zmin)/2

    # Simultaneously check the mode m=0 and m=1 within the same simulation
    # by initializing and laser in m=0 and m=1
    # - Mode 0: Build a radially-polarized pulse from 2 Laguerre-Gauss profiles
    profile0 = LaguerreGaussLaser( 0, 1, 0.5*a0, w0, tau, z0, zf=zf,
                lambda0=lambda0, theta_pol=0., theta0=0. ) \
             + LaguerreGaussLaser( 0, 1, 0.5*a0, w0, tau, z0, zf=zf,
                lambda0=lambda0, theta_pol=np.pi/2, theta0=np.pi/2 )
    # - Mode 1: Use a regular linearly-polarized pulse
    profile1 = GaussianLaser( a0=a0, waist=w0, tau=tau,
                lambda0=lambda0, z0=z0, zf=zf )

    # Add the profiles to the simulation
    add_laser_pulse( sim, profile0 )
    add_laser_pulse( sim, profile1 )

    # Calculate the number of steps to run between each diagnostic
    Ntot_step = int( round( L_prop/(c*dt) ) )
    N_step = int( round( Ntot_step/N_diag ) )

    # Loop over the iterations
    print('Running the simulation...')
    for it in range(N_diag):
        print( 'Diagnostic point %d/%d' %(it, N_diag) )
        # Advance the Maxwell equations
        sim.step( N_step )
        compare_fields( sim, profile0, field='Er', m=0,
                        rtol=rtol0, show=show, boundaries=boundaries )
        compare_fields( sim, profile1, field='Er', m=1,
                        rtol=rtol1, show=show, boundaries=boundaries )


def compare_fields( sim, profile, field, m, rtol, show, boundaries ):
    """
    Compare the Er field in `grid` to the theoretical fields given
    by `profile`.

    Parameters
    ----------
    sim: a Simulation object
        Contains the simulated fields
    profile: a LaserProfile object
        Has the method `E_field` that allows to calculate the theoretical profile
    field: string
        Indicate which field to compare: either `Er` or `Et`
    m: int
        Indicate which azimuthal mode to compare
    show: bool
        Whether to show the fields with matplotlib
    rtol: float
        Precision with which the fields are compared
    boundaries: string
        Allows to check whether the test is done in a periodic box
    """
    # Gather grids and remove guard and damp cells
    grid = sim.comm.gather_grid( sim.fld.interp[m] )

    # Only one MPI rank needs to check the results
    if sim.comm.rank != 0:
        return

    # Extract time
    t = sim.time
    # Select the field to plot
    E_sim = getattr( grid, field ).real
    if m != 0:
        # Factor 2 comes from definition of field in FBPIC
        E_sim *= 2

    # Calculate the theoretical profile
    # - Select the right component
    if field == 'Er':
        i_field = 0
    elif field == 'Et':
        i_field = 1
    else:
        raise ValueError("Unknown field: %s" %field)
    # - Calculate profile
    z, r = np.meshgrid( grid.z, grid.r, indexing='ij' )
    if boundaries=='periodic':
        Lz = grid.zmax - grid.zmin
        n_shift = np.floor( c*t/Lz )
        E_th = profile.E_field( r, 0, z + (n_shift+1)*Lz, t )[i_field] \
             + profile.E_field( r, 0, z + n_shift*Lz, t )[i_field]
    else:
        E_th = profile.E_field( r, 0, z, t )[i_field]

    # Calculate the difference:
    E_diff = E_sim - E_th

    if show:
        import matplotlib.pyplot as plt

        # Show the field also below the axis for a more realistic picture
        extent = 1.e6*np.array([grid.zmin, grid.zmax, -grid.rmax, grid.rmax])

        plt.subplot(131)
        plt.imshow( np.hstack( (E_sim[:,::-1], E_sim) ).T, aspect='auto',
                    interpolation='nearest', extent=extent, cmap='RdBu' )
        plt.xlabel('z')
        plt.ylabel('r')
        plt.colorbar()
        plt.title('Simulation')

        plt.subplot(132)
        plt.imshow( np.hstack( (E_th[:,::-1], E_th) ).T, aspect='auto',
                    interpolation='nearest', extent=extent, cmap='RdBu' )
        plt.xlabel('z')
        plt.ylabel('r')
        plt.colorbar()
        plt.title('Theoretical')

        plt.subplot(133)
        plt.imshow( np.hstack( (E_diff[:,::-1], E_diff) ).T, aspect='auto',
                    interpolation='nearest', extent=extent, cmap='RdBu' )
        plt.xlabel('z')
        plt.ylabel('r')
        plt.colorbar()
        plt.title('Difference')

        plt.tight_layout()
        plt.show()

    # Check that the fields agree to the required precision
    relative_error = abs( E_diff ).max() / abs( E_th ).max()
    print( 'Relative error on mode %d: %.4e' %(m,relative_error) )
    assert relative_error < rtol


if __name__ == '__main__' :

    # Run the testing function
    test_laser_periodic(show=show)
    test_laser_galilean(show=show)
