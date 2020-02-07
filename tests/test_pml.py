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

The test script is run in parallel with 2 MPI ranks.

Usage :
-------
This file is meant to be run from the top directory of fbpic,
by any of the following commands:

- In order to show the images of the laser, and manually check the
agreement between the simulation and the theory:
$ python tests/test_pml.py
(except when setting show to False in the parameters below)

- In order to let Python check the agreement between the curve without
having to look at the plots:
$ py.test -q tests/test_example_docs_scripts.py
"""
import os, re
import shutil
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c
from fbpic.lpa_utils.laser import GaussianLaser, LaguerreGaussLaser

# Folders
temporary_dir = './tests/tmp_test_dir'
original_script = './tests/unautomated/test_pml.py'

# Checking the results
show = True  # Whether to show the plots, and check them manually
rtol0 = 9.e-2 # Tolerance for mode 0
rtol1 = 5.e-2 # Tolerance for mode 1

# Laser pulse parameters (for comparison with theory)
w0 = 1.5e-6
lambda0 = 0.8e-6
tau = 10.e-15
a0 = 1.
zf = 0.
z0 = 0.

def test_laser_periodic(show=False):
    """
    Function that is run by py.test, when doing `python setup.py test`
    Test the propagation of a laser in a periodic box.
    """
    run_parallel( show=show, z_boundary='periodic', use_galilean=False )

def test_laser_galilean(show=False):
    """
    Function that is run by py.test, when doing `python setup.py test`
    Test the propagation of a laser with a galilean change of frame
    """
    run_parallel( show=show, z_boundary='open', use_galilean=True )


def run_parallel( show, z_boundary, use_galilean ):
    """Copy the script `script_file` from the `unautomated directory`
    and run the simulation in parallel with 2 MPI ranks

    Then compare the results with theory

    Parameters:
    -----------
    show: bool
        Whether to show the plots of the laser
    z_boundary: string
        Either 'periodic' or 'open'
        The boundary condition in the longitudinal direction
    use_galilean: bool
        Whether to use the galilean algorithm
    """
    # Set the options for the `Simulation` object
    options = "boundaries = {'z':'%s', 'r':'open'}" %z_boundary
    if use_galilean:
        options += ", use_galilean=True, v_comoving=0.999*c"

    # Create a temporary directory for the simulation
    if os.path.exists( temporary_dir ):
        shutil.rmtree( temporary_dir )
    os.mkdir( temporary_dir )
    # Modify the original script and copy it to the temporary directory
    with open(original_script) as f:
        script = f.read()
    script = re.sub( r"Simulation\(",
                     r"Simulation(%s,\n" %options, script )
    with open( os.path.join(temporary_dir, 'fbpic_script.py'), 'w' ) as f:
        f.write( script )

    # Launch the script from the OS, in parallel (carry out half the iterations)
    response = os.system(
        'cd %s; mpirun -np 2 python fbpic_script.py' %temporary_dir )
    assert response==0

    # Launch the script with restart (carry out second half of the iterations)
    script = re.sub( r"restart = False",
                     r"restart = True", script )
    with open( os.path.join(temporary_dir, 'fbpic_script.py'), 'w' ) as f:
        f.write( script )
    response = os.system(
        'cd %s; mpirun -np 2 python fbpic_script.py' %temporary_dir )
    assert response==0

    # Check the validity
    check_theory_pml( show, z_boundary )

    # Suppress the temporary directory
    shutil.rmtree( temporary_dir )


def check_theory_pml(show, boundaries):
    """
    Check that the transverse E and B field are close to the high-gamma
    theory for a gaussian bunch
    """
    ts = OpenPMDTimeSeries( os.path.join(temporary_dir, 'diags/hdf5/') )
    for iteration in ts.iterations:
        compare_E( ts, 'x', m=0, iteration=iteration,
                        rtol=rtol0, show=show, boundaries=boundaries )
        compare_E( ts, 'x', m=1, iteration=iteration,
                        rtol=rtol1, show=show, boundaries=boundaries )


def compare_E( ts, coord, m, iteration, rtol, show, boundaries ):
    """
    Compare the Er field in `grid` to the theoretical fields given
    by `profile`.

    Parameters
    ----------
    ts: an OpenPMDTimeSeries object
        Contains the simulated fields
    coord: string
        Indicate which field to compare: either `x` or `y`
    m: int
        Indicate which azimuthal mode to compare
    show: bool
        Whether to show the plots of the laser
    rtol: float
        Precision with which the fields are compared
    show: bool
        Whether to show the fields with matplotlib
    boundaries: string
        Allows to check whether the test is done in a periodic box
    """
    # Extract simulation data
    E_sim, info = ts.get_field('E', coord, m=m, iteration=iteration)
    t = ts.current_t

    # Calculate the theoretical profile
    # - Select the right component
    if coord == 'x':
        i_field = 0
    elif coord == 'y':
        i_field = 1
    else:
        raise ValueError("Unknown field: %s" %coord)
    # - Build the theoretical laser profile
    #  (same as the one that was used in order to initialize the simulation)
    if m == 0:
        # Build a radially-polarized pulse from 2 Laguerre-Gauss profiles
        profile = LaguerreGaussLaser( 0, 1, 0.5*a0, w0, tau, z0, zf=zf,
            lambda0=lambda0, theta_pol=0., theta0=0. ) \
                + LaguerreGaussLaser( 0, 1, 0.5*a0, w0, tau, z0, zf=zf,
            lambda0=lambda0, theta_pol=np.pi/2, theta0=np.pi/2 )
    elif m == 1:
        # Use a regular linearly-polarized pulse
        profile = GaussianLaser( a0=a0, waist=w0, tau=tau,
            lambda0=lambda0, z0=z0, zf=zf )
    # - Calculate profile
    r, z = np.meshgrid( info.r, info.z, indexing='ij' )
    if boundaries=='periodic':
        Lz = info.zmax - info.zmin + info.dz # Full length of the box
        n_shift = np.floor( c*t/Lz )
        E_th = profile.E_field( r, 0, z + (n_shift+1)*Lz, t )[i_field] \
             + profile.E_field( r, 0, z + n_shift*Lz, t )[i_field]
    else:
        E_th = profile.E_field( r, 0, z, t )[i_field]

    # Calculate the difference:
    E_diff = E_sim - E_th

    if show:
        import matplotlib.pyplot as plt

        plt.subplot(311)
        plt.imshow( E_sim, aspect='auto', origin='lower',
                    interpolation='nearest',
                    extent=info.imshow_extent, cmap='RdBu' )
        plt.xlabel('z')
        plt.ylabel('r')
        plt.colorbar()
        plt.title('Simulation')

        plt.subplot(312)
        plt.imshow( E_th, aspect='auto', origin='lower',
                    interpolation='nearest',
                    extent=info.imshow_extent, cmap='RdBu' )
        plt.xlabel('z')
        plt.ylabel('r')
        plt.colorbar()
        plt.title('Theoretical')

        plt.subplot(313)
        plt.imshow( E_diff, aspect='auto', origin='lower',
                    interpolation='nearest',
                    extent=info.imshow_extent, cmap='RdBu' )
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
