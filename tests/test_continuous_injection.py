# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It verifies that the continuous injection of plasma works as intended by:
- Initializing a plasma partially inside the box, and partially through
continuous plasma injection.
- Checking that the deposited plasma density is the intended one, and in
particular that there is no discontinuity between the plasma that was
initially in the box, and the plasma that has been continuously injected.

This is done in the lab frame and in the boosted frame

Usage :
from the top-level directory of FBPIC run
$ python tests/test_continuous_injection.py
"""
from scipy.constants import c, e, m_e
from fbpic.main import Simulation
import numpy as np

# Parameters
# ----------
use_cuda = True

show = True     # Whether to show the results to the user, or to
                # automatically determine if they are correct.

# Dimensions of the box
Nz = 100
Nr = 50
Nm = 2
zmin = -10.e-6
zmax = 5.e-6
rmax = 20.e-6
dz = (zmax-zmin)/Nz
# Particles
p_nr = 2
p_nz = 2
p_nt = 4
p_rmax = rmax
p_zmax = 1e6
n = 1.e24
ramp0 = 7.e-6 # linear density upramp in z
smooth_r = rmax*0.5 # smooth cos**2 decrease of density in r (towards rmax)

# -------------
# Test function
# -------------

def test_labframe_with_preexisting_plasma(show=False):
    "Run test in lab frame with some plasma at t=0"
    p_zmin = 0.e-6 # Chosen so there is some plasma inside the box at t=0
    run_continuous_injection(None, ramp0, p_zmin, p_zmax, show)

def test_boosted_with_preexisting_plasma(show=False):
    "Run test in boosted frame with some plasma at t=0"
    gamma_boost = 15.
    # The ramp is made longer so as to still resolve it in the boosted frame
    ramp = 2*gamma_boost*ramp0
    p_zmin = 0.e-6 # Chosen so there is some plasma inside the box at t=0
    run_continuous_injection(gamma_boost, ramp, p_zmin, p_zmax, show)

def test_labframe_without_preexisting_plasma(show=False):
    "Run test in lab frame without some plasma at t=0"
    p_zmin = zmax + 2*dz # Chosen outside the physical box
    run_continuous_injection(None, ramp0, p_zmin, p_zmax, show)

def run_continuous_injection( gamma_boost, ramp, p_zmin, p_zmax,
                              show, N_check=2 ):
    # Chose the time step
    dt = (zmax-zmin)/Nz/c

    def dens_func( z, r ):
        dens = np.ones_like( z )
        # Make the density smooth at rmax
        dens = np.where( r > rmax-smooth_r,
            np.cos(0.5*np.pi*(r-smooth_r)/smooth_r)**2, dens)
        # Make the density 0 below p_zmin
        dens = np.where( z < p_zmin, 0., dens )
        # Make a linear ramp
        dens = np.where( (z>=p_zmin) & (z<p_zmin+ramp),
                         (z-p_zmin)/ramp*dens, dens )
        return( dens )

    # Initialize the different structures
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
        p_zmin, p_zmax, 0, p_rmax, p_nz, p_nr, p_nt, 0.5*n,
        dens_func=dens_func, initialize_ions=False, zmin=zmin,
        use_cuda=use_cuda, gamma_boost=gamma_boost, boundaries='open' )

    # Add another species with a different number of particles per cell
    # and with a finite temperature
    uth = 0.0001
    sim.add_new_species( -e, m_e, 0.5*n, dens_func,
                            2*p_nz, 2*p_nr, 2*p_nt,
                            p_zmin, p_zmax, 0, p_rmax,
                            ux_th=uth, uy_th=uth, uz_th=uth )

    # Set the moving window, which handles the continuous injection
    # The moving window has an arbitrary velocity (0.7*c) so as to check
    # that the injection is correct in this case also
    sim.set_moving_window( v=c )

    # Check that the density is correct after different timesteps
    N_step = int( Nz/N_check/2 )
    for i in range( N_check ):
        sim.step( N_step, move_momenta=False )
        check_density( sim, gamma_boost, dens_func, show )

def check_density( sim, gamma_boost, dens_func, show ):

    # Get the grid without the guard cells (mode 0)
    gathered_grid = sim.comm.gather_grid( sim.fld.interp[0] )

    # Calculate the expected density
    z, r = np.meshgrid( gathered_grid.z, gathered_grid.r, indexing='ij' )
    if gamma_boost is None:
        rho_expected = - n * e * dens_func( z, r )
    else:
        v_boost = np.sqrt( 1. - 1./gamma_boost**2 ) * c
        shift = v_boost * sim.time
        rho_expected = - gamma_boost * n * e * dens_func( z + shift, r )

    # Show the results
    if show:
        import matplotlib.pyplot as plt
        extent = 1.e6*np.array([ gathered_grid.zmin, gathered_grid.zmax,
                   gathered_grid.rmin, gathered_grid.rmax ])

        plt.figure(1, figsize=(6,12) )
        plt.clf()

        plt.subplot(311)
        plt.title('Deposited density')
        plt.imshow( gathered_grid.rho.real.T[::-1],
            aspect='auto', extent=extent )
        plt.colorbar()

        plt.subplot(312)
        plt.title('Expected density')
        plt.imshow( rho_expected.T[::-1],
            aspect='auto', extent=extent )
        plt.colorbar()

        plt.subplot(313)
        plt.title('Difference')
        plt.imshow( rho_expected.T[::-1] - gathered_grid.rho.real.T[::-1],
            aspect='auto' )
        plt.colorbar()

        plt.show()

    else:
        # Automatically check that the density is
        # within 1 % of the expected value
        assert np.allclose( gathered_grid.rho.real, rho_expected,
                            atol=1.e-2*abs(rho_expected).max() )
        assert np.allclose( gathered_grid.rho.imag, 0.,
                            atol=1.e-2*abs(rho_expected).max() )


if __name__ == '__main__' :
    test_labframe_with_preexisting_plasma(show)
    test_boosted_with_preexisting_plasma(show)
    test_labframe_without_preexisting_plasma(show)
