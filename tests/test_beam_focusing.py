# Copyright 2018, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It tests the routines that initialize a Gaussian beam, by making
sure that they produce the right RMS beam at focus.

More precisely, the simulations are run in the boosted frame, and
the beam is propagated up to the focal position, under its own space-charge
field. Typically, because of the space charge field acting on long
distances in the boosted frame, the intended RMS beam size is not reached.
However, by using the ballistic injection before the focal plane, the
beam does not feel these forces and reaches the right focus.
"""
# -------
# Imports
# -------
import numpy as np
from scipy.constants import c
import shutil
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.bunch import add_elec_bunch_gaussian
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.openpmd_diag import BackTransformedParticleDiagnostic
from openpmd_viewer import OpenPMDTimeSeries

# ----------
# Parameters
# ----------
use_cuda = True

# The simulation box
Nz = 100
zmax = 0.e-6
zmin = -20.e-6
Nr = 200
rmax = 20.e-6
Nm = 1
# The simulation timestep
dt = (zmax-zmin)/Nz/c
N_step = 101

# Boosted frame
gamma_boost = 15.
boost = BoostConverter( gamma_boost )

# The bunch
sigma_r = 1.e-6
sigma_z = 3.e-6
Q = 200.e-12
gamma0 = 100.
sigma_gamma = 0.
n_emit = 0.1e-6
z_focus = 2000.e-6
z0 = -10.e-6
N = 40000

# The diagnostics
diag_period = 5
Ntot_snapshot_lab = 21
dt_snapshot_lab = 2*(z_focus-z0)/c/20
v_comoving = c*np.sqrt(1.-1./gamma0**2)

def test_beam_focusing( show=False ):
    """
    Runs the simulation of a focusing charged beam, in a boosted-frame,
    with and without the injection through a plane.
    The value of the RMS radius at focus is automatically checked.
    """
    # Simulate beam focusing with injection through plane or not
    simulate_beam_focusing( None, 'direct' )
    simulate_beam_focusing( z_focus, 'through_plane' )

    # Analyze the results and show that the beam reaches
    # the right RMS radius at focus
    ts1 = OpenPMDTimeSeries('./direct/hdf5/')
    r1 = get_rms_radius( ts1 )
    ts2 = OpenPMDTimeSeries('./through_plane/hdf5/')
    r2 = get_rms_radius( ts2 )
    if show:
        import matplotlib.pyplot as plt
        plt.plot( 1.e3*c*ts1.t, 1.e6*r1 )
        plt.plot( 1.e3*c*ts2.t, 1.e6*r2 )
        plt.xlabel( 'z (mm)' )
        plt.ylabel( 'RMS radius (microns)' )
        plt.show()
    # Find the index of the output at z_focus
    i = np.argmin( abs( c*ts2.t - z_focus ) )
    # With injection through plane, we get the right RMS value at focus
    assert abs( r2[i] - sigma_r ) < 0.05e-6
    # Without injection through plane, the RMS value is significantly different
    assert abs( r1[i] - sigma_r ) > 0.5e-6

    # Clean up the data folders
    shutil.rmtree( 'direct' )
    shutil.rmtree( 'through_plane' )

def simulate_beam_focusing( z_injection_plane, write_dir ):
    """
    Simulate a focusing beam in the boosted frame

    Parameters
    ----------
    z_injection_plane: float or None
        when this is not None, the injection through a plane is
        activated.
    write_dir: string
        The directory where the boosted diagnostics are written.
    """
    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
        gamma_boost=gamma_boost, boundaries='open',
        use_cuda=use_cuda, v_comoving=v_comoving )
    # Note: no macroparticles get created because we do not pass
    # the density and number of particle per cell

    # Remove the plasma particles
    sim.ptcl = []

    # Initialize the bunch, along with its space charge
    add_elec_bunch_gaussian( sim, sigma_r, sigma_z, n_emit, gamma0,
        sigma_gamma, Q, N, tf=(z_focus-z0)/c, zf=z_focus, boost=boost,
        z_injection_plane=z_injection_plane )

    # Configure the moving window
    sim.set_moving_window( v=c )

    # Add a field diagnostic
    sim.diags = [
        BackTransformedParticleDiagnostic( zmin, zmax, c, dt_snapshot_lab,
            Ntot_snapshot_lab, gamma_boost, period=100, fldobject=sim.fld,
            species={'bunch':sim.ptcl[0]}, comm=sim.comm, write_dir=write_dir)
        ]

    # Run the simulation
    sim.step( N_step )

def get_rms_radius(ts):
    """
    Calculate the RMS radius at the different iterations of the timeseries.
    """
    r = []
    for iteration in ts.iterations:
        x, w = ts.get_particle( ['x', 'w'], iteration=iteration )
        r.append( np.sqrt( np.average( x**2, weights=w ) ) )
    return( np.array(r) )

if __name__ == '__main__':
    test_beam_focusing( show=True )
