# Copyright 2018, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It makes sure that the current and charge deposition give the same results on
CPU and GPU. This is only relevant on a GPU plateform. (The test is skipped
on a CPU plateform.)

Usage :
from the top-level directory of FBPIC run
$ python tests/test_cpu_gpu_deposition.py
"""
# -------
# Imports
# -------
import numpy as np
import os, shutil
from scipy.constants import c
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.bunch import add_elec_bunch_gaussian
from fbpic.openpmd_diag import FieldDiagnostic
from fbpic.utils.cuda import cuda_installed
from openpmd_viewer import OpenPMDTimeSeries

# Parameters
# ----------
# The simulation box
Nz = 100         # Number of gridpoints along z
zmax = 30.e-6    # Right end of the simulation box (meters)
zmin = -10.e-6   # Left end of the simulation box (meters)
Nr = 50          # Number of gridpoints along r
rmax = 20.e-6    # Length of the box along r (meters)
Nm = 2           # Number of modes used

# The simulation timestep
dt = (zmax-zmin)/Nz/c   # Timestep (seconds)
N_step = 3     # Number of iterations to perform

# The particles: gaussian bunch
gamma0 = 10
sig_r = 20.e-6
sig_z = 10.e-6
sig_gamma = 0.
n_emit = 10.e-6
Q = 10.e-12
N = 2000

# The diagnostics and the checkpoints
diag_period = 1

# Test function
# -------------
def run_cpu_gpu_deposition(show=False, particle_shape='cubic'):

    # Skip this test if cuda is not installed
    if not cuda_installed:
        return

    # Perform deposition for a few timesteps, with both the CPU and GPU
    for hardware in ['cpu', 'gpu']:
        if hardware=='cpu':
            use_cuda = False
        elif hardware=='gpu':
            use_cuda = True

        # Initialize the simulation object
        sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
            zmin=zmin, use_cuda=use_cuda, particle_shape=particle_shape )
        sim.ptcl = []

        # Add an electron bunch (set the random seed first)
        np.random.seed(0)
        add_elec_bunch_gaussian(
            sim, sig_r, sig_z, n_emit, gamma0, sig_gamma, Q, N )

        # Add a field diagnostic
        sim.diags = [ FieldDiagnostic( diag_period, sim.fld,
            fieldtypes=['rho', 'J'], comm=sim.comm,
            write_dir=os.path.join('tests',hardware) )]

        ### Run the simulation
        sim.step( N_step )

    # Check that the results are identical
    ts_cpu = OpenPMDTimeSeries('tests/cpu/hdf5')
    ts_gpu = OpenPMDTimeSeries('tests/gpu/hdf5')
    for iteration in ts_cpu.iterations:
        for field, coord in [('rho',''), ('J','x'), ('J','z')]:
            # Jy is not tested because it is zero
            print('Testing %s at iteration %d' %(field+coord, iteration))
            F_cpu, info = ts_cpu.get_field( field, coord, iteration=iteration )
            F_gpu, info = ts_gpu.get_field( field, coord, iteration=iteration )
            tolerance = 1.e-13*( abs(F_cpu).max() + abs(F_gpu).max() )
            if not show:
                assert np.allclose( F_cpu, F_gpu, atol=tolerance )
            else:
                if not np.allclose( F_cpu, F_gpu, atol=tolerance ):
                    plot_difference( field, coord, iteration,
                                     F_cpu, F_gpu, info)

    # Remove the files used
    shutil.rmtree('tests/cpu')
    shutil.rmtree('tests/gpu')

def test_linear_deposition(show=False):
    """Test run by py.test"""
    run_cpu_gpu_deposition(show=show, particle_shape='linear')

def test_cubic_deposition(show=False):
    """Test run by py.test"""
    run_cpu_gpu_deposition(show=show, particle_shape='cubic')

def plot_difference( field, coord, iteration, F_cpu, F_gpu, info ):
    """
    Plots the simulation results on CPU and GPU
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.suptitle( field + coord )
    plt.subplot(311)
    plt.imshow( F_gpu, aspect='auto',
                origin='lower', extent=1.e6*info.imshow_extent )
    plt.colorbar()
    plt.title('GPU')
    plt.subplot(312)
    plt.imshow( F_cpu, aspect='auto',
                origin='lower', extent=1.e6*info.imshow_extent )
    plt.colorbar()
    plt.title('CPU')
    plt.subplot(313)
    plt.imshow( F_cpu - F_gpu, aspect='auto',
                origin='lower', extent=1.e6*info.imshow_extent )
    plt.colorbar()
    plt.title('Difference')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__' :

    test_linear_deposition(show=True)
    test_cubic_deposition(show=True)
