"""
Usage :
from the top-level directory of FBPIC run
$ python spin_planewave_test.py

-------------------------------------------------------------------------------

This file tests the creation of spin vectors upon ionization. Expected
behaviour is that the first ionized electron inherits its spin direction
from the parent ion, whereas all subsequent electrons have random
polarisation.

Author: Michael J. Quin, Kristjan Poder
Scientific supervision: Matteo Tamburini

"""
# -------
# Imports
# -------
import sys
sys.path.insert(0, '/Users/kpoder/python/dev_fbpic')
sys.path.insert(0, '/home/kpoder/devfbpic')
import numpy as np
from scipy.constants import c, e, m_e, m_p, epsilon_0, pi
from scipy.constants import physical_constants
anom = physical_constants['electron mag. mom. anomaly'][0]
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser


# ----------
# Parameters
# ----------

# Whether to use the GPU
use_cuda = True

# Order of the stencil for z derivatives in the Maxwell solver.
# Use -1 for infinite order, i.e. for exact dispersion relation in
# all direction (adviced for single-GPU/single-CPU simulation).
# Use a positive number (and multiple of 2) for a finite-order stencil
# (required for multi-GPU/multi-CPU with MPI). A large `n_order` leads
# to more overhead in MPI communications, but also to a more accurate
# dispersion relation for electromagnetic waves. (Typically,
# `n_order = 32` is a good trade-off.)
# See https://arxiv.org/abs/1611.05712 for more information.
n_order = -1 #32

# Driver laser parameters
lam = 0.8e-6                 # Wavelength (metres)
a0 = 10.                      # Laser amplitude
w0 = 10e-6                   # Laser waist
ctau = 30.e-15 / ( np.sqrt( 2.*np.log(2.) ) ) * c
z0 = -0e-6
zf = 00e-6


# Simulation of density
p_nz = 1            # Number of particles per cell along z
p_nr = 1            # Number of particles per cell along r
p_nt = 1            # Number of particles per cell along theta

# The simulation box
Nz = 10  # Number of gridpoints along z
Nr = 10   # Number of gridpoints along r
zmax = 1e-6             # Right end of the simulation box (meters)
zmin = -1e-6        # Left end of the simulation box (meters)
rmax = 1e-6        # 60*lam  # 30*lam  # Length of the box along r (meters)
Nm = 2               # Number of modes used

# Simulation timestep
dt = (zmax - zmin) / Nz / c   # Timestep (seconds)


def dens_func( z, r ) :
    """Returns relative density at position z and r"""
    # Allocate relative density
    n = np.ones_like(z)
    #n = np.where(r < rmax/Nr, n, 0.)
    #n = np.where(np.abs(z) < zmax/Nz, n, 0.)
    # n = np.where( z<ramp_start+ramp_length, (z-ramp_start)/ramp_length, n )
    return n


# NOTE!!! need to add the following code to the end of this function...
# particles/injection/continous_injection.py -> def generate_evenly_spaced: ...
# ... this forces fbpic to generate only one ion, so we can track the spin of
# each successive ionised electron

# # ---------- JUST ONE ION
# # only create an ion, which should have spin zero (or less than 0.5)
# if np.sqrt(sx_m ** 2 + sy_m ** 2 + sz_m ** 2) > 0.5:
#     # No particles are initialized ; the arrays are still created
#     Ntot = 0
#     return (Ntot, np.empty(0), np.empty(0), np.empty(0), np.empty(0),
#             np.empty(0), np.empty(0), np.empty(0), np.empty(0),
#             np.empty(0), np.empty(0), np.empty(0))
# else:
#     Ntot = 1
#     lam = 0.8e-6
#     x = 0.1 * lam * np.ones(1)
#     y = 0.1 * lam * np.ones(1)
#     z = 0. * np.ones(1)
#     r = np.sqrt(x ** 2 + y ** 2)
#     w = n * r * dtheta * dr * dz
#     ux = ux_m * np.ones(1)
#     uy = uy_m * np.ones(1)
#     uz = uz_m * np.ones(1)
#     inv_gamma = 1. / np.sqrt(1 + ux ** 2 + uy ** 2 + uz ** 2)
#     sx = sx_m * np.ones(1)
#     sy = sy_m * np.ones(1)
#     sz = sz_m * np.ones(1)
#     # Return the particle arrays
#     return (Ntot, x, y, z, ux, uy, uz, inv_gamma, w, sx, sy, sz)
# # ---------- JUST ONE ION


# ---------------------------
# Carrying out the simulation
# ---------------------------

def run_ionization_test_sim(show):
    # Initialize the simulation object
    sim = Simulation(Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
                     n_order=n_order, use_cuda=use_cuda,
                     boundaries={'z': 'open', 'r': 'reflective'})

    # Add hydrogen and make ionisable
    atoms_Cl = sim.add_new_species(q=0., m=34*m_p, n=1e24,
                                   dens_func=dens_func, p_nz=p_nz,
                                   p_nr=p_nr, p_nt=p_nt,
                                   continuous_injection=False)
    atoms_Cl.activate_spin_tracking(sz_m=1., anom=0.)

    elec_dict = {}
    for i in range(17):
        elec = sim.add_new_species(q=-e, m=m_e,
                                   continuous_injection=False)
        elec.activate_spin_tracking(anom=anom)
        elec_dict[i] = elec

    atoms_Cl.make_ionizable( 'Cl', target_species=elec_dict, level_start=0)

    print(('ion', atoms_Cl.Ntot, atoms_Cl.spin_tracker.sx.mean(),
           atoms_Cl.spin_tracker.sy.mean(), atoms_Cl.spin_tracker.sz.mean()))

    # Load initial fields
    # Add a laser to the fields of the simulation
    add_laser( sim, a0, w0, ctau, z0=z0, zf=zf, lambda0=lam)

    # Run the simulation

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()

    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('sx')
    ax.set_ylabel('sy')
    ax.set_zlabel('sz')

    elec_total = 5 * atoms_Cl.Ntot
    #label = ['First ionised electron', 'Second ionised electron',
    #         'Third ionised electron', 'Fourth ionised electron',
    #         'Fifth ionised electron']
    spin_data = np.zeros((elec_total, 6))

    # run until we have enough ionized electrons...
    elec_sum = 0
    step = 0
    while elec_sum < elec_total:
        elec_sum = 0
        step += 1
        # step and collect the data
        sim.step(1, show_progress=False)
        print(f'Step  Level  Ntot  N(sz==1)')
        for level, elec in elec_dict.items():
            elec_sum += elec.Ntot
            if level < 6:
                print(f'{step:4d}  {level+1:5d}  {elec.Ntot:4d} ' +
                      f'{np.sum(elec.spin_tracker.sz==1.):8d}')
        print('*'*40)


def test_spin_ionization_lab(show=False):
    """Function that is run by py.test, when doing `python setup.py test`"""
    run_ionization_test_sim(show)


if __name__ == '__main__' :
    test_spin_ionization_lab(show=False)
