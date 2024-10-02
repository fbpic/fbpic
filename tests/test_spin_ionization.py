"""
Usage :
from the top-level directory of FBPIC run
$ python test_spin_ionization.py

-------------------------------------------------------------------------------

This file tests the creation of spin vectors upon ionization. Expected
behaviour is that the first ionized electron inherits its spin direction
from the parent ion, whereas all subsequent electrons have random
polarisation.

Author: Michael J. Quin, Kristjan Poder
Scientific supervision: Matteo Tamburini

"""
import numpy as np
from scipy.constants import c, e, m_e, m_p
from scipy.constants import physical_constants
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser
from fbpic.utils.cuda import cuda_installed

anom = physical_constants['electron mag. mom. anomaly'][0]


use_cuda = cuda_installed
n_order = -1

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


def run_ionization_test_sim():
    # Initialize the simulation object
    sim = Simulation(Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
                     n_order=n_order, use_cuda=use_cuda,
                     boundaries={'z': 'open', 'r': 'reflective'})

    # Add hydrogen and make ionisable
    atoms_Cl = sim.add_new_species(q=0., m=34*m_p, n=1e24, p_nz=p_nz,
                                   p_nr=p_nr, p_nt=p_nt,
                                   continuous_injection=False)

    elec_dict = {}
    for i in range(17):
        elec = sim.add_new_species(q=-e, m=m_e,
                                   continuous_injection=False)
        elec.activate_spin_tracking(anom=anom)
        elec_dict[i] = elec

    atoms_Cl.activate_spin_tracking(sz_m=1., anom=0.)
    atoms_Cl.make_ionizable( 'Cl', target_species=elec_dict, level_start=0)

    # Check that the spin vectors are set up properly for the parent ion
    assert atoms_Cl.spin_tracker.sx.mean() < 0.01
    assert atoms_Cl.spin_tracker.sy.mean() < 0.01
    assert atoms_Cl.spin_tracker.sz.mean() > 0.99

    # Add a laser to the fields of the simulation
    add_laser( sim, a0, w0, ctau, z0=z0, zf=zf, lambda0=lam)

    # Run the simulation
    n_elec_to_ionize = 5 * atoms_Cl.Ntot
    n_ion_elec_from_prev_step = dict(list(enumerate([0]*17)))
    for step in range(50):
        n_elec_ionized = 0

        # step and collect the data
        sim.step(1, show_progress=False)
        print(f'Step {step+1} ')
        for level, elec in elec_dict.items():
            n_elec_ionized += elec.Ntot
            strack = elec.spin_tracker

            if level < 6:
                # Check that the newly ionized electrons have expected spin:
                # <s_z> = 1 for first level and less for others
                n_prev_elec = n_ion_elec_from_prev_step[level]
                n_new_elec = elec.Ntot - n_prev_elec
                if n_new_elec > 0:
                    # Assertion only done when running on GPU!
                    if not use_cuda:
                        if level == 0:
                            assert strack.sz[n_prev_elec:].mean() == 1.
                        else:
                            assert strack.sz[n_prev_elec:].mean() != 1.
                        # Print some stats for clarity
                    print(f'\tLevel {level+1} had {elec.Ntot-n_prev_elec} new electrons '
                          f'with <s_z>={strack.sz[n_prev_elec:].mean()}')

            # Store the amount of new electrons this step
            n_ion_elec_from_prev_step[level] = elec.Ntot

        # Run until we have ionized an electrons from all outer shell
        if n_elec_ionized > n_elec_to_ionize:
            break


def test_spin_ionization_lab():
    """Function that is run by pytest"""
    run_ionization_test_sim()


if __name__ == '__main__':
    test_spin_ionization_lab()
