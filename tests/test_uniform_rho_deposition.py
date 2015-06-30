"""
This file tests the charge density deposition by initializing a uniform
plasma and shifting the electrons by a small amout in r.

If the deposition the charge density should still be 0 everywhere,
except at the edges. If there is a problem with the charge deposition,
one will see a non-zero charge density on the axis.

Usage :
from the top-level directory of FBPIC run
$ python tests/test_uniform_rho_deposition.py
"""

from scipy.constants import c
from fbpic.main import Simulation
import matplotlib.pyplot as plt

if __name__ == '__main__' :

    # Dimensions of the box
    Nz = 250
    zmax = 20.e-6
    Nr = 50
    rmax= 20.e-6
    Nm = 2
    # Particles
    p_nr = 2
    p_nz = 2
    p_nt = 4
    p_rmax = 10.e-6
    n = 9.e24
    # Shift of the electrons, as a fraction of dr
    frac_shift = 0.01

    
    # Initialize the different structures
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, zmax/Nz/c,
        0, zmax, 0, p_rmax, p_nz, p_nr, p_nt, n, initialize_ions=True )

    # Shift the electrons
    dr = rmax/Nr
    sim.ptcl[0].x += frac_shift * dr
    
    # Deposit the charge
    sim.fld.erase('rho')
    for species in sim.ptcl :
        species.deposit( sim.fld.interp, 'rho')
    sim.fld.divide_by_volume('rho')
    
    # Show the results
    sim.fld.interp[0].show('rho')
    plt.show()
