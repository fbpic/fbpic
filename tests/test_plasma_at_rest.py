"""
This file tests the field evolution inside a plasma column at rest.

Usage :
from the top-level directory of FBPIC run
$ python tests/test_plasma_at_rest.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, e
# Import the relevant structures in FBPIC                                       
from fbpic.main import Simulation

scale_factor = 1

# The simulation box                                                            
Nz = 50 * scale_factor         # Number of gridpoints along z
zmax = 10.e-6    # Length of the box along z (meters)
zmin = 0.e-6
Nr = 50         # Number of gridpoints along r 
rmax = 10.e-6   # Length of the box along r (meters)
Nm = 2           # Number of modes used             
n_order = -1     # The order of the stencil in z

# The simulation timestep                                              
dt = zmax/Nz/c  # Timestep (seconds)                                           
N_step = 50 * scale_factor     # Number of iterations to perform
N_show = N_step     # Number of timestep between every plot

# The particles
p_zmin = 2.e-6  # Position of the beginning of the plasma (meters)
p_zmax = 8.e-6  # Position of the end of the plasma (meters)                   
p_rmin = 0.      # Minimal radial position of the plasma (meters)               
p_rmax = 9.e-6   # Maximal radial position of the plasma (meters)               
n_e = 1.e18*1.e6 * scale_factor # Density (electrons.meters^-3)                                
p_nz = 2         # Number of particles per cell along z                         
p_nr = 2         # Number of particles per cell along r                         
p_nt = 4         # Number of particles per cell along theta 

def dens_func( z, r ) :
    """Returns relative density at position z and r"""    
    # Allocate relative density
    n = np.ones_like(z)

    return(n)

# Initialize the simulation object
sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
    p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e, 
    n_order=n_order, dens_func = dens_func, use_cuda = False,
    initialize_ions = True )

# Show the initial fields
plt.figure(0)
sim.fld.interp[0].show('rho')
plt.figure(1)
sim.fld.interp[0].show('Jz')
plt.figure(2)
sim.fld.interp[0].show('Ez')
plt.figure(3)
sim.fld.interp[0].show('Er')
plt.show()
print 'Done'

# Carry out the simulation
for k in range(N_step/N_show) :
    sim.step(N_show, moving_window = False, use_true_rho = True)

    plt.figure(0)
    sim.fld.interp[0].show('rho')
    plt.figure(1)
    sim.fld.interp[0].show('Jz')
    plt.figure(2)
    sim.fld.interp[0].show('Ez')
    plt.figure(3)
    sim.fld.interp[0].show('Er')
    plt.show()
        


