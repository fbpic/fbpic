"""
This file tests the space charge initialization implemented
in lpa_utils.py, by initializing a charged bunch and propagating
it for a few steps

Usage :
from the top-level directory of FBPIC run
$ python tests/test_space_charge.py
"""
import matplotlib.pyplot as plt
from scipy.constants import c
# Import the relevant structures in FBPIC                                       
from fbpic.main import Simulation
from fbpic.lpa_utils.bunch import add_elec_bunch_gaussian

# The simulation box                                                            
Nz = 400         # Number of gridpoints along z
zmax = 40.e-6    # Length of the box along z (meters)
Nr = 100         # Number of gridpoints along r 
rmax = 100.e-6   # Length of the box along r (meters)
Nm = 2           # Number of modes used             
n_order = -1     # The order of the stencil in z

# The simulation timestep                                              
dt = zmax/Nz/c   # Timestep (seconds)                                           
N_step = 300     # Number of iterations to perform
N_show = 300     # Number of timestep between every plot                

# Initialize the simulation object
sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
    0., 1.e-6, 0., 1.e6., 1, 1, 1, 1.e18,
    n_order=n_order, boundaries='open' )

# Configure the moving window
sim.set_moving_window(v=c)

# Suppress the particles that were intialized by default and add the bunch
sim.ptcl = [ ]

# Bunch parameters
sig_r = 3.e-6
sig_z = 3.e-6
n_emit = 1.e-6
gamma0 = 25.
sig_gamma = 0.5
Q = 10.e-12
N = 10000
tf = 20.e-6 / (np.sqrt(1.-1./gamma0**2)*c)
zf = 20.e-6

# Add gaussian electron bunch
add_elec_bunch_gaussian( sim, sig_r, sig_z, n_emit, gamma0, sig_gamma, 
                         Q, N, tf, zf )


# Show the initial fields
plt.figure(0)
sim.fld.interp[0].show('Ez')
plt.figure(1)
sim.fld.interp[0].show('Er')
plt.show()
print 'Done'

# Carry out the simulation
for k in range(N_step/N_show) :
    sim.step(N_show)

    plt.figure(0)
    plt.clf()
    sim.fld.interp[0].show('Ez')
    plt.figure(1)
    plt.clf()
    sim.fld.interp[0].show('Er')
    plt.show()
    


