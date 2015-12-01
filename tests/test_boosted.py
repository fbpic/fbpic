"""
This is a typical input script that runs a simulation of
laser-wakefield acceleration using FBPIC.

Usage
-----
- Modify the parameters below to suit your needs
- Type "python -i lpa_sim.py" in a terminal
- When the simulation finishes, the python session will *not* quit.
    Therefore the simulation can be continued by running sim.step()

Help
----
All the structures implemented in FBPIC are internally documented.
Enter "print(fbpic_object.__doc__)" to have access to this documentation,
where fbpic_object is any of the objects or function of FBPIC.
"""

# -------
# Imports
# -------
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils import add_laser
from fbpic.moving_window import MovingWindow
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic

# ----------
# Parameters
# ----------

# The simulation box
Nz = 336         # Number of gridpoints along z
zmax = 10.e-6    # Length of the box along z (meters)
zmin = -200.e-6
Nr = 64          # Number of gridpoints along r
rmax = 40.e-6    # Length of the box along r (meters)
Nm = 2           # Number of modes used
# The simulation timestep
dt = (zmax-zmin)/Nz/c   # Timestep (seconds)
N_step = 101     # Number of iterations to perform

# The boost
gamma0 = 130.

# The particles
p_zmin = -35.e-6  # Position of the beginning of the plasma (meters)
p_zmax = 5.e-6  # Position of the end of the plasma (meters)
p_rmin = 0.      # Minimal radial position of the plasma (meters)
p_rmax = 41.e-6  # Maximal radial position of the plasma (meters)
n_e = 5.e25 * gamma0
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r
p_nt = 4         # Number of particles per cell along theta

# The laser
a0 = 0.          # Laser amplitude
w0 = 5.e-6       # Laser waist
ctau = 5.e-6     # Laser duration
z0 = 25.e-6      # Laser centroid

# The moving window
v_window = c       # Speed of the window
ncells_zero = 50    # Number of cells over which the field is set to 0
                   # at the left end of the simulation box
ncells_damp = 30   # Number of cells over which the field is damped,
                   # at the left of the simulation box, after ncells_zero
                   # in order to prevent it from wrapping around.


# The density profile
r_max = 35.e-6

def dens_func( z, r ):
    """
    Return the relative density with respect to n_e,
    at the position z and r
    (i.e. return a number between 0 and 1)
    """
    # Allocate relative density
    n = np.ones_like(z)
    # Suppress density at high radius
    n = np.where( r<r_max, n, 0.)
    
    return(n)

# -----------------------
# Checking the parameters
# -----------------------
if p_nr%2 == 1 :
    raise UserWarning("Running the simulation with an odd number \n"
                      "of macroparticles may result in a very \n"
                      "noisy simulation.")

# ---------------------------
# Carrying out the simulation
# ---------------------------

# Initialize the simulation object
sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
    p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
    dens_func=dens_func, zmin=zmin, initialize_ions=True ) 

# Add a laser to the fields of the simulation
add_laser( sim.fld, a0, w0, ctau, z0 )

# Configure the moving window
sim.moving_win = MovingWindow( sim.fld.interp[0],
                               ncells_damp=ncells_damp,
                               ncells_zero=ncells_zero,
                               uz_m = -np.sqrt(gamma0**2-1) )

# Show the initial fields
plt.figure(0)
sim.fld.interp[0].show('Ez')
plt.figure(1)
sim.fld.interp[0].show('Er')
plt.figure(2)
sim.fld.interp[0].show('Jz')
plt.figure(3)
sim.fld.interp[0].show('rho')
plt.show()
print 'Done'

### Run the simulation
print('\n Performing %d PIC cycles' % N_step) 
sim.step( N_step )
print('')

# Show the initial fields
plt.figure(0)
sim.fld.interp[0].show('Ez')
plt.figure(1)
sim.fld.interp[0].show('Er')
plt.figure(2)
sim.fld.interp[0].show('Jz')
plt.figure(3)
sim.fld.interp[0].show('rho')
plt.show()
print 'Done'
