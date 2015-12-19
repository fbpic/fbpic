"""
This is a test with a relativistic plasma flowing through a periodic
box. It can typically be used to investigate the Cherenkov instability.

In particular, there is an option to set the Galilean frame, and thus
try to reduce the Cherenkov instability. 

Usage
-----
Type "python -i test_boosted.py" in a terminal
"""

# -------
# Imports
# -------
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_e, m_p
# Import the relevant structures in FBPIC
from fbpic.main import Simulation, adapt_to_grid
from fbpic.particles import Particles
from fbpic.lpa_utils import add_laser
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic

# ----------
# Parameters
# ----------

# Speed of galilean frame (set to 0 for a normal simulation)
v_galilean = -0.999999*c

# Whether to correct the currents
correct_currents = True

# The simulation box
Nz = 128         # Number of gridpoints along z
zmax = 0.e-6    # Length of the box along z (meters)
zmin = -128.e-6
Nr = 64          # Number of gridpoints along r
rmax = 40.e-6    # Length of the box along r (meters)
Nm = 2           # Number of modes used
# The simulation timestep
dt = (zmax-zmin)/Nz/c   # Timestep (seconds)
N_step = 501     # Number of iterations to perform

# The boost
gamma0 = 130.

# The particles
p_zmin = zmin  # Position of the beginning of the plasma (meters)
p_zmax = zmax  # Position of the end of the plasma (meters)
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

# The diagnostics
diag_period = 10        # Period of the diagnostics in number of timesteps
fieldtypes = [ "E", "rho", "B", "J" ]  # The fields that will be written


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
    dens_func=dens_func, zmin=zmin, initialize_ions=True,
    v_galilean=v_galilean ) 

# Reinitialize the particles, in order to give them a no-zero velocity
p_zmin, p_zmax, Npz = adapt_to_grid( sim.fld.interp[0].z,
                                p_zmin, p_zmax, p_nz, ncells_empty=0 )
p_rmin, p_rmax, Npr = adapt_to_grid( sim.fld.interp[0].r,
                                p_rmin, p_rmax, p_nr, ncells_empty=0 )
sim.ptcl = [
    Particles( q=-e, m=m_e, n=n_e, Npz=Npz, zmin=p_zmin,
            zmax=p_zmax, Npr=Npr, rmin=p_rmin, rmax=p_rmax,
            Nptheta=p_nt, dt=dt, uz_m=-np.sqrt(gamma0**2-1),
            v_galilean=sim.v_galilean ),
    Particles( q=e, m=m_p, n=n_e, Npz=Npz, zmin=p_zmin,
            zmax=p_zmax, Npr=Npr, rmin=p_rmin, rmax=p_rmax,
            Nptheta=p_nt, dt=dt, uz_m=-np.sqrt(gamma0**2-1),
            v_galilean=sim.v_galilean ) ]    

# Add a laser to the fields of the simulation
add_laser( sim.fld, a0, w0, ctau, z0 )

# Add a field diagnostic
sim.diags = [ FieldDiagnostic(diag_period, sim.fld, fieldtypes=fieldtypes),
              ParticleDiagnostic(diag_period, {"electrons" : sim.ptcl[0]}) ]

### Run the simulation
print('\n Performing %d PIC cycles' % N_step) 
sim.step( N_step, correct_currents=correct_currents )
print('')


