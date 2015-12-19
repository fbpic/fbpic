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
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic

# ----------
# Parameters
# ----------

# The simulation box
Nz = 400         # Number of gridpoints along z
zmax = 40.e-6    # Length of the box along z (meters)
Nr = 50         # Number of gridpoints along r
rmax = 20.e-6    # Length of the box along r (meters)
Nm = 2           # Number of modes used
# The simulation timestep
dt = zmax/Nz/c   # Timestep (seconds)
N_step = 100     # Number of iterations to perform

# The particles
p_zmin = 35.e-6  # Position of the beginning of the plasma (meters)
p_zmax = 41.e-6  # Position of the end of the plasma (meters)
p_rmin = 0.      # Minimal radial position of the plasma (meters)
p_rmax = 18.e-6  # Maximal radial position of the plasma (meters)
n_e = 4.e18*1.e6 # Density (electrons.meters^-3)
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r
p_nt = 4         # Number of particles per cell along theta

# The laser
a0 = 4.          # Laser amplitude
w0 = 5.e-6       # Laser waist
ctau = 5.e-6     # Laser duration
z0 = 25.e-6      # Laser centroid

# The moving window
v_window = c       # Speed of the window

# The diagnostics
diag_period = 10        # Period of the diagnostics in number of timesteps
fieldtypes = [ "E", "rho", "B", "J" ]  # The fields that will be written


# The density profile
ramp_start = 40.e-6
ramp_length = 50.e-6

def dens_func( z, r ) :
    """Returns relative density at position z and r"""    
    # Allocate relative density
    n = np.ones_like(z)
    # Make linear ramp
    n = np.where( z<ramp_start+ramp_length, (z-ramp_start)/ramp_length, n )
    # Supress density before the ramp
    n = np.where( z<ramp_start, 0., n )
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
    dens_func=dens_func ) 

# Add a laser to the fields of the simulation
add_laser( sim.fld, a0, w0, ctau, z0 )

# Configure the moving window
sim.set_moving_window( v=v_window )

# Add a field diagnostic
sim.diags = [ FieldDiagnostic( diag_period, sim.fld, fieldtypes=fieldtypes ),
              ParticleDiagnostic( diag_period, {"electrons" : sim.ptcl[0]},
                                  select={"uz" : [1., None ]} ) ]

### Run the simulation
print('\n Performing %d PIC cycles' % N_step) 
sim.step( N_step )
print('')


