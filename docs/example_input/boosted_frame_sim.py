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
from scipy.constants import c
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic, \
                                BoostedFieldDiagnostic

# ----------
# Parameters
# ----------

use_cuda = False

# The simulation box
Nz = 800         # Number of gridpoints along z
zmax = 0.e-6     # Length of the box along z (meters)
zmin = -40.e-6
Nr = 150         # Number of gridpoints along r
rmax = 150.e-6   # Length of the box along r (meters)
Nm = 2           # Number of modes used
# The simulation timestep
dt = (zmax-zmin)/Nz/c   # Timestep (seconds)
N_step = 50      # Number of iterations to perform 
                 # (increase this number for a real simulation)

# Boosted frame
gamma_boost = 15.

# The particles
p_zmin = 0.e-6   # Position of the beginning of the plasma (meters)
p_zmax = 10000.e-6 # Position of the end of the plasma (meters)
p_rmin = 0.      # Minimal radial position of the plasma (meters)
p_rmax = 90.e-6  # Maximal radial position of the plasma (meters)
n_e = 1.e24      # The density in the labframe (electrons.meters^-3)
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r
p_nt = 4         # Number of particles per cell along theta
uz_m = 0.        # Initial momentum of the electrons in the lab frame

# The laser
a0 = 2.          # Laser amplitude
w0 = 50.e-6      # Laser waist
ctau = 9.e-6     # Laser duration
z0 = -20.e-6     # Laser centroid
zfoc = 0.e-6     # Focal position
lambda0 = 0.8e-6 # Laser wavelength

# The moving window
v_window = c       # Speed of the window

# The diagnostics
diag_period = 10        # Period of the diagnostics in number of timesteps
fieldtypes = [ "E", "rho", "B", "J" ]  # The fields that will be written
# Whether to write the fields in the lab frame
Ntot_snapshot_lab = 10
dt_snapshot_lab = (zmax-zmin)/c

def dens_func( z, r ):
    """
    Return the relative density with respect to n_e,
    at the position z and r
    (i.e. return a number between 0 and 1)
    """
    # Allocate relative density
    n = np.ones_like(z)
    n = np.where( z < p_zmin, 0., n )
    n = np.where( z > p_zmax, 0., n )

    return(n)

# ---------------------------
# Carrying out the simulation
# ---------------------------

# Initialize the simulation object
sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
    p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
    dens_func=dens_func, zmin=zmin, initialize_ions=True,
    v_comoving=-0.999*c, use_galilean=False,
    gamma_boost=gamma_boost, boundaries='open', use_cuda=use_cuda )

# Add a laser to the fields of the simulation
add_laser( sim.fld, a0, w0, ctau, z0, lambda0=lambda0,
           zf=zfoc, gamma_boost=gamma_boost )

# Configure the moving window
sim.set_moving_window( v=v_window, gamma_boost=gamma_boost )

# Add a field diagnostic
sim.diags = [ FieldDiagnostic(diag_period, sim.fld, sim.comm ),
              ParticleDiagnostic(diag_period,
                                 {"electrons" : sim.ptcl[0]}, sim.comm),
              BoostedFieldDiagnostic( zmin, zmax, c,
                dt_snapshot_lab, Ntot_snapshot_lab, gamma_boost,
                period=diag_period, fldobject=sim.fld, comm=sim.comm) ]

### Run the simulation
print('\n Performing %d PIC cycles' % N_step)
sim.step( N_step, use_true_rho=True )
print('')
