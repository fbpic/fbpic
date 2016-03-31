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
from fbpic.lpa_utils.bunch import add_elec_bunch_gaussian
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic, \
                            BoostedFieldDiagnostic, BoostedParticleDiagnostic
# ----------
# Parameters
# ----------
use_cuda = False

# Boosted frame
gamma_boost = 10.
boost = BoostConverter(gamma_boost)


# The simulation box                                                            
Nz = 400         # Number of gridpoints along z
zmin = -40.e-6
zmax = 0.e-6    # Length of the box along z (meters)
Nr = 50         # Number of gridpoints along r 
rmax = 100.e-6   # Length of the box along r (meters)
Nm = 2           # Number of modes used  
n_guard = 40     # Number of guard cells
exchange_period = 10 # Exchange period         
n_order = -1     # The order of the stencil in z
                                             
# The simulation timestep
dt = (zmax-zmin)/Nz/c   # Timestep (seconds)
N_step = 5000     # Number of iterations to perform
                 # (increase this number for a real simulation)

# Placeholder bunch (will be overwritten)
p_zmin = -25.e-6  # Position of the beginning of the bunch (meters)
p_zmax = -15.e-6  # Position of the end of the bunch (meters)                   
p_rmin = 0.      # Minimal radial position of the bunch (meters)               
p_rmax = 5.e-6   # Maximal radial position of the bunch (meters)               
n_e = 4.e18*1.e6 # Density (electrons.meters^-3)                                
p_nz = 2         # Number of particles per cell along z                         
p_nr = 2         # Number of particles per cell along r                         
p_nt = 4         # Number of particles per cell along theta 

# Gaussian bunch parameters
sig_r = 3.e-6
sig_z = 5.e-6
n_emit = 3e-6
gamma0 = 100.
sig_gamma = 0.1
Q = 1.e-30
N = 100000
tf = 320.e-6 / (np.sqrt(1.-1./gamma0**2)*c)
zf = 300.e-6

# The moving window (moves with the group velocity in a plasma)
v_window = c
# Convert parameter to boosted frame
v_window, = boost.velocity( [ v_window ] )

# The diagnostics
diag_period = 200        # Period of the diagnostics in number of timesteps
# Whether to write the fields in the lab frame
Ntot_snapshots_lab = 50
dt_snapshots_lab = 0.5*(zmax-zmin)/c

# ---------------------------
# Carrying out the simulation
# ---------------------------

# Initialize the simulation object
sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
    p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
    zmin=zmin, initialize_ions=False,
#    v_comoving=-0.9999*c, use_galilean=True,
    n_guard=n_guard, exchange_period=exchange_period,
    gamma_boost=gamma_boost, boundaries='open', use_cuda=use_cuda )

sim.ptcl = []

# Add an electron bunch
add_elec_bunch_gaussian( sim, sig_r, sig_z, n_emit, gamma0, sig_gamma, 
                        Q, N, tf, zf, boost )

# Configure the moving window
sim.set_moving_window( v=v_window, gamma_boost=gamma_boost )

# Add a field diagnostic
sim.diags = [ FieldDiagnostic(diag_period, sim.fld, sim.comm ),
              ParticleDiagnostic(diag_period,
                {"bunch":sim.ptcl[0]}, sim.comm),
              BoostedFieldDiagnostic( zmin, zmax, c,
                dt_snapshots_lab, Ntot_snapshots_lab, gamma_boost,
                period=diag_period, fldobject=sim.fld, comm=sim.comm),
              BoostedParticleDiagnostic(zmin, zmax, c, dt_snapshots_lab,
                 Ntot_snapshots_lab, gamma_boost, diag_period, 
                 sim.fld.interp[0].zmin, sim.fld.interp[0].dz, 
                 species = {"bunch":sim.ptcl[0]}, comm=sim.comm) ]

### Run the simulation
print('\n Performing %d PIC cycles' % N_step)
sim.step( N_step, use_true_rho=True )
print('')
