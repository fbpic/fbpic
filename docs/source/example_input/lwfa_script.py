"""
This is a typical input script that runs a simulation of
laser-wakefield acceleration using FBPIC.

Usage
-----
- Modify the parameters below to suit your needs
- Type "python lwfa_script.py" in a terminal

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
from scipy.constants import c, e, m_e
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic, \
     set_periodic_checkpoint, restart_from_checkpoint

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
n_order = -1

# The simulation box
Nz = 800         # Number of gridpoints along z
zmax = 30.e-6    # Right end of the simulation box (meters)
zmin = -10.e-6   # Left end of the simulation box (meters)
Nr = 50          # Number of gridpoints along r
rmax = 20.e-6    # Length of the box along r (meters)
Nm = 2           # Number of modes used

# The simulation timestep
dt = (zmax-zmin)/Nz/c   # Timestep (seconds)

# The particles
p_zmin = 30.e-6  # Position of the beginning of the plasma (meters)
p_zmax = 500.e-6 # Position of the end of the plasma (meters)
p_rmax = 18.e-6  # Maximal radial position of the plasma (meters)
n_e = 4.e18*1.e6 # Density (electrons.meters^-3)
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r
p_nt = 4         # Number of particles per cell along theta

# The laser
a0 = 4.          # Laser amplitude
w0 = 5.e-6       # Laser waist
ctau = 5.e-6     # Laser duration
z0 = 15.e-6      # Laser centroid

# The moving window
v_window = c       # Speed of the window

# The diagnostics and the checkpoints/restarts
diag_period = 50         # Period of the diagnostics in number of timesteps
save_checkpoints = False # Whether to write checkpoint files
checkpoint_period = 100  # Period for writing the checkpoints
use_restart = False      # Whether to restart from a previous checkpoint
track_electrons = False  # Whether to track and write particle ids

# The density profile
ramp_start = 30.e-6
ramp_length = 40.e-6

def dens_func( z, r ) :
    """Returns relative density at position z and r"""
    # Allocate relative density
    n = np.ones_like(z)
    # Make linear ramp
    n = np.where( z<ramp_start+ramp_length, (z-ramp_start)/ramp_length, n )
    # Supress density before the ramp
    n = np.where( z<ramp_start, 0., n )
    return(n)

# The interaction length of the simulation (meters)
L_interact = 50.e-6 # increase to simulate longer distance!
# Interaction time (seconds) (to calculate number of PIC iterations)
T_interact = ( L_interact + (zmax-zmin) ) / v_window
# (i.e. the time it takes for the moving window to slide across the plasma)

# ---------------------------
# Carrying out the simulation
# ---------------------------

# NB: The code below is only executed when running the script,
# (`python lwfa_script.py`), but not when importing it (`import lwfa_script`).
if __name__ == '__main__':

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
        n_order=n_order, use_cuda=use_cuda,
        boundaries={'z':'open', 'r':'reflective'})
        # 'r': 'open' can also be used, but is more computationally expensive

    # Create the plasma electrons
    elec = sim.add_new_species( q=-e, m=m_e, n=n_e,
        dens_func=dens_func, p_zmin=p_zmin, p_zmax=p_zmax, p_rmax=p_rmax,
        p_nz=p_nz, p_nr=p_nr, p_nt=p_nt )

    # Load initial fields
    # Add a laser to the fields of the simulation
    add_laser( sim, a0, w0, ctau, z0 )

    if use_restart is False:
        # Track electrons if required (species 0 correspond to the electrons)
        if track_electrons:
            elec.track( sim.comm )
    else:
        # Load the fields and particles from the latest checkpoint file
        restart_from_checkpoint( sim )

    # Configure the moving window
    sim.set_moving_window( v=v_window )

    # Add diagnostics
    sim.diags = [ FieldDiagnostic( diag_period, sim.fld, comm=sim.comm ),
                  ParticleDiagnostic( diag_period, {"electrons" : elec},
                    select={"uz" : [1., None ]}, comm=sim.comm ) ]
    # Add checkpoints
    if save_checkpoints:
        set_periodic_checkpoint( sim, checkpoint_period )

    # Number of iterations to perform
    N_step = int(T_interact/sim.dt)

    ### Run the simulation
    sim.step( N_step )
    print('')
