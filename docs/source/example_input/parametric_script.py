"""
This script runs two simulations in parallel using MPI,
with each simulation having a different value of a0 as input.
(Each simulation is performed by a different MPI process.)

This makes use of the parameter `use_all_mpi_ranks=False` in
the `Simulation` object, which allows each MPI rank to carry out a
different simulation. Search for the lines tagged with the comment
`Parametric scan` to find the lines that are key for this technique.

This can be useful for instance to run several simulations on one
node that has several GPUs.

Usage
-----
In a terminal, type:
  mpirun -np 2 python parametric_script.py
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
# Parametric scan: import mpi4py so as to be able to give different
# input parameters to each MPI rank.
from mpi4py.MPI import COMM_WORLD as comm

# ----------
# Parameters
# ----------

# Whether to use the GPU
use_cuda = True

# Order of accuracy of the spectral, Maxwell (PSATD) solver.
# -1 correspond to infinite order, i.e. wave propagation is perfectly
# dispersion-free in all directions. This is adviced for single GPU/CPU
# simulations. For multi GPU/CPU simulations, choose n_order > 4
# (and multiple of 2). A large n_order leads to more overhead in MPI
# communications, but also to a more accurate dispersion for waves.
# (Typically, n_order = 32 gives accurate physical results)
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
p_zmin = 25.e-6  # Position of the beginning of the plasma (meters)
p_zmax = 500.e-6 # Position of the end of the plasma (meters)
p_rmax = 18.e-6  # Maximal radial position of the plasma (meters)
n_e = 4.e18*1.e6 # Density (electrons.meters^-3)
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r
p_nt = 4         # Number of particles per cell along theta

# The laser
w0 = 5.e-6       # Laser waist
ctau = 5.e-6     # Laser duration
z0 = 15.e-6      # Laser centroid

# Parametric scan: Give a list of a0 values to scan,
# and pick one value that this rank takes as input parameter
a0_list = [ 2.0, 4.0 ]
if len(a0_list) != comm.size:
    raise ValueError(
        'This script should be launched with %d MPI ranks.'%len(a0_list))
a0 = a0_list[ comm.rank ]

# The moving window
v_window = c       # Speed of the window

# The diagnostics and the checkpoints/restarts
diag_period = 50         # Period of the diagnostics in number of timesteps
save_checkpoints = False # Whether to write checkpoint files
checkpoint_period = 100  # Period for writing the checkpoints
use_restart = False      # Whether to restart from a previous checkpoint

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
# (`python -i lpa_sim.py`), but not when importing it (`import lpa_sim`).
if __name__ == '__main__':

    # Initialize the simulation object
    # Parametric scan: use the flag `use_all_mpi_ranks=False` to
    # have each MPI rank run an independent simulation
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
        boundaries='open', n_order=n_order, use_cuda=use_cuda,
        use_all_mpi_ranks=False )

    # Create the plasma electrons
    elec = sim.add_new_species( q=-e, m=m_e,
                n=n_e, dens_func=dens_func,
                p_zmin=p_zmin, p_zmax=p_zmax, p_rmax=p_rmax,
                p_nz=p_nz, p_nr=p_nr, p_nt=p_nt )

    # Load initial fields
    # Add a laser to the fields of the simulation
    add_laser( sim, a0, w0, ctau, z0 )

    if use_restart is True:
        # Load the fields and particles from the latest checkpoint file
        restart_from_checkpoint( sim )

    # Configure the moving window
    sim.set_moving_window( v=v_window )

    # Add a field diagnostic
    # Parametric scan: each MPI rank should output its data to a
    # different directory
    write_dir = 'diags_a0_%.2f' %a0
    sim.diags = [ FieldDiagnostic( diag_period, sim.fld,
                    comm=sim.comm, write_dir=write_dir ),
                ParticleDiagnostic( diag_period, {"electrons" : elec},
                    select={"uz" : [1., None ]},
                    comm=sim.comm, write_dir=write_dir ) ]

    # Add checkpoints
    if save_checkpoints:
        set_periodic_checkpoint( sim, checkpoint_period )

    # Number of iterations to perform
    N_step = int(T_interact/sim.dt)

    ### Run the simulation
    sim.step( N_step )
    print('')
