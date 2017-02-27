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
from scipy.constants import c
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

# The simulation box
Nz = 400         # Number of gridpoints along z
zmax = 30.e-6    # Right end of the simulation box (meters)
zmin = -10.e-6   # Left end of the simulation box (meters)
Nr = 50          # Number of gridpoints along r
rmax = 20.e-6    # Length of the box along r (meters)
Nm = 2           # Number of modes used
# The simulation timestep
dt = (zmax-zmin)/Nz/c   # Timestep (seconds)
N_step = 51     # Number of iterations to perform

# The particles
p_zmin = 25.e-6  # Position of the beginning of the plasma (meters)
p_zmax = 31.e-6  # Position of the end of the plasma (meters)
p_rmin = 0.      # Minimal radial position of the plasma (meters)
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
diag_period = 10         # Period of the diagnostics in number of timesteps
save_checkpoints = False # Whether to write checkpoint files
checkpoint_period = 50   # Period for writing the checkpoints
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

# ---------------------------
# Carrying out the simulation
# ---------------------------

# NB: The code below is only executed when running the script,
# (`python -i lpa_sim.py`), but not when importing it (`import lpa_sim`).
if __name__ == '__main__':

    # Initialize the simulation object
    # Parametric scan: use the flag `use_all_mpi_ranks=False` to
    # have each MPI rank run an independent simulation
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
        p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
        dens_func=dens_func, zmin=zmin, boundaries='open',
        use_cuda=use_cuda, use_all_mpi_ranks=False )

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
    write_dir = 'diags_a0_%.1f' %a0
    sim.diags = [ FieldDiagnostic( diag_period, sim.fld,
                    comm=sim.comm, write_dir=write_dir ),
                ParticleDiagnostic( diag_period, {"electrons" : sim.ptcl[0]},
                    select={"uz" : [1., None ]},
                    comm=sim.comm, write_dir=write_dir ) ]

    # Add checkpoints
    if save_checkpoints:
        set_periodic_checkpoint( sim, checkpoint_period )

    ### Run the simulation
    sim.step( N_step )
    print('')
