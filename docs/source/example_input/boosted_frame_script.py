"""
This is a typical input script that runs a simulation of
laser-wakefield acceleration using FBPIC.

Usage
-----
- Modify the parameters below to suit your needs
- Type "python boosted_frame_script.py" in a terminal

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
from fbpic.lpa_utils.bunch import add_elec_bunch
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic, \
                  BoostedFieldDiagnostic, BoostedParticleDiagnostic
# ----------
# Parameters
# ----------
use_cuda = True

# The simulation box
Nz = 400         # Number of gridpoints along z
zmax = 0.e-6     # Length of the box along z (meters)
zmin = -20.e-6
Nr = 75          # Number of gridpoints along r
rmax = 150.e-6   # Length of the box along r (meters)
Nm = 2           # Number of modes used
# Boosted frame
gamma_boost = 15.
# The simulation timestep
dt = min( rmax/(2*gamma_boost*Nr), (zmax-zmin)/Nz/c )  # Timestep (seconds)
# (See the section Advanced use > Running boosted-frame simulation
# of the FBPIC documentation for an explanation of the above calculation of dt)
N_step = 101     # Number of iterations to perform
                 # (increase this number for a real simulation)

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

# Boosted frame converter
boost = BoostConverter(gamma_boost)

# The laser (conversion to boosted frame is done inside 'add_laser')
a0 = 2.          # Laser amplitude
w0 = 50.e-6      # Laser waist
ctau = 5.e-6     # Laser duration
z0 = -10.e-6     # Laser centroid
zfoc = 0.e-6     # Focal position
lambda0 = 0.8e-6 # Laser wavelength

# The density profile
w_matched = 50.e-6
ramp_up = 5.e-3
plateau = 8.e-2
ramp_down = 5.e-3

# The particles of the plasma
p_zmin = 0.e-6   # Position of the beginning of the plasma (meters)
p_zmax = ramp_up + plateau + ramp_down
p_rmin = 0.      # Minimal radial position of the plasma (meters)
p_rmax = 100.e-6 # Maximal radial position of the plasma (meters)
n_e = 1.e24      # The density in the labframe (electrons.meters^-3)
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r
p_nt = 6         # Number of particles per cell along theta
uz_m = 0.        # Initial momentum of the electrons in the lab frame

# Density profile
# Convert parameters to boosted frame
# (NB: the density is converted inside the Simulation object)
ramp_up, plateau, ramp_down = \
    boost.static_length( [ ramp_up, plateau, ramp_down ] )
# Relative change divided by w_matched^2 that allows guiding
rel_delta_n_over_w2 = 1./( np.pi * 2.81e-15 * w_matched**4 * n_e )
# Define the density function
def dens_func( z, r ):
    """
    User-defined function: density profile of the plasma

    It should return the relative density with respect to n_plasma,
    at the position x, y, z (i.e. return a number between 0 and 1)

    Parameters
    ----------
    z, r: 1darrays of floats
        Arrays with one element per macroparticle
    Returns
    -------
    n : 1d array of floats
        Array of relative density, with one element per macroparticles
    """
    # Allocate relative density
    n = np.ones_like(z)
    # Make ramp up
    inv_ramp_up = 1./ramp_up
    n = np.where( z<ramp_up, z*inv_ramp_up, n )
    # Make ramp down
    inv_ramp_down = 1./ramp_down
    n = np.where( (z >= ramp_up+plateau) & (z < ramp_up+plateau+ramp_down),
              - (z - (ramp_up+plateau+ramp_down) )*inv_ramp_down, n )
    n = np.where( z >= ramp_up+plateau+ramp_down, 0, n)
    # Add transverse guiding parabolic profile
    n = n * ( 1. + rel_delta_n_over_w2 * r**2 )
    return(n)

# The bunch
bunch_zmin = z0 - 10.e-6
bunch_zmax = bunch_zmin + 4.e-6
bunch_rmax = 10.e-6
bunch_gamma = 400.
bunch_n = 5.e23

# The moving window (moves with the group velocity in a plasma)
v_window = c*( 1 - 0.5*n_e/1.75e27 )
# Convert parameter to boosted frame
v_window, = boost.velocity( [ v_window ] )

# Velocity of the Galilean frame (for suppression of the NCI)
v_comoving = - c * np.sqrt( 1. - 1./gamma_boost**2 )

# The diagnostics
diag_period = 50        # Period of the diagnostics in number of timesteps
# Whether to write the fields in the lab frame
Ntot_snapshot_lab = 20
dt_snapshot_lab = (zmax-zmin)/c
track_bunch = False  # Whether to tag and track the particles of the bunch

# ---------------------------
# Carrying out the simulation
# ---------------------------
# NB: The code below is only executed when running the script,
# (`python boosted_frame_sim.py`), but not when importing it.
if __name__ == '__main__':

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
        p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
        dens_func=dens_func, zmin=zmin, initialize_ions=True,
        v_comoving=v_comoving, gamma_boost=gamma_boost, n_order=n_order,
        boundaries='open', use_cuda=use_cuda )

    # Add an electron bunch
    add_elec_bunch( sim, bunch_gamma, bunch_n, bunch_zmin,
                bunch_zmax, 0, bunch_rmax, boost=boost )
    if track_bunch:
        sim.ptcl[2].track( sim.comm )

    # Add a laser to the fields of the simulation
    add_laser( sim, a0, w0, ctau, z0, lambda0=lambda0,
           zf=zfoc, gamma_boost=gamma_boost )

    # Configure the moving window
    sim.set_moving_window( v=v_window )

    # Add a field diagnostic
    sim.diags = [ FieldDiagnostic(diag_period, sim.fld, sim.comm ),
                 ParticleDiagnostic(diag_period,
                     {"electrons":sim.ptcl[0], "bunch":sim.ptcl[2]}, sim.comm),
                 BoostedFieldDiagnostic( zmin, zmax, c,
                    dt_snapshot_lab, Ntot_snapshot_lab, gamma_boost,
                    period=diag_period, fldobject=sim.fld, comm=sim.comm),
                BoostedParticleDiagnostic( zmin, zmax, c, dt_snapshot_lab,
                    Ntot_snapshot_lab, gamma_boost, diag_period, sim.fld,
                    select={'uz':[0.,None]}, species={'electrons':sim.ptcl[2]},
                    comm=sim.comm )
                    ]

    ### Run the simulation
    sim.step( N_step )
    print('')
