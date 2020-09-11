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
from scipy.constants import c, e, m_e, m_p
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser
from fbpic.lpa_utils.bunch import add_particle_bunch
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic, \
        BackTransformedFieldDiagnostic, BackTransformedParticleDiagnostic
# ----------
# Parameters
# ----------
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

# Boosted frame
gamma_boost = 10.
# Boosted frame converter
boost = BoostConverter(gamma_boost)

# The simulation box
Nz = 600         # Number of gridpoints along z
zmax = 0.e-6     # Length of the box along z (meters)
zmin = -30.e-6
Nr = 75          # Number of gridpoints along r
rmax = 150.e-6   # Length of the box along r (meters)
Nm = 2           # Number of modes used

# The simulation timestep
# (See the section Advanced use > Running boosted-frame simulation
# of the FBPIC documentation for an explanation of the calculation of dt)
dt = min( rmax/(2*boost.gamma0*Nr)/c, (zmax-zmin)/Nz/c )  # Timestep (seconds)

# The laser (conversion to boosted frame is done inside 'add_laser')
a0 = 2.          # Laser amplitude
w0 = 50.e-6      # Laser waist
ctau = 5.e-6     # Laser duration
z0 = -10.e-6     # Laser centroid
zfoc = 0.e-6     # Focal position
lambda0 = 0.8e-6 # Laser wavelength

# The density profile
w_matched = 50.e-6
ramp_up = .5e-3
plateau = 3.5e-3
ramp_down = .5e-3

# The particles of the plasma
p_zmin = 0.e-6   # Position of the beginning of the plasma (meters)
p_zmax = ramp_up + plateau + ramp_down
p_rmax = 100.e-6 # Maximal radial position of the plasma (meters)
n_e = 3.e24      # The density in the labframe (electrons.meters^-3)
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r
p_nt = 6         # Number of particles per cell along theta

# Density profile
# Convert parameters to boosted frame
# (NB: the density is converted inside the Simulation object)
ramp_up_b, plateau_b, ramp_down_b = \
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
    # Make ramp up (note: use boosted-frame values of the ramp length)
    inv_ramp_up_b = 1./ramp_up_b
    n = np.where( z<ramp_up_b, z*inv_ramp_up_b, n )
    # Make ramp down
    inv_ramp_down_b = 1./ramp_down_b
    n = np.where( (z >= ramp_up_b+plateau_b) & \
                  (z < ramp_up_b+plateau_b+ramp_down_b),
              - (z - (ramp_up_b+plateau_b+ramp_down_b) )*inv_ramp_down_b, n )
    n = np.where( z >= ramp_up_b+plateau_b+ramp_down_b, 0, n)
    # Add transverse guiding parabolic profile
    n = n * ( 1. + rel_delta_n_over_w2 * r**2 )
    return(n)

# The bunch
bunch_zmin = z0 - 15.e-6
bunch_zmax = bunch_zmin + 3.e-6
bunch_rmax = 10.e-6
bunch_gamma = 400.
bunch_n = 5.e23

# The moving window (moves with the group velocity in a plasma)
v_window = c*( 1 - 0.5*n_e/1.75e27 )

# Velocity of the Galilean frame (for suppression of the NCI)
v_comoving = - c * np.sqrt( 1. - 1./boost.gamma0**2 )

# The interaction length of the simulation, in the lab frame (meters)
L_interact = (p_zmax-p_zmin) # the plasma length
# Interaction time, in the boosted frame (seconds)
T_interact = boost.interaction_time( L_interact, (zmax-zmin), v_window )
# (i.e. the time it takes for the moving window to slide across the plasma)

## The diagnostics

# Number of discrete diagnostic snapshots, for the diagnostics in the
# boosted frame (i.e. simulation frame) and in the lab frame
# (i.e. back-transformed from the simulation frame to the lab frame)
N_boosted_diag = 15+1
N_lab_diag = 10+1
# Time interval between diagnostic snapshots *in the lab frame*
# (first at t=0, last at t=T_interact)
dt_lab_diag_period = (L_interact + (zmax-zmin)) / v_window / (N_lab_diag - 1)
# Time interval between diagnostic snapshots *in the boosted frame*
dt_boosted_diag_period = T_interact / (N_boosted_diag - 1)
# Period of writing the cached, backtransformed lab frame diagnostics to disk
write_period = 50

# Whether to tag and track the particles of the bunch
track_bunch = False

# ---------------------------
# Carrying out the simulation
# ---------------------------
# NB: The code below is only executed when running the script,
# (`python boosted_frame_sim.py`), but not when importing it.
if __name__ == '__main__':

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
        v_comoving=v_comoving, gamma_boost=boost.gamma0,
        n_order=n_order, use_cuda=use_cuda,
        boundaries={'z':'open', 'r':'reflective'})
        # 'r': 'open' can also be used, but is more computationally expensive

    # Add the plasma electron and plasma ions
    plasma_elec = sim.add_new_species( q=-e, m=m_e,
                    n=n_e, dens_func=dens_func,
                    p_zmin=p_zmin, p_zmax=p_zmax, p_rmax=p_rmax,
                    p_nz=p_nz, p_nr=p_nr, p_nt=p_nt )
    plasma_ions = sim.add_new_species( q=e, m=m_p,
                    n=n_e, dens_func=dens_func,
                    p_zmin=p_zmin, p_zmax=p_zmax, p_rmax=p_rmax,
                    p_nz=p_nz, p_nr=p_nr, p_nt=p_nt )

    # Add a relativistic electron bunch
    bunch = add_particle_bunch( sim, -e, m_e, bunch_gamma,
        bunch_n, bunch_zmin, bunch_zmax, 0, bunch_rmax, boost=boost )
    if track_bunch:
        bunch.track( sim.comm )

    # Add a laser to the fields of the simulation
    add_laser( sim, a0, w0, ctau, z0, lambda0=lambda0,
           zf=zfoc, gamma_boost=boost.gamma0 )

    # Convert parameter to boosted frame
    v_window_boosted, = boost.velocity( [ v_window ] )
    # Configure the moving window
    sim.set_moving_window( v=v_window_boosted )

    # Add a field diagnostic
    sim.diags = [
                  # Diagnostics in the boosted frame
                  FieldDiagnostic( dt_period=dt_boosted_diag_period,
                                   fldobject=sim.fld, comm=sim.comm ),
                  ParticleDiagnostic( dt_period=dt_boosted_diag_period,
                        species={"electrons":plasma_elec, "bunch":bunch},
                        comm=sim.comm),
                  # Diagnostics in the lab frame (back-transformed)
                  BackTransformedFieldDiagnostic( zmin, zmax, v_window,
                    dt_lab_diag_period, N_lab_diag, boost.gamma0,
                    fieldtypes=['rho','E','B'], period=write_period,
                    fldobject=sim.fld, comm=sim.comm ),
                  BackTransformedParticleDiagnostic( zmin, zmax, v_window,
                    dt_lab_diag_period, N_lab_diag, boost.gamma0,
                    write_period, sim.fld, select={'uz':[0.,None]},
                    species={'bunch':bunch}, comm=sim.comm )
                ]

    # Number of iterations to perform
    N_step = int(T_interact/sim.dt)

    ### Run the simulation
    sim.step( N_step )
    print('')
