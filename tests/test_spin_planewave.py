"""
Usage :
from the top-level directory of FBPIC run
$ python spin_planewave_test.py

Script is based on test_external_fields.py by Remi Lehe & Manuel Kirchen

Author: Michael J. Quin, Kristjan Poder
Scientific supervision: Matteo Tamburini

-------------------------------------------------------------------------------

This file tests the spin pusher for one electron initially at the origin,
in a monochromatic linearly polarised plane wave.

For an electron with initial position, velocity, Lorentz factor and spin:
r_i = (0, 0, 0)
u_i = (0, 0, uz_i)
gamma_i = sqrt( 1 + uz_i**2 )
s_i = (sx_i, sy_i, sz_i)

In a plane wave:
phi = k0*(ct - z)
A(phi) = (a0*m_e*c/e)*sin(phi)

The equations of motion in a plane wave are well known:
ux(phi) = a0*sin(phi)
uy(phi) = 0
uz(phi) = u0z + 0.5*(a0*sin(phi))**2/(gamma_i - uz_i)

(see e.g. Salamin & Faisal. Phys. Rev. A 54 (5 Nov. 1996), pp. 4383–4395.
DOI : 10.1103/PhysRevA.54.4383)

For these equations of motion, one can solve the
Bargmann-Michel-Telegdi equation to obtain the following
solutions (see  M. J. Quin MSc thesis, DOI: 10.11588/heidok.00033704):

sx(phi) = sx_i*cos(I[phi]) - sz_i*sin(I[phi])
sy(phi) = sy_i
sz(phi) = sz_i*cos(I[phi]) + sx_i*sin(I[phi])
I(phi) = a0*anom*sin(phi) + 2*arctan(a0*sin(phi)/(gamma_i - uz_i))

Where anom ~ 1.16e-3  is the anomalous electron magnetic moment,
a one loop radiative correction to the electron's magnetic moment, and arctan
is solved in the closed interval [-pi/2, pi/2].

"""
import math
import numpy as np

# Constants:
from scipy.constants import e, m_e, c, pi
from scipy.constants import physical_constants
anom = physical_constants['electron mag. mom. anomaly'][0]

# Simulation class:
from fbpic.main import Simulation
from fbpic.lpa_utils.external_fields import ExternalField

# Parameters
# ----------
use_cuda = True

# Properties of plane wave
a0 = 1.
lambda0 = 0.8e-6
T = lambda0 / c
k0 = 2 * pi / lambda0

# Time parameters
dt = lambda0 / c / 50  # 50 points per oscillation
N_step = 50 + 1      # one oscillation, including data from final step

# Dimensions of the box
# (low_resolution since the external field are not resolved on the grid)
Nz = 5
Nr = 10
Nm = 2
zmin = -10 * lambda0
zmax = 10 * lambda0
rmax = 20 * lambda0

# Particles (one per cell, only initialized near the axis)
p_rmax = rmax / Nr
p_nt = 1
p_nr = 1
p_nz = 1
n = 1.

# Initial momentum and spin
uz_i = -a0 ** 2 / (4 * math.sqrt(1. + a0 ** 2 / 2))  # figure-of-eight velocity
sx_i = 0.
sy_i = 0.
sz_i = 1.


def run_external_laser_field_simulation(show):

    # (1) FIRST SIMULATION: PUSH MOMENTUM & SPIN BACK ONE HALF STEP
    # FBPIC uses a leapfrog pusher, where momentum and spin are defined at
    # half integer steps, postion at integer steps. Need to push initial
    # momentum back one half step.

    # Initialise simulation
    halfstepback = Simulation(Nz, zmax, Nr, rmax, Nm, -dt/2,
                              initialize_ions=False, zmin=zmin,
                              use_cuda=use_cuda,
                              boundaries={'z': 'periodic',
                                          'r': 'reflective'})
    # Add electrons
    halfstepback.ptcl = []
    elec = halfstepback.add_new_species(q=-e, m=m_e, n=n, p_rmax=p_rmax,
                                        p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
                                        ux_m=0., uy_m=0., uz_m=uz_i)
    # And activate spin tracking
    elec.activate_spin_tracking(sx_m=sx_i, sy_m=sy_i, sz_m=sz_i,
                                anom=anom)

    # Add the external fields
    halfstepback.external_fields = [
        ExternalField(laser_func, 'Ex', a0*m_e*c**2*k0/e, lambda0),
        ExternalField(laser_func, 'By', a0*m_e*c*k0/e, lambda0)]

    # In deriving analytic solutions electron was assumed to be initially at
    # origin. In a cylindrical co-ord system, it is problematic computing
    # fields on axis. We can equivalently set the initial position to:
    elec.x = -lambda0*np.ones_like(elec.x)
    elec.y = -lambda0*np.ones_like(elec.y)
    elec.z = -lambda0*np.ones_like(elec.z)
    # Half step backward:
    halfstepback.step(1)

    # (2) MAIN SIMULATION:
    sim = Simulation(Nz, zmax, Nr, rmax, Nm, dt,
                     initialize_ions=False, zmin=zmin,
                     use_cuda=use_cuda,
                     boundaries={'z': 'periodic', 'r': 'reflective'})
    # Add electrons
    sim.ptcl = []
    elec2 = sim.add_new_species(q=-e, m=m_e, n=n, p_rmax=p_rmax,
                                p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
                                ux_m=elec.ux[0],
                                uy_m=elec.uy[0],
                                uz_m=elec.uz[0])
    # Activate spin tracking
    elec2.activate_spin_tracking(sz_m=sz_i, anom=anom)
    # And manually overwrite the spin array
    print(elec.spin_tracker.sx.shape)
    elec2.spin_tracker.sx.fill(elec.spin_tracker.sx[0])
    elec2.spin_tracker.sy.fill(elec.spin_tracker.sy[0])
    elec2.spin_tracker.sz.fill(elec.spin_tracker.sz[0])

    # Add the external fields
    sim.external_fields = [
        ExternalField(laser_func, 'Ex', a0*m_e*c**2*k0/e, lambda0),
        ExternalField(laser_func, 'By', a0*m_e*c*k0/e, lambda0)]

    # Set positions of all particles to origin
    elec2.x = -lambda0*np.ones_like(elec2.x)
    elec2.y = -lambda0*np.ones_like(elec2.y)
    elec2.z = -lambda0*np.ones_like(elec2.z)

    # Prepare the arrays for the time history of the pusher
    # Nptcl = sim.ptcl[0].Ntot
    x = np.zeros(N_step)
    y = np.zeros(N_step)
    z = np.zeros(N_step)
    ux = np.zeros(N_step)
    uy = np.zeros(N_step)
    uz = np.zeros(N_step)
    sx = np.zeros(N_step)
    sy = np.zeros(N_step)
    sz = np.zeros(N_step)

    # Push the particles over N_step and record the corresponding history
    j = 0  # which particle we select is irrelevant
    for i in range(N_step):
        # Record the history
        x[i] = elec2.x[j]
        y[i] = elec2.y[j]
        z[i] = elec2.z[j]
        ux[i] = elec2.ux[j]
        uy[i] = elec2.uy[j]
        uz[i] = elec2.uz[j]
        sx[i] = elec2.spin_tracker.sx[j]
        sy[i] = elec2.spin_tracker.sy[j]
        sz[i] = elec2.spin_tracker.sz[j]
        # Take a simulation step
        sim.step(1, show_progress=False)

    # (3) COMPUTE THE ANALYTIC SOLUTIONS

    # time - use half steps here for easier comparison
    t = sim.dt * np.arange(N_step )
    # Setup arrays for analytic solutions, for momentum and spin
    ux_analytic = np.zeros(N_step)
    uz_analytic = np.zeros(N_step)
    sx_analytic = np.zeros(N_step)
    sz_analytic = np.zeros(N_step)

    for i in range(N_step):
        # phase
        phi = k0*(c * t[i] - z[i])
        # conserved quantity in plane wave propagating along z (where u_0=uz_i)
        delta = math.sqrt(1 + uz_i ** 2) - uz_i
        # Calculate analytic solutions
        a0sinphi = a0 * np.sin(phi + lambda0 / 4)
        ux_analytic[i] = a0sinphi
        uz_analytic[i] = uz_i + 0.5 * a0sinphi ** 2 / delta
        I = anom*a0sinphi + 2 * np.arctan(a0sinphi / (1+delta))
        sz_analytic[i] = sz_i * np.cos(I) + sx_i * np.sin(I)
        sx_analytic[i] = sx_i * np.cos(I) - sz_i * np.sin(I)

    # Show the results
    if show is False:
        # Automatically check that the fields agree,
        # to an absolute tolerance
        atol = 0.01
        rtol = 1e-3
        # Interpolate to shift the analytic result by half time-step
        ux_anal_shifted = np.interp(t[1:]-dt*0.5, t, ux_analytic)
        assert np.allclose( ux_anal_shifted, ux[1:], atol=atol, rtol=rtol )
        print(f'The momentum ux agrees with the theory to atol={atol},\n' +
              'over the whole duration of the simulation.')
        sx_anal_shifted = np.interp(t[1:] - dt * 0.5, t, sx_analytic)
        assert np.allclose(sx_anal_shifted, sx[1:], atol=atol, rtol=rtol)
        print(f'The spin sx agrees with the theory to atol={atol},\n' +
              'over the whole duration of the simulation.')
    else:
        # Show the images to the user
        import matplotlib.pyplot as plt

        def add_gridlines():
            plt.grid(True, which='major', color='#666666',
                     linestyle='-', alpha=0.8)
            plt.minorticks_on()
            plt.grid(True, which='minor', color='#999999',
                     linestyle='-', alpha=0.2)
            plt.tight_layout()

        # Plot momentum
        plt.figure(figsize=(13, 5))
        plt.subplot(221)
        plt.plot(t/T, ux_analytic, '.-')
        plt.plot((t-dt/2)/T, ux, 'o')
        plt.ylabel('$u_x$')
        add_gridlines()
        plt.tick_params(axis='x', labelbottom=False)

        plt.subplot(222)
        plt.plot(t/T, uz_analytic, '.-')
        plt.plot((t-dt/2)/T, uz, 'o')
        plt.xlabel('$t/T$')
        plt.ylabel('$u_z$')
        add_gridlines()
        plt.legend(['Analytic', 'FB-PIC'])

        # Plot spin
        plt.subplot(223)
        plt.plot(t/T, sx_analytic, '.-')
        plt.plot((t-dt/2)/T, sx, 'o' )
        plt.ylabel('$s_x$')
        add_gridlines()
        plt.tick_params(axis='x', labelbottom=False)

        plt.subplot(224)
        plt.plot(t/T, sz_analytic, '.-')
        plt.plot((t-dt/2)/T, sz, 'o')
        plt.xlabel('$t/T$')
        plt.ylabel('$s_z$')
        add_gridlines()
        plt.legend(['Analytic', 'FB-PIC'])
        plt.show()

def laser_func(F, x, y, z, t, amplitude, length_scale ):
    """
    Function to be called at each timestep on the particles
    """
    return F - amplitude * math.cos(2*np.pi*(c*t-z)/length_scale )


def test_external_fields_lab(show=False):
    """Function that is run by py.test, when doing `python setup.py test`"""
    run_external_laser_field_simulation(show)


if __name__ == '__main__' :
    test_external_fields_lab(show=False)
