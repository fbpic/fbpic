# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file tests the whole PIC-Cycle by simulating a
linear, laser-driven plasma wakefield and comparing
it to the analytical solution.

Usage :
-----
from the top-level directory of FBPIC run
$ python tests/test_linear_wakefield.py

Theory:
-------
This test considers a laser of the form
$$ \vec{a} = a_0 e^{-(xi-xi_0)^2/(c\tau)^2}\vec{f}(r, \theta) $$
where $f$ represents the transverse profile of the laser, and is either
a radially polarized annular beam, or a linear polarized Laguerre-Gauss pulse.

Then, in the linear regime, the pseudo-potential is given by:
$$ \psi = \frac{k_p}{2}\int^xi_{-\infty} \langle \vec{a}^2 \rangle
\sin(kp(xi-xi'))dxi' $$
$$ \psi = \frac{k_p a_0^2}{4} f^2(r, \theta)\left[ \int^xi_{-\infty}
e^{-2(xi-xi_0)^2/(c\tau)^2}\sin(kp(xi-xi'))dxi'\right] $$
$$ E_z = \frac{m c^2 k_p^2 a_0^2}{4e} f^2(r, \theta)\left[ \int^xi_{-\infty}
e^{-2(xi-xi_0)^2/(c\tau)^2}\cos(kp(xi-xi'))dxi'\right] $$
$$ E_r = -\frac{m c^2 k_p a_0^2}{4e} \partial_r f^2(r, \theta) \left[ \int^
xi_{-\infty} e^{-2(xi-xi_0)^2/(c\tau)^2}\sin(kp(xi-xi'))dxi'\right] $$
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_e, epsilon_0
from scipy.integrate import quad
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser_pulse, \
    GaussianLaser, LaguerreGaussLaser

from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic

# ---------------------------
# Analytical solution
# ---------------------------

def kernel_Ez( xi0, xi) :
    """Longitudinal integration kernel for Ez"""
    return( np.cos( kp*(xi-xi0) )*np.exp( -2*(xi - z0)**2/ctau**2 ) )

def kernel_Er( xi0, xi) :
    """Integration kernel for Er"""
    return( np.sin( kp*(xi-xi0) )*np.exp( -2*(xi - z0)**2/ctau**2 ) )

def Ez( z, r, t) :
    """
    Get the 2d Ez field

    Parameters
    ----------
    z, r : 1darray
    t : float
    """
    Nz = len(z)
    window_zmax = z.max()

    # Longitudinal profile of the wakefield
    long_profile = np.zeros(Nz)
    for iz in range(Nz):
        long_profile[iz] = quad( kernel_Ez, z[iz]-c*t, window_zmax-c*t,
                        args = ( z[iz]-c*t,), limit=30 )[0]
    # Transverse profile
    if Nm == 2:
        trans_profile = np.exp( -2*r**2/w0**2 )

    # Combine longitudinal and transverse profile
    ez = m_e*c**2*kp**2*a0**2/(4.*e) * \
        trans_profile[np.newaxis, :] * long_profile[:, np.newaxis]

    return( ez )

def Er( z, r, t) :
    """
    Get the 2d Ez field

    Parameters
    ----------
    z, r : 1darray
    t : float
    """
    Nz = len(z)
    window_zmax = z.max()

    # Longitudinal profile of the wakefield
    long_profile = np.zeros(Nz)
    for iz in range(Nz):
        long_profile[iz] = quad( kernel_Er, z[iz]-c*t, window_zmax-c*t,
                        args = (z[iz]-c*t,), limit=200 )[0]
    # Transverse profile: gradient of transverse intensity
    if Nm == 2:
        trans_profile = -4*r/w0**2 * np.exp( -2*r**2/w0**2 )

    # Combine longitudinal and transverse profile
    er = m_e*c**2*kp*a0**2/(4.*e) * \
        trans_profile[np.newaxis, :] * long_profile[:, np.newaxis]

    return( er )

# ---------------------------
# Comparison plots
# ---------------------------

def compare_wakefields(Ez_analytic, Er_analytic, grid):
    """
    Draws a series of plots to compare the analytical and theoretical results
    """
    # Get extent from grid object
    extent = np.array([ grid.zmin-0.5*grid.dz, grid.zmax+0.5*grid.dz,
                        -0.5*grid.dr, grid.rmax + 0.5*grid.dr ])
    # Rescale extent to microns
    extent = extent/1.e-6

    # Create figure
    plt.figure(figsize=(8,7))

    plt.suptitle('Analytical vs. PIC Simulation for Ez and Er')

    # Plot analytic Ez in 2D
    plt.subplot(321)
    plt.imshow(Ez_analytic.T, extent=extent, origin='lower',
        aspect='auto', interpolation='nearest')
    plt.xlabel('z')
    plt.ylabel('r')
    cb = plt.colorbar()
    cb.set_label('Ez')
    plt.title('Analytical Ez')

    # Plot analytic Er in 2D
    plt.subplot(322)
    plt.imshow(Er_analytic.T, extent=extent, origin='lower',
        aspect='auto', interpolation='nearest')
    plt.xlabel('z')
    plt.ylabel('r')
    cb = plt.colorbar()
    plt.title('Analytical Er')

    # Plot simulated Ez in 2D
    plt.subplot(323)
    plt.imshow(grid.Ez.real.T, extent=extent, origin='lower',
        aspect='auto', interpolation='nearest')
    plt.xlabel('z')
    plt.ylabel('r')
    cb = plt.colorbar()
    cb.set_label('Ez')
    plt.title('Simulated Ez')

    # Get z
    z = grid.z

    # Plot simulated Er in 2D
    plt.subplot(324)
    plt.imshow(grid.Er.real.T, extent=extent, origin='lower',
        aspect='auto', interpolation='nearest')
    plt.xlabel('z')
    plt.ylabel('r')
    cb = plt.colorbar()
    cb.set_label('Er')
    plt.title('Simulated Er')

    # Plot lineouts of Ez (simulation and analytical solution)
    plt.subplot(325)
    plt.plot(1.e6*z, grid.Ez[:,0].real,
        color = 'b', label = 'Simulation')
    plt.plot(1.e6*z, Ez_analytic[:,0], color = 'r', label = 'Analytical')
    plt.xlabel('z')
    plt.ylabel('Ez')
    plt.legend(loc=0)
    plt.title('PIC vs. Analytical - On-axis lineout of Ez')

    # Plot lineouts of Er (simulation and analytical solution)
    plt.subplot(326)
    plt.plot(1.e6*z, grid.Er[:,5].real,
        color = 'b', label = 'Simulation')
    plt.plot(1.e6*z, Er_analytic[:,5], color = 'r', label = 'Analytical')
    plt.xlabel('z')
    plt.ylabel('Er')
    plt.legend(loc=0)
    plt.title('PIC vs. Analytical - Off-axis lineout of Er')

    # Show plots
    plt.show()

def compare_fields(sim) :
    """
    Gather the results and compare them with the analytical predicitions
    """
    gathered_grid = sim.comm.gather_grid(sim.fld.interp[0])
    if sim.comm.rank==0 :
        z = gathered_grid.z
        r = gathered_grid.r

        # Analytical solution
        print( 'Calculate analytical solution for Ez' )
        ez = Ez(z, r, sim.time)
        print( 'Done...\n' )

        print( 'Calculate analytical solution for Er' )
        er = Er(z, r, sim.time)
        print('Done...\n')

        compare_wakefields(ez, er, gathered_grid)

# ---------------------------
# Setup simulation & parameters
# ---------------------------
use_cuda = True

# The simulation box
Nz = 800         # Number of gridpoints along z
zmax = 40.e-6    # Length of the box along z (meters)
Nr = 60          # Number of gridpoints along r
rmax = 60.e-6    # Length of the box along r (meters)
Nm = 2           # Number of modes used
# The simulation timestep
dt = zmax/Nz/c   # Timestep (seconds)
# The number of steps
N_step = 1500

# The particles
p_zmin = 39.e-6  # Position of the beginning of the plasma (meters)
p_zmax = 41.e-6  # Position of the end of the plasma (meters)
p_rmin = 0.      # Minimal radial position of the plasma (meters)
p_rmax = 50.e-6  # Maximal radial position of the plasma (meters)
n_e = 8.e24      # Density (electrons.meters^-3)
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r
p_nt = 2*Nm      # Number of particles per cell along theta

# The laser
a0 = 0.01        # Laser amplitude
w0 = 20.e-6       # Laser waist
ctau = 6.e-6     # Laser duration
tau = ctau/c
z0 = 27.e-6      # Laser centroid

# Diagnostics
write_fields = False
write_particles = False
diag_period = 5

# Plasma and laser wavenumber
kp = 1./c * np.sqrt( n_e * e**2 / (m_e * epsilon_0) )
k0 = 2*np.pi/0.8e-6

# Initialize the simulation object
sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
                  p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
                  use_cuda=use_cuda, boundaries='open' )

# Create the relevant laser profile
if Nm == 1:
    # Build a radially-polarized pulse from 2 Laguerre-Gauss profiles
    profile = LaguerreGaussLaser( 0, 1, 0.5*a0, w, tau, z0,
                lambda0=lambda0, theta_pol=0., theta0=0. ) \
            + LaguerreGaussLaser( 0, 1, 0.5*a0, w, tau, z0,
                lambda0=lambda0, theta_pol=np.pi/2, theta0=np.pi/2 )
elif Nm == 2:
    profile = GaussianLaser(a0=a0, waist=w0, tau=tau, z0=z0 )
elif Nm == 3:
    profile = LaguerreGaussLaser(0, 1, a0=a0, waist=w0, tau=tau, z0=z0 )

add_laser_pulse( sim, profile )

# Configure the moving window
sim.set_moving_window( v=c )

# Add diagnostics
if write_fields:
    sim.diags.append( FieldDiagnostic(diag_period, sim.fld, sim.comm ) )
    sim.diags.append( FieldDiagnostic(diag_period, sim.fld, None,
                                      write_dir='proc%d' %sim.comm.rank) )
if write_particles:
    sim.diags.append( ParticleDiagnostic(diag_period,
                    {'electrons': sim.ptcl[0]}, sim.comm ) )

if __name__ == '__main__' :
    # Prevent current correction for MPI simulation
    if sim.comm.size > 1:
        correct_currents=False
    else:
        correct_currents=True

    # Run the simulation
    sim.step(N_step, correct_currents=correct_currents)

    # Plot the fields
    compare_fields(sim)
