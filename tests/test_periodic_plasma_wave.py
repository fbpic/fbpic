# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It tests the global PIC loop by launching a linear periodic plasma wave,
and letting it evolve in time. Its fields are then compared with theory.

No moving window is involved, and periodic conditions are userd.

Usage:
------
In order to show the images of the laser, and manually check the
agreement between the simulation and the theory:
(except when setting show to False in the parameters below)
$ python tests/test_periodic_plasma_wave.py  # Single-proc simulation
$ mpirun -np 2 python tests/test_periodic_plasma_wave.py # Two-proc simulation

In order to let Python check the agreement between the curve without
having to look at the plots
$ py.test -q tests/test_periodic_plasma_wave.py
or
$ python setup.py test

Theory:
-------

The fields are given by the analytical formulas :
$$ \phi = \epsilon \,\frac{m c^2}{e} \exp\left(-\frac{r^2}{w_0^2}\right)
\sin(k_0 z) \sin(\omega_p t)$$

$$ E_r = -\partial_r \phi = \epsilon \,\frac{mc^2}{e}\frac{2\,r}{w_0^2}
\exp\left(-\frac{r^2}{w_0^2}\right) \sin(k_0 z) \sin(\omega_p t) $$

$$ E_z = -\partial_z \phi = -\epsilon \,\frac{mc^2}{e}  k_0
\exp\left(-\frac{r^2}{w_0^2}\right) \cos(k_0 z) \sin(\omega_p t) $$

$$ v_r/c = \epsilon \, \frac{c}{\omega_p} \, \frac{2\,r}{w_0^2}
\exp\left(-\frac{r^2}{w_0^2}\right) \sin(k_0 z) \cos(\omega_p t) $$

$$ v_z/c = - \epsilon \, \frac{c}{\omega_p} \, k_0
\exp\left(-\frac{r^2}{w_0^2}\right) \cos(k_0 z) \cos(\omega_p t) $$

where $\epsilon$ is the dimensionless amplitude.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_e, epsilon_0
# Import the relevant structures in FBPIC
from fbpic.main import Simulation

# Parameters
# ----------
show = True      # Whether to show the comparison between simulation
                 # and theory to the user, or to automatically determine
                 # whether they agree.

use_cuda=True    # Whether to run with cuda

# The simulation box
Nz = 200         # Number of gridpoints along z
zmax = 40.e-6    # Length of the box along z (meters)
Nr = 64          # Number of gridpoints along r
rmax = 20.e-6    # Length of the box along r (meters)
Nm = 2           # Number of modes used
n_order = -1     # Order of the finite stencil
# The simulation timestep
dt = zmax/Nz/c   # Timestep (seconds)

# The particles
p_zmin = 0.e-6   # Position of the beginning of the plasma (meters)
p_zmax = 41.e-6  # Position of the end of the plasma (meters)
p_rmin = 0.      # Minimal radial position of the plasma (meters)
p_rmax = 18.e-6  # Maximal radial position of the plasma (meters)
n_e = 2.e24      # Density (electrons.meters^-3)
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r
p_nt = 4         # Number of particles per cell along theta

# The plasma wave
epsilon = 0.001  # Dimensionless amplitude of the wave
w0 = 5.e-6      # The transverse size of the plasma wave
N_periods = 3   # Number of periods in the box
# Calculated quantities
k0 = 2*np.pi/zmax*N_periods
wp = np.sqrt( n_e*e**2/(m_e*epsilon_0) )

# Run the simulation for 0.75 plasma period
N_step = int( 2*np.pi/(wp*dt)*0.75 )

# -------------
# Test function
# -------------

def test_periodic_plasma_wave(show=False):
    "Function that is run by py.test, when doing `python setup.py test`"

    # Initialization of the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
                  p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr,
                  p_nt, n_e, n_order=n_order, use_cuda=use_cuda )

    # Save the initial density in spectral space, and consider it
    # to be the density of the (uninitialized) ions
    sim.deposit('rho_prev')
    rho_ions = [ ]
    for m in range(len(sim.fld.spect)):
        rho_ions.append( -sim.fld.spect[m].rho_prev.copy() )

    # Impart velocities to the electrons
    # (The electrons are initially homogeneous, but have an
    # intial non-zero velocity that develops into a plasma wave)
    impart_momenta( sim.ptcl[0], epsilon, k0, w0, wp )

    # Choose whether to correct the currents
    if sim.comm.size == 1:
        correct_currents = True
    else:
        correct_currents = False

    # Run the simulation
    sim.step( N_step, correct_currents=correct_currents )

    # Test check that div(E) - rho = 0 (directly in spectral space)
    if correct_currents:
        check_charge_conservation( sim, rho_ions )
    # Plot the results and compare with analytical theory
    compare_fields( sim, show )

# -----------------------------------------
# Analytical solutions for the plasma wave
# -----------------------------------------

def Er( z, r, epsilon, k0, w0, wp, t) :
    """
    Return the radial electric field as an array
    of the same length as z and r
    """
    Er_array = epsilon * m_e*c**2/e * 2*r/w0**2 * np.exp( -r**2/w0**2 ) * \
      np.sin( k0*z ) * np.sin( wp*t )
    return( Er_array )

def Ez( z, r, epsilon, k0, w0, wp, t) :
    """
    Return the longitudinal electric field as an array
    of the same length as z and r
    """
    Ez_array = - epsilon * m_e*c**2/e * k0 * np.exp( -r**2/w0**2 ) * \
      np.cos( k0*z ) * np.sin( wp*t )
    return( Ez_array )

def ur( z, r, epsilon, k0, w0, wp, t) :
    """
    Return the radial normalized velocity as an array
    of the same length as z and r
    """
    ur_array = epsilon * c/wp * 2*r/w0**2 * np.exp( -r**2/w0**2 ) * \
      np.sin( k0*z ) * np.cos( wp*t )
    return( ur_array )

def uz( z, r, epsilon, k0, w0, wp, t) :
    """
    Return the longitudinal normalized velocity as an array
    of the same length as z and r
    """
    uz_array = - epsilon * c/wp * k0 * np.exp( -r**2/w0**2 ) * \
      np.cos( k0*z ) * np.cos( wp*t )
    return( uz_array )

# --------------------------------------------
# Functions for initialization of the momenta
# --------------------------------------------

def impart_momenta( ptcl, epsilon, k0, w0, wp) :
    """
    Modify the momenta of the input particle object,
    so that they correspond to a plasma wave at t=0
    """
    r = np.sqrt( ptcl.x**2 + ptcl.y**2 )
    # Impart the momenta
    ptcl.uz = uz( ptcl.z, r, epsilon, k0, w0, wp, 0 )
    Ur = ur( ptcl.z, r, epsilon, k0, w0, wp, 0 )
    ptcl.ux = Ur * ptcl.x/r
    ptcl.uy = Ur * ptcl.y/r
    # Get the corresponding inverse gamma
    ptcl.inv_gamma = 1./np.sqrt( 1 + ptcl.ux**2 + ptcl.uy**2 + ptcl.uz**2 )

# --------------------
# Diagnostic function
# --------------------

def check_charge_conservation( sim, rho_ions ):
    """
    Check that the relation div(E) - rho/epsilon_0 is satisfied, with a
    relative precision close to the machine precision (directly in spectral space)

    Parameters
    ----------
    sim: Simulation object
    rho_ions: list of 2d complex arrays (one per mode)
        The density of the ions (which are not explicitly present in the `sim`
        object, since they are motionless)
    """
    # Loop over modes
    for m in range( len(sim.fld.interp) ):
        spect = sim.fld.spect[m]
        # Calculate div(E) in spectral space
        divE = spect.kr * ( spect.Ep - spect.Em ) + 1.j * spect.kz * spect.Ez
        # Calculate rho/epsilon_0 in spectral space
        rho_eps0 = (spect.rho_prev + rho_ions[m])/epsilon_0
        # Calculate relative RMS error
        rel_err = np.sqrt( np.sum(abs(divE - rho_eps0)**2) \
            / np.sum(abs(rho_eps0)**2) )
        print('Relative error on divE in mode %d: %e' %(m, rel_err) )
        assert rel_err < 1.e-11

def compare_fields( sim, show ) :
    """
    Gathers the fields and compare them with the analytical theory
    """
    gathered_grid = sim.comm.gather_grid(sim.fld.interp[0])

    if sim.comm.rank == 0:
        # Check the Ez field
        check_E_field( gathered_grid, epsilon, k0, w0, wp,
                    sim.time, field='Ez', show=show )
        # Check the Er field
        check_E_field( gathered_grid, epsilon, k0, w0, wp,
                    sim.time, field='Er', show=show )

def check_E_field( interp, epsilon, k0, w0, wp, t, field='Ez', show=False ):
    """
    Compare the longitudinal and radial field with the
    simulation.

    If show=True : show the plots to the user
    If show=False : compare the 2D maps automatically
    """
    # 2D maps of the field
    r, z = np.meshgrid( interp.r, interp.z )
    if field == 'Ez' :
        E_analytical = Ez( z, r, epsilon, k0, w0, wp, t )
        E_simulation = interp.Ez.real
    if field == 'Er' :
        E_analytical = Er( z, r, epsilon, k0, w0, wp, t )
        E_simulation = interp.Er.real

    if show is False:
        # Automatically check that the fields agree,
        # to an absolute tolerance
        atol = 1e6
        rtol = 2e-2
        assert np.allclose( E_analytical, E_simulation, atol=atol, rtol=rtol )
        print('The field %s agrees with the theory to %e,\n' %(field, atol) + \
               'over the whole simulation box.'  )
    else:
        # Show the images to the user
        plt.figure(figsize=(8,10))
        plt.suptitle('%s field' %field)

        plt.subplot(221)
        plt.imshow( E_analytical.T[::-1], extent=1.e6*np.array(
        [interp.zmin, interp.zmax, interp.rmin, interp.rmax]), aspect='auto' )
        plt.colorbar()
        plt.title('Analytical')
        plt.xlabel('z (microns)')
        plt.ylabel('r (microns)')
        plt.subplot(222)
        plt.imshow( E_simulation.T[::-1], extent=1.e6*np.array(
        [interp.zmin, interp.zmax, interp.rmin, interp.rmax]), aspect='auto' )
        plt.colorbar()
        plt.title('Simulated')
        plt.xlabel('z (microns)')
        plt.ylabel('r (microns)')

        # On-axis plot
        plt.subplot(223)
        plt.plot( 1.e6*interp.z, E_analytical[:,0], label='Analytical' )
        plt.plot( 1.e6*interp.z, E_simulation[:,0].real, label='Simulated' )
        plt.xlabel('z (microns)')
        plt.ylabel('Ez')
        plt.legend(loc=0)
        plt.title('Field on axis')

        # Plot at a radius w0
        plt.subplot(224)
        ir = int(w0/interp.dr)
        plt.plot( 1.e6*interp.z, E_analytical[:,ir], label='Analytical' )
        plt.plot( 1.e6*interp.z, E_simulation[:,ir].real, label='Simulated' )
        plt.xlabel('z (microns)')
        plt.ylabel(field)
        plt.legend(loc=0)
        plt.title('Field off axis')

        plt.show()


# -------------------------
# Launching the simulation
# -------------------------

if __name__ == '__main__' :

    # Run the simulation and show the results to the user
    test_periodic_plasma_wave(show=show)
