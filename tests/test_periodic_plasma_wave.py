# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It tests the global PIC loop by launching a linear periodic plasma wave,
and letting it evolve in time. Its fields are then compared with theory.
This tests is run both for linear and cubic shapes.

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
$$ \phi =
\epsilon \,\frac{m c^2}{e}
    \exp\left(-\frac{r^2}{w_0^2}\right) \sin(k_0 z) \sin(\omega_p t)
+ \epsilon_1 \,\frac{m c^2}{e} \frac{2\,r\cos(\theta)}{w_0}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \sin(\omega_p t)
+ \epsilon_2 \,\frac{m c^2}{e} \frac{4\,r^2\cos(2\theta)}{w_0^2}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \sin(\omega_p t)$$

$$ E_r = -\partial_r \phi =
\epsilon \,\frac{mc^2}{e}\frac{2\,r}{w_0^2}
    \exp\left(-\frac{r^2}{w_0^2}\right) \sin(k_0 z) \sin(\omega_p t)
- \epsilon_1 \,\frac{m c^2}{e} \frac{2\cos(\theta)}{w_0}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \sin(\omega_p t)
+ \epsilon_1 \,\frac{m c^2}{e} \frac{4\,r^2\cos(\theta)}{w_0^3}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \sin(\omega_p t)
- \epsilon_2 \,\frac{m c^2}{e} \frac{8\,r\cos(2\theta)}{w_0^2}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \sin(\omega_p t)
+ \epsilon_2 \,\frac{m c^2}{e} \frac{8\,r^3\cos(2\theta)}{w_0^4}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \sin(\omega_p t) $$

$$ E_\theta = - \frac{1}{r} \partial_\theta \phi =
 \epsilon_1 \,\frac{m c^2}{e} \frac{2\,\sin(\theta)}{w_0}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \sin(\omega_p t)
+ \epsilon_2 \,\frac{m c^2}{e} \frac{8\,r\sin(2\theta)}{w_0^2}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \sin(\omega_p t) $$

$$ E_x = \cos(\theta)E_r - \sin(\theta)E_\theta =
\epsilon \,\frac{mc^2}{e}\frac{2\,x}{w_0^2}
    \exp\left(-\frac{r^2}{w_0^2}\right) \sin(k_0 z) \sin(\omega_p t)
- \epsilon_1 \,\frac{m c^2}{e} \frac{2}{w_0}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \sin(\omega_p t)
+ \epsilon_1 \,\frac{m c^2}{e} \frac{4\,x^2}{w_0^3}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \sin(\omega_p t)
- \epsilon_2 \,\frac{m c^2}{e} \frac{8\,x}{w_0^2}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \sin(\omega_p t)
+ \epsilon_2 \,\frac{m c^2}{e} \frac{8\,x(x^2-y^2)}{w_0^4}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \sin(\omega_p t) $$

$$ E_y = \sin(\theta)E_r + \cos(\theta)E_\theta =
\epsilon \,\frac{mc^2}{e}\frac{2\,y}{w_0^2}
    \exp\left(-\frac{r^2}{w_0^2}\right) \sin(k_0 z) \sin(\omega_p t)
+ \epsilon_1 \,\frac{m c^2}{e} \frac{4\,x y}{w_0^3}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \sin(\omega_p t)
+ \epsilon_2 \,\frac{m c^2}{e} \frac{8\,y}{w_0^2}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \sin(\omega_p t)
+ \epsilon_2 \,\frac{m c^2}{e} \frac{8\,y(x^2-y^2)}{w_0^4}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \sin(\omega_p t) $$

$$ E_z = -\partial_z \phi =
 - \epsilon \,\frac{mc^2}{e} k_0
    \exp\left(-\frac{r^2}{w_0^2}\right) \cos(k_0 z) \sin(\omega_p t)
 - \epsilon_1 \,\frac{m c^2}{e} \frac{2\,r\cos(\theta)}{w_0} k_0
    \exp\left(-\frac{r^2}{w_0^2}\right)\cos(k_0 z) \sin(\omega_p t)
 - \epsilon_2 \, \frac{m c^2}{e} \frac{4\,r^2\cos(\theta)}{w_0^2} k_0
    \exp\left(-\frac{r^2}{w_0^2}\right)\cos(k_0 z) \sin(\omega_p t) $$

$$ v_x/c =
 \epsilon \, \frac{c}{\omega_p} \, \frac{2\,x}{w_0^2}
    \exp\left(-\frac{r^2}{w_0^2}\right) \sin(k_0 z) \cos(\omega_p t)
 - \epsilon_1 \,\frac{c}{\omega_p} \frac{2}{w_0}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \cos(\omega_p t)
 + \epsilon_1 \,\frac{c}{\omega_p} \frac{4\,x^2}{w_0^3})
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \cos(\omega_p t)
- \epsilon_2 \,\frac{c}{\omega_p} \frac{8\,x}{w_0^2}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \cos(\omega_p t)
+ \epsilon_2 \,\frac{c}{\omega_p} \frac{8\,x(x^2-y^2)}{w_0^4}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \cos(\omega_p t) $$

$$ v_y/c =
 \epsilon \, \frac{c}{\omega_p} \, \frac{2\,y}{w_0^2}
    \exp\left(-\frac{r^2}{w_0^2}\right) \sin(k_0 z) \cos(\omega_p t)
 + \epsilon_1 \,\frac{c}{\omega_p} \frac{4\,x y}{w_0^3})
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \cos(\omega_p t)
+ \epsilon_2 \,\frac{c}{\omega_p} \frac{8\,y}{w_0^2}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \cos(\omega_p t)
+ \epsilon_2 \,\frac{c}{\omega_p} \frac{8\,y(x^2-y^2)}{w_0^4}
    \exp\left(-\frac{r^2}{w_0^2}\right)\sin(k_0 z) \cos(\omega_p t) $$

$$ v_z/c =
 - \epsilon \, \frac{c}{\omega_p} \, k_0
    \exp\left(-\frac{r^2}{w_0^2}\right) \cos(k_0 z) \cos(\omega_p t)
 - \epsilon_1 \,\frac{c}{\omega_p} \frac{2\,x}{w_0} k_0
    \exp\left(-\frac{r^2}{w_0^2}\right) \cos(k_0 z) \cos(\omega_p t)
 - \epsilon_2 \,\frac{c}{\omega_p} \frac{4\,(x^2-y^2)}{w_0^2} k_0
    \exp\left(-\frac{r^2}{w_0^2}\right) \cos(k_0 z) \cos(\omega_p t)$$

where $\epsilon$ is the dimensionless amplitude of the mode 0 and
$\epsilon_1$, $\epsilon_2$ are the dimensionless amplitudes of modes 1 and 2.
"""
import numpy as np
from scipy.constants import c, e, m_e, epsilon_0
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.utils.cuda import GpuMemoryManager
from fbpic.fields import Fields

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
Nm = 3           # Number of modes used
n_order = 16     # Order of the finite stencil
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
p_nt = 8         # Number of particles per cell along theta

# The plasma wave
epsilon = 0.001    # Dimensionless amplitude of the wave in mode 0
epsilon_1 = 0.001  # Dimensionless amplitude of the wave in mode 1
epsilon_2 = 0.001  # Dimensionless amplitude of the wave in mode 2
epsilons = [ epsilon, epsilon_1, epsilon_2 ]
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

def test_periodic_plasma_wave_linear_shape( show=False ):
    "Function that is run by py.test, when doing `python setup.py test"
    simulate_periodic_plasma_wave( 'linear', show=show )

def test_periodic_plasma_wave_cubic_shape( show=False ):
    "Function that is run by py.test, when doing `python setup.py test"
    simulate_periodic_plasma_wave( 'cubic', show=show )

def simulate_periodic_plasma_wave( particle_shape, show=False ):
    "Simulate a periodic plasma wave and check its fields"

    # Initialization of the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
                  p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr,
                  p_nt, n_e, n_order=n_order, use_cuda=use_cuda,
                  particle_shape=particle_shape )

    # Save the initial density in spectral space, and consider it
    # to be the density of the (uninitialized) ions
    # (Move the simulation to GPU if needed, for this step)
    with GpuMemoryManager(sim):
        sim.deposit('rho_prev', exchange=True)
        sim.fld.spect2interp('rho_prev')
    rho_ions = [ ]
    for m in range(len(sim.fld.interp)):
        rho_ions.append( -sim.fld.interp[m].rho.copy() )

    # Impart velocities to the electrons
    # (The electrons are initially homogeneous, but have an
    # intial non-zero velocity that develops into a plasma wave)
    impart_momenta( sim.ptcl[0], epsilons, k0, w0, wp )

    # Run the simulation
    sim.step( N_step, correct_currents=True )

    # Plot the results and compare with analytical theory
    compare_fields( sim, show )
    # Test check that div(E) - rho = 0 (directly in spectral space)
    check_charge_conservation( sim, rho_ions )


# -----------------------------------------
# Analytical solutions for the plasma wave
# -----------------------------------------

def Er( z, r, epsilons, k0, w0, wp, t) :
    """
    Return the radial electric field as an array
    of the same length as z and r, in the half-plane theta=0
    """
    Er_array = \
        epsilons[0] * m_e*c**2/e * 2*r/w0**2 * \
            np.exp( -r**2/w0**2 ) * np.sin( k0*z ) * np.sin( wp*t ) \
        - epsilons[1] * m_e*c**2/e * 2/w0 * \
            np.exp( -r**2/w0**2 ) * np.sin( k0*z ) * np.sin( wp*t ) \
        + epsilons[1] * m_e*c**2/e * 4*r**2/w0**3 * \
            np.exp( -r**2/w0**2 ) * np.sin( k0*z ) * np.sin( wp*t ) \
        - epsilons[2] * m_e*c**2/e * 8*r/w0**2 * \
            np.exp( -r**2/w0**2 ) * np.sin( k0*z ) * np.sin( wp*t ) \
        + epsilons[2] * m_e*c**2/e * 8*r**3/w0**4 * \
            np.exp( -r**2/w0**2 ) * np.sin( k0*z ) * np.sin( wp*t )
    return( Er_array )

def Ez( z, r, epsilons, k0, w0, wp, t) :
    """
    Return the longitudinal electric field as an array
    of the same length as z and r, in the half-plane theta=0
    """
    Ez_array = \
        - epsilons[0] * m_e*c**2/e * k0 * \
            np.exp( -r**2/w0**2 ) * np.cos( k0*z ) * np.sin( wp*t ) \
        - epsilons[1] * m_e*c**2/e * k0 * 2*r/w0 * \
            np.exp( -r**2/w0**2 ) * np.cos( k0*z ) * np.sin( wp*t ) \
        - epsilons[2] * m_e*c**2/e * k0 * 4*r**2/w0**2 * \
            np.exp( -r**2/w0**2 ) * np.cos( k0*z ) * np.sin( wp*t )
    return( Ez_array )

def ux( z, r, x, y, epsilons, k0, w0, wp, t) :
    """
    Return the radial normalized velocity as an array
    of the same length as z, r, x, y
    """
    ux_array = \
        epsilons[0] * c/wp * 2*x/w0**2 * \
            np.exp( -r**2/w0**2 ) * np.sin( k0*z ) * np.cos( wp*t ) \
        - epsilons[1] * c/wp * 2/w0 * \
            np.exp( -r**2/w0**2 ) * np.sin( k0*z ) * np.cos( wp*t ) \
        + epsilons[1] * c/wp * 4*x**2/w0**3 * \
            np.exp( -r**2/w0**2 ) * np.sin( k0*z ) * np.cos( wp*t ) \
        - epsilons[2] * c/wp * 8*x/w0**2 * \
            np.exp( -r**2/w0**2 ) * np.sin( k0*z ) * np.cos( wp*t ) \
        + epsilons[2] * c/wp * 8*x*(x**2-y**2)/w0**4 * \
            np.exp( -r**2/w0**2 ) * np.sin( k0*z ) * np.cos( wp*t )
    return( ux_array )

def uy( z, r, x, y, epsilons, k0, w0, wp, t) :
    """
    Return the radial normalized velocity as an array
    of the same length as z, r, x, y
    """
    uy_array = \
        epsilons[0] * c/wp * 2*y/w0**2 * \
            np.exp( -r**2/w0**2 ) * np.sin( k0*z ) * np.cos( wp*t ) \
        + epsilons[1] * c/wp * 4*x*y/w0**3 * \
            np.exp( -r**2/w0**2 ) * np.sin( k0*z ) * np.cos( wp*t ) \
        + epsilons[2] * c/wp * 8*y/w0**2 * \
            np.exp( -r**2/w0**2 ) * np.sin( k0*z ) * np.cos( wp*t ) \
        + epsilons[2] * c/wp * 8*y*(x**2-y**2)/w0**4 * \
            np.exp( -r**2/w0**2 ) * np.sin( k0*z ) * np.cos( wp*t )
    return( uy_array )

def uz( z, r, x, y, epsilons, k0, w0, wp, t) :
    """
    Return the longitudinal normalized velocity as an array
    of the same length as z and r
    """
    uz_array = \
        - epsilons[0] * c/wp * k0 * \
            np.exp( -r**2/w0**2 ) * np.cos( k0*z ) * np.cos( wp*t ) \
        - epsilons[1] * c/wp * k0 * 2*x/w0 * \
            np.exp( -r**2/w0**2 ) * np.cos( k0*z ) * np.cos( wp*t ) \
        - epsilons[2] * c/wp * k0 * 4*(x**2-y**2)/w0**2 * \
            np.exp( -r**2/w0**2 ) * np.cos( k0*z ) * np.cos( wp*t )
    return( uz_array )

# --------------------------------------------
# Functions for initialization of the momenta
# --------------------------------------------

def impart_momenta( ptcl, epsilons, k0, w0, wp) :
    """
    Modify the momenta of the input particle object,
    so that they correspond to a plasma wave at t=0
    """
    r = np.sqrt( ptcl.x**2 + ptcl.y**2 )
    # Impart the momenta
    ptcl.ux = ux(ptcl.z, r, ptcl.x, ptcl.y, epsilons, k0, w0, wp, 0)
    ptcl.uy = uy(ptcl.z, r, ptcl.x, ptcl.y, epsilons, k0, w0, wp, 0)
    ptcl.uz = uz(ptcl.z, r, ptcl.x, ptcl.y, epsilons, k0, w0, wp, 0)
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
    # Create a global field object across all subdomains, and copy the fields
    global_Nz, _ = sim.comm.get_Nz_and_iz(
            local=False, with_damp=False, with_guard=False )
    global_zmin, global_zmax = sim.comm.get_zmin_zmax(
            local=False, with_damp=False, with_guard=False )
    global_fld = Fields( global_Nz, global_zmax,
            sim.fld.Nr, sim.fld.rmax, sim.fld.Nm, sim.fld.dt,
            zmin=global_zmin, n_order=sim.fld.n_order, use_cuda=False)
    # Gather the fields of the interpolation grid
    for m in range(sim.fld.Nm):
        # Gather E
        for field in ['Er', 'Et', 'Ez' ]:
            local_array = getattr( sim.fld.interp[m], field )
            gathered_array = sim.comm.gather_grid_array( local_array )
            setattr( global_fld.interp[m], field, gathered_array )
        # Gather rho
        global_fld.interp[m].rho = \
            sim.comm.gather_grid_array( sim.fld.interp[m].rho + rho_ions[m] )

    # Loop over modes and check charge conservation in spectral space
    if sim.comm.rank == 0:
        global_fld.interp2spect('E')
        global_fld.interp2spect('rho_prev')
        for m in range( global_fld.Nm ):
            spect = global_fld.spect[m]
            # Calculate div(E) in spectral space
            divE = spect.kr*( spect.Ep - spect.Em ) + 1.j*spect.kz*spect.Ez
            # Calculate rho/epsilon_0 in spectral space
            rho_eps0 = spect.rho_prev/epsilon_0
            # Calculate relative RMS error
            rel_err = np.sqrt( np.sum(abs(divE - rho_eps0)**2) \
                / np.sum(abs(rho_eps0)**2) )
            print('Relative error on divE in mode %d: %e' %(m, rel_err) )
            assert rel_err < 1.e-11

def compare_fields( sim, show ) :
    """
    Gathers the fields and compare them with the analytical theory
    """
    # Get the fields in the half-plane theta=0 (Sum mode 0 and mode 1)
    gathered_grids = [ sim.comm.gather_grid(sim.fld.interp[m]) \
                           for m in range(Nm) ]

    if sim.comm.rank == 0:
        rgrid = gathered_grids[0].r
        zgrid = gathered_grids[0].z
        # Check the Ez field
        Ez_simulated = gathered_grids[0].Ez.real
        for m in range(1,Nm):
            Ez_simulated += 2*gathered_grids[m].Ez.real
        check_E_field( Ez_simulated, rgrid, zgrid, epsilons,
                    k0, w0, wp, sim.time, field='Ez', show=show )
        # Check the Er field
        Er_simulated = gathered_grids[0].Er.real
        for m in range(1,Nm):
            Er_simulated += 2*gathered_grids[m].Er.real
        check_E_field( Er_simulated, rgrid, zgrid, epsilons,
                    k0, w0, wp, sim.time, field='Er', show=show )

def check_E_field( E_simulation, rgrid, zgrid, epsilons,
                    k0, w0, wp, t, field='Ez', show=False ):
    """
    Compare the longitudinal and radial field with the
    simulation.

    If show=True : show the plots to the user
    If show=False : compare the 2D maps automatically
    """
    # 2D maps of the field
    r, z = np.meshgrid( rgrid, zgrid )
    if field == 'Ez' :
        E_analytical = Ez( z, r, epsilons, k0, w0, wp, t )
    if field == 'Er' :
        E_analytical = Er( z, r, epsilons, k0, w0, wp, t )

    if show is False:
        # Automatically check that the fields agree,
        # to an absolute tolerance
        atol = 1.1e6
        rtol = 2e-2
        assert np.allclose( E_analytical, E_simulation, atol=atol, rtol=rtol )
        print('The field %s agrees with the theory to %e,\n' %(field, atol) + \
               'over the whole simulation box.'  )
    else:
        # Show the images to the user
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,10))
        plt.suptitle('%s field' %field)

        extent = 1.e6*np.array([zgrid.min(), zgrid.max(),
                                rgrid.min(), rgrid.max()])
        plt.subplot(221)
        plt.imshow( E_analytical.T[::-1], extent=extent, aspect='auto' )
        plt.colorbar()
        plt.title('Analytical')
        plt.xlabel('z (microns)')
        plt.ylabel('r (microns)')
        plt.subplot(222)
        plt.imshow( E_simulation.T[::-1], extent=extent, aspect='auto' )
        plt.colorbar()
        plt.title('Simulated')
        plt.xlabel('z (microns)')
        plt.ylabel('r (microns)')

        # On-axis plot
        plt.subplot(223)
        plt.plot( 1.e6*zgrid, E_analytical[:,0], label='Analytical' )
        plt.plot( 1.e6*zgrid, E_simulation[:,0], label='Simulated' )
        plt.xlabel('z (microns)')
        plt.ylabel('Ez')
        plt.legend(loc=0)
        plt.title('Field on axis')

        # Plot at a radius w0
        plt.subplot(224)
        ir = np.argmin( abs(rgrid-w0) )
        plt.plot( 1.e6*zgrid, E_analytical[:,ir], label='Analytical' )
        plt.plot( 1.e6*zgrid, E_simulation[:,ir], label='Simulated' )
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
    test_periodic_plasma_wave_linear_shape(show=show)
    test_periodic_plasma_wave_cubic_shape(show=show)
