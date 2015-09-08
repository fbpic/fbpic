"""
This file tests the global PIC loop by evolving a
linear periodic plasma wave, in time.

No moving window is involved.

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

# --------
# Imports
# --------

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_e, epsilon_0
# Import the relevant structures in FBPIC
from fbpic.main import Simulation, adapt_to_grid
from fbpic.particles import Particles

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
    
def check_E_field( interp, epsilon, k0, w0, wp, t, field='Ez' ) :
    """
    Compare the longitudinal and radial field with the
    simulation.
    """
    # -------------------
    # Longitudinal field 
    # -------------------
    plt.figure(figsize=(8,10))
    plt.suptitle('%s field' %field)
    
    # 2D plots
    r, z = np.meshgrid( interp.r, interp.z )
    if field == 'Ez' : 
        E_analytical = Ez( z, r, epsilon, k0, w0, wp, t )
        E_simulation = interp.Ez.real
    if field == 'Er' :
        E_analytical = Er( z, r, epsilon, k0, w0, wp, t )
        E_simulation = interp.Er.real
    
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
    
if __name__ == '__main__' :

    # ----------
    # Parameters
    # ----------

    use_cuda=True
    
    # The simulation box
    Nz = 512         # Number of gridpoints along z
    zmax = 40.e-6    # Length of the box along z (meters)
    Nr = 32          # Number of gridpoints along r
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
    
    # -------------------------
    # Launching the simulation
    # -------------------------
    
    # Initialization of the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
    p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
    n_order=n_order, use_cuda=use_cuda )

    # Redo the initialization of the electrons, to avoid to two last empty
    # cells at the right end of the box
    p_zmin, p_zmax, Npz = adapt_to_grid( sim.fld.interp[0].z,
                                p_zmin, p_zmax, p_nz, ncells_empty=0 )
    p_rmin, p_rmax, Npr = adapt_to_grid( sim.fld.interp[0].r,
                                p_rmin, p_rmax, p_nr, ncells_empty=0 )
    sim.ptcl[0] = Particles( q=-e, m=m_e, n=n_e, Npz=Npz, zmin=p_zmin,
                             zmax=p_zmax, Npr=Npr, rmin=p_rmin, rmax=p_rmax,
                             Nptheta=p_nt, dt=dt, use_cuda = sim.use_cuda )
    # Do the initial charge deposition (at t=0) now
    sim.fld.erase('rho')
    for species in sim.ptcl :
        species.deposit( sim.fld.interp, 'rho' )
    sim.fld.divide_by_volume('rho')
    # Bring it to the spectral space
    sim.fld.interp2spect('rho_prev')
    sim.fld.filter_spect('rho_prev')
    
    # Impart velocities to the electrons
    # (The electrons are initially homogeneous, but have an
    # intial non-zero velocity that develops into a plasma wave)
    impart_momenta( sim.ptcl[0], epsilon, k0, w0, wp )

    # Launch the simulation
    sim.step( N_step, moving_window=False )

    # Check the Ez field
    check_E_field( sim.fld.interp[0], epsilon, k0, w0, wp,
                   sim.time, field='Ez' )
    # Check the Er field
    check_E_field( sim.fld.interp[0], epsilon, k0, w0, wp,
                   sim.time, field='Er' )
