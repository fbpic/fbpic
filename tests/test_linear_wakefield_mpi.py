"""
This is file tests the whole PIC-Cycle by simulating a 
linear, laser-driven plasma wakefield and comparing
it to the analytical solution.

Usage :
-----
from the top-level directory of FBPIC run
$ python tests/test_linear_wakefield.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_e, epsilon_0
from scipy.integrate import quad
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils import add_laser
from fbpic.moving_window import MovingWindow
from fbpic.diagnostics import FieldDiagnostic, ParticleDiagnostic
from mpi4py import MPI as mpi

# ---------------------------
# Analytical solution
# ---------------------------

# Laser field
def a2( xi, r ) :
    """Average of a^2 ; envelope of the intensity of the laser"""
    return( 0.5*a0**2*np.exp( -2*(xi + z0)**2/ctau**2 )*np.exp(-2*r**2/w0**2) )

def kernel_Ez( xi0, xi, r) :
    """Integration kernel for Ez"""
    return( m_e*c**2/e * kp**2/2 * np.cos( kp*(xi-xi0) )*a2( xi0, r ) )

def kernel_Er( xi0, xi, r) :
    """Integration kernel for Er"""
    return( m_e*c**2/e * 2*kp*r/w0**2 * np.sin( kp*(xi-xi0) )*a2( xi0, r ) )

def Ez( z, r, t) :
    """
    Get the 2d Ez field
    
    Parameters
    ----------
    z : 1darray
    t, r : float
    """
    Nz = len(z)
    Nr = len(r)

    ez = np.zeros((Nz, Nr))
    for iz in xrange(Nz) :
        for ir in xrange(Nr) :
          ez[iz, ir] = quad( kernel_Ez, -zmax, -z[iz]+c*t, 
            args = ( -z[iz]+c*t, r[ir] ), limit=30 )[0]
    return( ez )

def Er( z, r, t) :
    """
    Get the 2d Ez field
    
    Parameters
    ----------
    z : 1darray
    t, r : float
    """
    Nz = len(z)
    Nr = len(r)

    er = np.zeros((Nz, Nr))
    for iz in xrange(Nz) :
        for ir in xrange(Nr) :
          er[iz, ir] = quad( kernel_Er, -zmax, -z[iz]+c*t, 
            args = ( -z[iz]+c*t, r[ir] ), limit=200 )[0]
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
    plt.imshow(Ez_analytic[:,::-1].T, extent = extent, 
        aspect='auto', interpolation='nearest')
    plt.xlabel('z')
    plt.ylabel('r')
    cb = plt.colorbar()
    cb.set_label('Ez')
    plt.title('Analytical Ez')

    # Plot analytic Er in 2D
    plt.subplot(322)
    plt.imshow(Er_analytic[:,::-1].T, extent = extent, 
        aspect='auto', interpolation='nearest')
    plt.xlabel('z')
    plt.ylabel('r')
    cb = plt.colorbar()
    plt.title('Analytical Er')

    # Plot simulated Ez in 2D
    plt.subplot(323)
    plt.imshow(grid.Ez[:,::-1].real.T, extent = extent, 
        aspect='auto', interpolation='nearest')
    plt.xlabel('z')
    plt.ylabel('r')
    cb = plt.colorbar()
    cb.set_label('Ez')
    plt.title('Simulated Ez')

    # Plot simulated Er in 2D
    plt.subplot(324)
    plt.imshow(grid.Er[:,::-1].real.T, extent = extent, 
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

if __name__ == '__main__' :

    # ---------------------------
    # Setup simulation & parameters
    # ---------------------------

    use_cuda = False
    use_mpi = True
    n_guard = 50

    # Setup MPI
    mpi_comm = mpi.COMM_WORLD
    rank = mpi_comm.rank
    size = mpi_comm.size
    
    # The simulation box
    Nz = 601         # Number of gridpoints along z
    zmax = 25.e-6    # Length of the box along z (meters)
    Nr = 30          # Number of gridpoints along r
    rmax = 15.e-6    # Length of the box along r (meters)
    Nm = 2           # Number of modes used
    # The simulation timestep
    dt = zmax/Nz/c   # Timestep (seconds)
    # The number of steps
    N_step = 800

    # The particles
    p_zmin = 24.e-6  # Position of the beginning of the plasma (meters)
    p_zmax = 26.e-6  # Position of the end of the plasma (meters)
    p_rmin = 0.      # Minimal radial position of the plasma (meters)
    p_rmax = 12.e-6  # Maximal radial position of the plasma (meters)
    n_e = 16.e18*1.e6 # Density (electrons.meters^-3)
    p_nz = 2         # Number of particles per cell along z
    p_nr = 2         # Number of particles per cell along r
    p_nt = 4         # Number of particles per cell along theta

    # The laser
    a0 = 0.01        # Laser amplitude
    w0 = 5.e-6       # Laser waist
    ctau = 3.e-6     # Laser duration
    z0 = 14.e-6      # Laser centroid

    # Plasma and laser wavenumber
    kp = 1./c * np.sqrt( n_e * e**2 / (m_e * epsilon_0) )
    k0 = 2*np.pi/0.8e-6

    # The moving window
    v_window = c       # Speed of the window
    ncells_zero = 50   # Number of cells over which the field is set to 0
                       # at the left end of the simulation box
    ncells_damp = 50   # Number of cells over which the field is damped,
                       # at the left of the simulation box, after ncells_zero
                       # in order to prevent it from wrapping around.
    mw_period = 25     # How many steps to wait until moving the window 

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
        p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
        use_cuda=use_cuda, use_mpi=use_mpi, n_guard=n_guard ) 

    # Add a laser to the fields of the simulation
    add_laser( sim.fld, a0, w0, ctau, z0 )

    # Configure the moving window
    sim.moving_win = MovingWindow( ncells_damp=ncells_damp,
                                   ncells_zero=ncells_zero,
                                   period=mw_period )

    # ---------------------------
    # Carry out simulation
    # ---------------------------

    print 'Calculate PIC solution for the wakefield'
    sim.step(N_step, moving_window=True, correct_currents=False)
    print 'Done...'
    print ''

    # Gather the results
    if use_mpi:
        gathered_grid = sim.comm.gather_grid(sim.fld.interp[0])
        z = gathered_grid.z
        r = gathered_grid.r
    else :
        z = sim.fld.interp[0].z
        r = sim.fld.interp[0].r

    # Analytical solution
    if (use_mpi==False) or (rank==0):
        print 'Calculate analytical solution for Ez'
        ez = Ez(z-z.min(), r, 0.)
        print 'Done...'
        print ''

        print 'Calculate analytical solution for Er'
        er = Er(z-z.min(), r, 0.)
        print 'Done...'
        print ''

    # Plot the results
    if use_mpi:
        compare_wakefields(ez, er, gathered_grid)
    else :
        compare_wakefields(ez, er, sim.fld.interp[0])

