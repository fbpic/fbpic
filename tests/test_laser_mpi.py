"""
This is file tests a laser propagation
when using parallel domain decomposition
with MPI threads

Usage :
-----
from the top-level directory of FBPIC run
$ python tests/test_laser_mpi.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils import add_laser
from fbpic.diagnostics import FieldDiagnostic
from mpi4py import MPI as mpi

# ---------------------------
# Comparison plots
# ---------------------------

def plot_gathered_fields(grid):

    # Get extent from grid object
    extent = np.array([ grid.zmin-0.5*grid.dz, grid.zmax+0.5*grid.dz,
                        -0.5*grid.dr, grid.rmax + 0.5*grid.dr ])
    # Rescale extent to microns
    extent = extent/1.e-6

    # Plot simulated Ez in 2D
    plt.subplot(221)
    plt.imshow(grid.Ez[:,::-1].real.T, extent = extent, 
        aspect='auto', interpolation='nearest')
    plt.xlabel('z')
    plt.ylabel('r')
    cb = plt.colorbar()
    cb.set_label('Ez')
    plt.title('2D Ez - mode 1 - real part')

    # Plot simulated Er in 2D
    plt.subplot(222)
    plt.imshow(grid.Er[:,::-1].real.T, extent = extent, 
        aspect='auto', interpolation='nearest')
    plt.xlabel('z')
    plt.ylabel('r')
    cb = plt.colorbar()
    cb.set_label('Er')
    plt.title('2D Er - mode 1 - real part')

    # Plot lineouts of Ez (simulation and analytical solution)
    plt.subplot(223)
    plt.plot(1.e6*grid.z, grid.Ez[:,0].real, 
        color = 'b', label = 'Simulation')
    plt.xlabel('z')
    plt.ylabel('Ez')
    plt.title('On-axis lineout of Ez')

    # Plot lineouts of Er (simulation and analytical solution)
    plt.subplot(224)
    plt.plot(1.e6*grid.z, grid.Er[:,5].real, 
        color = 'b', label = 'Simulation')
    plt.xlabel('z')
    plt.ylabel('Er')
    plt.title('Off-axis lineout of Er')

    # Show plots
    plt.show()

def plot_field_error(grid_mpi, grid):

    # Get extent from grid object
    extent = np.array([ grid_mpi.zmin-0.5*grid_mpi.dz, grid_mpi.zmax+0.5*grid_mpi.dz,
                        -0.5*grid_mpi.dr, grid_mpi.rmax + 0.5*grid_mpi.dr ])
    # Rescale extent to microns
    extent = extent/1.e-6

    # Plot simulated Ez in 2D
    plt.subplot(221)
    plt.imshow((grid.Ez[:,::-1].real.T - grid_mpi.Ez[:,::-1].real.T), extent = extent, 
        aspect='auto', interpolation='nearest')
    plt.xlabel('z')
    plt.ylabel('r')
    cb = plt.colorbar()
    cb.set_label('Ez')
    plt.title('2D Ez - mode 1 - real part')

    # Plot simulated Er in 2D
    plt.subplot(222)
    plt.imshow(grid.Er[:,::-1].real.T-grid_mpi.Er[:,::-1].real.T, extent = extent, 
        aspect='auto', interpolation='nearest')
    plt.xlabel('z')
    plt.ylabel('r')
    cb = plt.colorbar()
    cb.set_label('Er')
    plt.title('2D Er - mode 1 - real part')

    # Plot lineouts of Ez (simulation and analytical solution)
    plt.subplot(223)
    plt.plot(1.e6*grid_mpi.z, grid.Ez[:,0].real-grid_mpi.Ez[:,0].real, 
        color = 'b', label = 'Simulation')
    plt.xlabel('z')
    plt.ylabel('Ez')
    plt.title('On-axis lineout of Ez')

    # Plot lineouts of Er (simulation and analytical solution)
    plt.subplot(224)
    plt.plot(1.e6*grid_mpi.z, grid.Er[:,5].real-grid_mpi.Er[:,5].real, 
        color = 'b', label = 'Simulation')
    plt.xlabel('z')
    plt.ylabel('Er')
    plt.title('Off-axis lineout of Er')


    print 'Maximum error Ez in percent:'
    print 100*np.amax(grid.Ez[:,::-1].real.T - grid_mpi.Ez[:,::-1].real.T)/np.amax(grid.Ez[:,::-1].real.T)
    print 'Maximum error Er in percent:'
    print 100*np.amax(grid.Er[:,::-1].real.T - grid_mpi.Er[:,::-1].real.T)/np.amax(grid.Er[:,::-1].real.T)

    # Show plots
    plt.show()

if __name__ == '__main__' :

    # Setup MPI

    mpi_comm = mpi.COMM_WORLD

    rank = mpi_comm.rank
    size = mpi_comm.size

    # ---------------------------
    # Setup simulation & parameters
    # ---------------------------
    use_mpi = True
    # The simulation box
    Nz = 1604         # Number of gridpoints along z
    zmax = 70.e-6    # Length of the box along z (meters)
    Nr = 40          # Number of gridpoints along r
    rmax = 20.e-6    # Length of the box along r (meters)
    Nm = 2           # Number of modes used
    # The simulation timestep
    dt = zmax/Nz/c   # Timestep (seconds)

    # The particles
    p_zmin = 1.e-6  # Position of the beginning of the plasma (meters)
    p_zmax = 61.e-6  # Position of the end of the plasma (meters)
    p_rmin = 0.      # Minimal radial position of the plasma (meters)
    p_rmax = 20.e-6  # Maximal radial position of the plasma (meters)
    n_e = 4.e18*1.e6 # Density (electrons.meters^-3)
    p_nz = 2         # Number of particles per cell along z
    p_nr = 2         # Number of particles per cell along r
    p_nt = 4         # Number of particles per cell along theta

    # The laser
    a0 = 1.0        # Laser amplitude
    w0 = 5.e-6       # Laser waist
    ctau = 7.e-6     # Laser duration
    z0 = zmax/2     # Laser centroid

    # Initialize the simulation object
    if rank == 0:
        sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
            p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
            use_mpi = False)

    sim_mpi = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
        p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
        use_mpi = use_mpi, n_guard = 100)

    # Remove Plasma
    if rank == 0:
        sim.ptcl = []
    sim_mpi.ptcl = []

    # Add a laser to the fields of the simulation
    if rank == 0:
        add_laser( sim.fld, a0, w0, ctau, z0 )
    add_laser( sim_mpi.fld, a0, w0, ctau, z0 )


    # ---------------------------
    # Carry out simulation
    # ---------------------------

    # Carry out 300 PIC steps
    print 'Calculate PIC solution for the wakefield'
    if rank == 0:
        sim.step(10, moving_window = False)

    mpi_comm.Barrier()
    sim_mpi.step(10, moving_window = False)
    print 'Done...'
    print ''

    # Gather grid
    if use_mpi:
        gathered_grid = sim_mpi.comm.gather_grid(sim_mpi.fld.interp[1])

    # Plot the wakefields
    if use_mpi:
        if rank == 0:
            plot_gathered_fields(gathered_grid)
    
    if rank == 0:
        plot_gathered_fields(sim.fld.interp[1])

    if use_mpi and rank == 0:
        print 'plotting now..'
        plot_field_error(gathered_grid, sim.fld.interp[1])


