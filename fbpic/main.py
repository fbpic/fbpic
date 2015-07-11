"""
Fourier-Bessel Particle-In-Cell (FB-PIC) main file

This file steers and controls the simulation.
"""
import sys
from scipy.constants import m_e, m_p, e
from fields import Fields
from particles import Particles

try:
    from cuda_utils import *
    cuda_installed = True
except ImportError:
    cuda_installed = False

try:
    from parallel import MPI_Communicator
    mpi_installed = True
except ImportError:
    mpi_installed = False

    
class Simulation(object) :
    """
    Top-level simulation class that contains all the simulation
    data, as well as the methods to perform the PIC cycle.

    Attributes
    ----------
    - fld : a Fields object
    - ptcl : a list of Particles objects (one element per species)

    Methods
    -------
    - step : perform n PIC cycles
    """

    def __init__(self, Nz, zmax, Nr, rmax, Nm, dt,
                 p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt,
                 n_e, zmin = 0., dens_func=None, filter_currents=True,
                 initialize_ions=False, use_cuda=False, use_mpi=False,
                 n_guard=50, boundaries='periodic' ) :
        """
        Initializes a simulation, by creating the following structures :
        - the Fields object, which contains the EM fields
        - a set of electrons
        - a set of ions (if initialize_ions is True)

        Parameters
        ----------
        Nz, Nr : ints
            The number of gridpoints in z and r

        zmax, rmax : floats
            The position of the edge of the simulation in z and r
            (More precisely, the position of the edge of the last cell)

        Nm : int
            The number of azimuthal modes taken into account

        dt : float
            The timestep of the simulation

        p_zmin, p_zmax : floats
            z positions between which the particles are initialized

        p_rmin, p_rmax : floats
            r positions between which the fields are initialized

        p_nz, p_nr : ints
            Number of macroparticles per cell along the z and r directions

        p_nt : int
            Number of macroparticles along the theta direction

        n_e : float (in particles per m^3)
           Peak density of the electrons

        zmin : float, optional
           The position of the edge of the simulation box
           (More precisely, the position of the edge of the first cell)
           
        dens_func : callable, optional
           A function of the form :
           def dens_func( z, r ) ...
           where z and r are 1d arrays, and which returns
           a 1d array containing the density *relative to n*
           (i.e. a number between 0 and 1) at the given positions

        initialize_ions : bool, optional
           Whether to initialize the neutralizing ions

        filter_currents : bool, optional
            Whether to filter the currents and charge in k space
           
        use_cuda : bool, optional
            Wether to use CUDA (GPU) acceleration

        use_mpi : bool, optional
            Wether to decompose the simulation longitudinally
            and use multiple MPI threads to calculate in parallel

        n_guard : int, optional
            Number of guard cells to use at the left and right of
            a domain, when using MPI.

        boundaries : str
            Indicates how to exchange the fields at the left and right
            boundaries of the global simulation box
            Either 'periodic' or 'open'
        """
        # Check whether to use cuda
        self.use_cuda = use_cuda
        if (use_cuda==True) and (cuda_installed==False) :
            self.use_cuda = False

        # Check wether to use MPI
        self.use_mpi = use_mpi
        if (self.use_mpi) and (mpi_installed == False):
            self.use_mpi = False
        if self.use_mpi:
            # Initialize the MPI communicator
            self.comm = MPI_Communicator(Nz, Nr, n_guard, Nm, boundaries)
            # Modify domain region
            zmin, zmax, p_zmin, p_zmax = self.comm.divide_into_domain(
                                            zmin, zmax, p_zmin, p_zmax)
            # Register new number of cells for local domain
            Nz = self.comm.Nz_local
        else :
            self.comm = None

        # Initialize the field structure
        self.fld = Fields(Nz, zmax, Nr, rmax, Nm, dt,
                          zmin=zmin, use_cuda=self.use_cuda)

        # Modify the input parameters p_zmin, p_zmax, r_zmin, r_zmax, so that
        # they fall exactly on the grid, and infer the number of particles
        p_zmin, p_zmax, Npz = adapt_to_grid( self.fld.interp[0].z,
                                p_zmin, p_zmax, p_nz )
        p_rmin, p_rmax, Npr = adapt_to_grid( self.fld.interp[0].r,
                                p_rmin, p_rmax, p_nr )

        # Initialize the electrons and the ions
        self.ptcl = [
            Particles( q=-e, m=m_e, n=n_e, Npz=Npz, zmin=p_zmin,
                       zmax=p_zmax, Npr=Npr, rmin=p_rmin, rmax=p_rmax,
                       Nptheta=p_nt, dt=dt, dens_func=dens_func,
                       use_cuda=self.use_cuda) ]
        if initialize_ions :
            self.ptcl.append(
                Particles(q=e, m=m_p, n=n_e, Npz=Npz, zmin=p_zmin,
                          zmax=p_zmax, Npr=Npr, rmin=p_rmin, rmax=p_rmax,
                          Nptheta=p_nt, dt=dt, dens_func=dens_func,
                          use_cuda=self.use_cuda ) )
        
        # Register the number of particles per cell along z, and dt
        # (Necessary for the moving window)
        self.dt = dt
        self.p_nz = p_nz
        # Register the time and the iteration
        self.time = 0.
        self.iteration = 0
        # Register the filtering flag
        self.filter_currents = filter_currents
        
        # Do the initial charge deposition (at t=0) now
        self.deposit('rho_prev')

        # Initialize an empty list of diagnostics
        self.diags = []

    def step(self, N=1, ptcl_feedback=True, correct_currents=True,
             move_positions=True, move_momenta=True, moving_window=True ) :
        """
        Perform N PIC cycles
        
        Parameter
        ---------
        N : int, optional
            The number of timesteps to take
            Default : N=1

        ptcl_feedback : bool, optional
            Whether to take into account the particle density and
            currents when pushing the fields

        correct_currents : bool, optional
            Whether to correct the currents in spectral space

        move_positions : bool, optional
            Whether to move or freeze the particles' positions

        move_momenta : bool, optional
            Whether to move or freeze the particles' momenta

        moving_window : bool, optional
            Whether to move using a moving window. In this case,
            a MovingWindow object has to be attached to the simulation
            beforehand. e.g : sim.moving_win = MovingWindow(v=c)
        """
        # Shortcuts
        ptcl = self.ptcl
        fld = self.fld

        # Send simulation data to GPU (if CUDA is used)
        if self.use_cuda:
            send_data_to_gpu(self)

        # Loop over timesteps
        for i_step in xrange(N) :

            # Exchange the fields (EB) and the particles 
            # in the guard cells between domains
            if self.use_mpi:
                self.comm.exchange_fields(self.fld.interp, 'EB')
                # Particle exchange every exchange_part_period
                if self.iteration % self.comm.exchange_part_period == 0:
                    for species in self.ptcl:
                        self.comm.exchange_particles(species,
                            fld.interp[0].zmin, fld.interp[0].zmax )

            # Run the diagnostics
            for diag in self.diags :
                # Check if the fields should be written at
                # this iteration and do it if needed.
                # (Send the data to the GPU if needed.)
                diag.write( self.iteration )
            
            # Show a progression bar
            progression_bar( i_step, N )

            # Handle the moving window
            if moving_window :

                # Move the window if needed
                if self.iteration % self.moving_win.period == 0 :
                    
                    # Receive the data from the GPU (if CUDA is used)
                    if self.use_cuda:
                        receive_data_from_gpu(self)
                    # Shift the fields and add new particles
                    # (Exchange fields and particles between
                    # processors when using MPI)
                    self.moving_win.move( fld.interp, ptcl, self.p_nz,
                                          self.dt, self.comm )
                    # Send the data to the GPU (if Cuda is used)
                    if self.use_cuda:
                        send_data_to_gpu(self)
                    # Reproject the charge on the interpolation grid
                    # (Since particles have been added/suppressed)
                    self.deposit('rho_prev')

                # Damp the fields at every time step (at the left
                # and right boundary). When using MPI, only the last
                # and first processors damp the fields.
                self.moving_win.damp_EB( fld.interp, self.comm )

            # Gather the fields at t = n dt
            for species in ptcl :
                species.gather( fld.interp )

            # Push the particles' positions and velocities to t = (n+1/2) dt
            if move_momenta :
                for species in ptcl :
                    species.push_p()
 
            if move_positions :
                for species in ptcl :
                    species.halfpush_x()

            # Get the current at t = (n+1/2) dt
            self.deposit('J')

            # Push the particles' positions to t = (n+1) dt
            if move_positions :
                for species in ptcl :
                    species.halfpush_x()
            # Get the charge density at t = (n+1) dt
            self.deposit('rho_next')

            # Correct the currents (requires rho at t = (n+1) dt )
            if correct_currents :
                fld.correct_currents()

            if moving_window or self.use_mpi :
                if self.use_mpi:
                    # Damp the fields in the guard cells
                    self.comm.damp_guard_fields(self.fld.interp)
                # Get the exchanged and/or damped fields 
                # E and B on the spectral grid
                fld.interp2spect('E')
                fld.interp2spect('B')
            # Get the fields E and B on the spectral grid at t = (n+1) dt
            fld.push( ptcl_feedback )
            # Get the fields E and B on the interpolation grid at t = (n+1) dt
            fld.spect2interp('E')
            fld.spect2interp('B')

            # Increment the global time and iteration
            self.time += self.dt
            self.iteration += 1

        # Receive simulation data from GPU (if CUDA is used)
        if self.use_cuda:
            receive_data_from_gpu(self)

        # Print a space at the end of the loop, for esthetical reasons
        print('')

    def deposit( self, fieldtype ) :
        """
        Deposit the charge or the currents to the interpolation
        grid and then to the spectral grid.
    
        Parameters :
        ------------
        fieldtype : str
            The designation of the spectral field that
            should be changed by the deposition
            Either 'rho_prev', 'rho_next' or 'J'
        """
        # Shortcut
        fld = self.fld

        # Deposit charge or currents on the interpolation grid
        # Charge
        if fieldtype in ['rho_prev', 'rho_next'] :
            fld.erase('rho')
            for species in self.ptcl :
                species.deposit( fld.interp, 'rho' )
            fld.divide_by_volume('rho')
            # Exchange the charge density of the guard cells between domains
            if self.use_mpi:
                self.comm.exchange_fields(self.fld.interp, 'rho')
        # Currents
        elif fieldtype == 'J' :
            fld.erase('J')
            for species in self.ptcl :
                species.deposit( fld.interp, 'J' )
            fld.divide_by_volume('J')
            # Exchange the current of the guard cells between domains
            if self.use_mpi:
                self.comm.exchange_fields(self.fld.interp, 'J')
        else :
            raise ValueError('Unknown fieldtype : %s' %fieldtype)
            
        # Get the charge or currents on the spectral grid
        fld.interp2spect( fieldtype )
        if self.filter_currents :
            fld.filter_spect( fieldtype )


def progression_bar(i, Ntot, Nbars=60, char='-') :
    "Shows a progression bar with Nbars"
    nbars = int( (i+1)*1./Ntot*Nbars )
    sys.stdout.write('\r[' + nbars*char )
    sys.stdout.write((Nbars-nbars)*' ' + ']')
    sys.stdout.write(' %d/%d' %(i,Ntot))
    sys.stdout.flush()

def adapt_to_grid( x, p_xmin, p_xmax, p_nx, ncells_empty=0 ) :
    """
    Adapt p_xmin and p_xmax, so that they fall exactly on the grid x
    Return the total number of particles, assuming p_nx particles
    per gridpoint
    
    Parameters
    ----------
    x : 1darray
        The positions of the gridpoints along the x direction

    p_xmin, p_xmax : float
        The minimal and maximal position of the particles
        These may not fall exactly on the grid

    p_nx : int
        Number of particle per gridpoint

    ncells_empty : int
        Number of empty cells at the righthand side of the box
        (Typically used when using a moving window)
        
    Returns
    -------
    A tuple with :
       - p_xmin : a float that falls exactly on the grid
       - p_xmax : a float that falls exactly on the grid
       - Npx : the total number of particles
    """
    
    # Find the max and the step of the array
    xmin = x.min()
    xmax = x.max()
    dx = x[1] - x[0]
    
    # Do not load particles below the lower bound of the box
    if p_xmin < xmin - 0.5*dx :
        p_xmin = xmin - 0.5*dx
    # Do not load particles in the two last upper cells
    # (This is because the charge density may extend over these cells
    # when it is smoothed. If particles are loaded closer to the right
    # boundary, this extended charge density can wrap around and appear
    # at the left boundary.)
    if p_xmax > xmax + (0.5-ncells_empty)*dx :
        p_xmax = xmax + (0.5-ncells_empty)*dx
    
    # Find the gridpoints on which the particles should be loaded
    x_load = x[ ( x > p_xmin ) & ( x < p_xmax ) ]
    # Deduce the total number of particles
    Npx = len(x_load) * p_nx
    # Reajust p_xmin and p_xmanx so that they match the grid
    if Npx > 0 :
        p_xmin = x_load.min() - 0.5*dx
        p_xmax = x_load.max() + 0.5*dx

    return( p_xmin, p_xmax, Npx )    
