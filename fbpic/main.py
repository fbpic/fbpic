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
except ImportError:
    print '\n Cuda not installed or GPU not available \n'

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
                 p_zmin, p_zmax, p_rmin, p_rmax,
                 p_nz, p_nr, p_nt, n_e, dens_func=None,
                 initialize_ions=False, use_cuda = False) :
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
            The size of the simulation box along z and r

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
           
        dens_func : callable, optional
           A function of the form :
           def dens_func( z, r ) ...
           where z and r are 1d arrays, and which returns
           a 1d array containing the density *relative to n*
           (i.e. a number between 0 and 1) at the given positions

        initialize_ions : bool, optional
           Whether to initialize the neutralizing ions

        use_cuda : bool, optional
            Wether to use CUDA (GPU) acceleration
        """
        self.use_cuda = use_cuda
        # Initialize the field structure
        self.fld = Fields(Nz, zmax, Nr, rmax, Nm, dt)

        # Modify the input parameters p_zmin, p_zmax, r_zmin, r_zmax, so that
        # they fall exactly on the grid, and infer the number of particles
        p_zmin, p_zmax, Npz = adapt_to_grid( self.fld.interp[0].z,
                                p_zmin, p_zmax, p_nz )
        p_rmin, p_rmax, Npr = adapt_to_grid( self.fld.interp[0].r,
                                p_rmin, p_rmax, p_nr )
        
        # Initialize the electrons and the ions
        self.ptcl = [
            Particles( q=-e, m=m_e, n=n_e, Npz=Npz, zmin=p_zmin, zmax=p_zmax,
                       Npr=Npr, rmin=p_rmin, rmax=p_rmax, Nptheta=p_nt, dt=dt,
                       dens_func=dens_func, use_cuda = use_cuda) ]
        if initialize_ions :
            self.ptcl.append(
                Particles( q=e, m=m_p, n=n_e, Npz=Npz, zmin=p_zmin, zmax=p_zmax,
                        Npr=Npr, rmin=p_rmin, rmax=p_rmax, Nptheta=p_nt, dt=dt,
                        dens_func=dens_func, use_cuda = use_cuda ) )
        
        # Register the number of particles per cell along z, and dt
        # (Necessary for the moving window)
        self.dt = dt
        self.p_nz = p_nz
        # Register the time and the iteration
        self.time = 0.
        self.iteration = 0
        
        # Do the initial charge deposition (at t=0) now
        for species in self.ptcl :
            species.deposit( self.fld.interp, 'rho' )
        self.fld.divide_by_volume('rho')
        # Bring it to the spectral space
        self.fld.interp2spect('rho_prev')
        self.fld.filter_spect('rho_prev')

        # Initialize an empty list of diagnostics
        self.diags = []


        
    def step(self, N=1, ptcl_feedback=True, correct_currents=True,
             filter_currents=True, move_positions=True,
             move_momenta=True, moving_window=True, move_window_nsteps = 1) :
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

        filter_currents : bool, optional
            Whether to filter the currents (in k space by default)

        move_positions : bool, optional
            Whether to move or freeze the particles' positions

        move_momenta : bool, optional
            Whether to move or freeze the particles' momenta

        moving_window : bool, optional
            Whether to move using a moving window. In this case,
            a MovingWindow object has to be attached to the simulation
            beforehand. e.g : sim.moving_win = MovingWindow(v=c)

        move_window_nsteps : int, optional
            Advance the moving window every n steps for n steps. 
            This ensures higher performance on GPU version of the code,
            as the particle and field arrays only need to be copied 
            to and from the device every move_window_nsteps. 
        """
        # Shortcuts
        ptcl = self.ptcl
        fld = self.fld

        # Check if a moving window is attached, in the case
        # moving_window == True
        if moving_window :
            if hasattr(self, 'moving_win')==False or \
              self.moving_win is None :
                raise AttributeError(
        "Please attach a MovingWindow to this object, as `self.moving_win`")
        
        # Send simulation data to GPU (if CUDA is used)
        send_data_to_gpu(self)

        # Loop over timesteps
        for i_step in xrange(N) :

            # Run the diagnostics
            for diag in self.diags :
                # Check if the fields should 
                # be written at this iteration
                if self.iteration % diag.period == 0 :
                    # Write the diagnostics
                    diag.write( self.iteration )
            
            # Show a progression bar
            progression_bar( i_step, N )

            # Move the window if needed
            if moving_window and self.iteration % move_window_nsteps == 0:
                # Receive the data from the GPU (if CUDA is used)
                # for the advance of the moving window
                receive_data_from_gpu(self)
                # Shift the fields and add new particles
                self.moving_win.move( 
                    fld, ptcl, self.p_nz, move_window_nsteps*self.dt )
                # Send the data to the GPU (if Cuda is used)
                send_data_to_gpu(self)
                # Reprojected the charge on the interpolation grid
                # (Particles have been added/removed.)
                fld.erase('rho')
                for species in ptcl :
                    species.deposit( fld.interp, 'rho' )
                fld.divide_by_volume('rho')
                self.moving_win.damp( fld.interp, 'rho' )
                # Get the new charge on the spectral grid
                fld.interp2spect('rho_prev')
                if filter_currents :
                    fld.filter_spect('rho_prev')

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
            # Get the current on the interpolation grid at t = (n+1/2) dt
            fld.erase('J')
            for species in ptcl :
                species.deposit( fld.interp, 'J' )
            fld.divide_by_volume('J')
            if moving_window :
                self.moving_win.damp( fld.interp, 'J' )
            # Get the current on the spectral grid at t = (n+1/2) dt
            fld.interp2spect('J')
            if filter_currents :
                fld.filter_spect('J')

            # Push the particles' positions to t = (n+1) dt
            if move_positions :
                for species in ptcl :
                    species.halfpush_x()
            # Get the charge density on the interpolation grid at t = (n+1) dt
            fld.erase('rho')
            for species in ptcl :
                species.deposit( fld.interp, 'rho' )
            fld.divide_by_volume('rho')
            if moving_window :
                self.moving_win.damp( fld.interp, 'rho' )
            # Get the charge density on the spectral grid at t = (n+1) dt
            fld.interp2spect('rho_next')
            if filter_currents :
                fld.filter_spect('rho_next')
            # Correct the currents (requires rho at t = (n+1) dt )
            if correct_currents :
                fld.correct_currents()
            
            # Get the fields E and B on the spectral grid at t = (n+1) dt
            fld.push( ptcl_feedback )
            # Get the fields E and B on the interpolation grid at t = (n+1) dt
            fld.spect2interp('E')
            fld.spect2interp('B')
    
            # Increment the global time and iteration
            self.time += self.dt
            self.iteration += 1

        # Receive simulation data from GPU (if CUDA is used)
        receive_data_from_gpu(self)

        # Print a space at the end of the loop, for esthetical reasons
        print('')

def progression_bar(i, Ntot, Nbars=60, char='-') :
    "Shows a progression bar with Nbars"
    nbars = int( (i+1)*1./Ntot*Nbars )
    sys.stdout.write('\r[' + nbars*char )
    sys.stdout.write((Nbars-nbars)*' ' + ']')
    sys.stdout.write(' %d/%d' %(i,Ntot))
    sys.stdout.flush()

def adapt_to_grid( x, p_xmin, p_xmax, p_nx ) :
    """
    Adapt p_xmin and p_xmax, so that they fall exactly on the grid x
    Return the total number of particles, assuming p_nx particles per gridpoint
    
    Parameters
    ----------
    x : 1darray
        The positions of the gridpoints along the x direction

    p_xmin, p_xmax : float
        The minimal and maximal position of the particles
        These may not fall exactly on the grid

    p_nx : int
        Number of particle per gridpoint
    
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
    if p_xmax > xmax - 1.5*dx :
        p_xmax = xmax - 1.5*dx
            
    # Find the gridpoints on which the particles should be loaded
    x_load = x[ ( x > p_xmin ) & ( x < p_xmax ) ]
    p_xmin = x_load.min() - 0.5*dx
    p_xmax = x_load.max() + 0.5*dx
    
    # Deduce the total number of particles
    Npx = len(x_load) * p_nx

    return( p_xmin, p_xmax, Npx )    
