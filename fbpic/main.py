"""
Fourier-Bessel Particle-In-Cell (FB-PIC) main file

This file steers and controls the simulation.
"""
import sys
from scipy.constants import m_e, e
from fields import Fields
from particles import Particles
from moving_window import move_window

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
                 p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e ) :
        """
        Initializes a simulation, by creating the following structures :
        - the Fields object, which contains the EM fields
        - a set of electrons (if no electric field is initialized, this is as if 

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
           Density of the electrons
        """
        # Initialize the field structure
        self.fld = Fields(Nz, zmax, Nr, rmax, Nm, dt)

        # Modify the input parameters p_zmin, p_zmax, r_zmin, r_zmax, so that
        # they fall exactly on the grid, and infer the number of particles
        p_zmin, p_zmax, Npz = adapt_to_grid( self.fld.interp[0].z,
                                p_zmin, p_zmax, p_nz )
        p_rmin, p_rmax, Npr = adapt_to_grid( self.fld.interp[0].r,
                                p_rmin, p_rmax, p_nr )
        
        # Initialize the electrons
        # (using 4 macroparticles per cell along the azimuthal direction)
        self.ptcl = [
            Particles( q=-e, m=m_e, n=n_e, Npz=Npz, zmin=p_zmin, zmax=p_zmax,
                       Npr=Npr, rmin=p_rmin, rmax=p_rmax, Nptheta=4, dt=dt )
            ]
        
        # Register the number of particles per cell along z, and the time
        # (Necessary for the moving window)
        self.time = 0.
        self.dt = dt
        self.p_nz = p_nz
        
        # Do the initial charge deposition (at t=0) now
        for species in self.ptcl :
            species.deposit( self.fld.interp, 'rho' )
        self.fld.divide_by_volume('rho')
        # Bring it to the spectral space
        self.fld.interp2spect('rho_prev')


        
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
            Whether to move the window at c
        """
        # Shortcuts
        ptcl = self.ptcl
        fld = self.fld
        
        # Loop over timesteps
        for i_step in xrange(N) :

            # Show a progression bar
            progression_bar( i_step, N )

            # Move the window if needed
            if moving_window :
                move_window( fld, ptcl, self.p_nz, self.time )
            
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
            # Get the current on the spectral grid at t = (n+1/2) dt
            fld.interp2spect('J')

            # Push the particles' positions to t = (n+1) dt
            if move_positions :
                for species in ptcl :
                    species.halfpush_x()
            # Get the charge density on the interpolation grid at t = (n+1) dt
            fld.erase('rho')
            for species in ptcl :
                species.deposit( fld.interp, 'rho' )
            fld.divide_by_volume('rho')
            # Get the charge density on the spectral grid at t = (n+1) dt
            fld.interp2spect('rho_next')
            # Correct the currents (requires rho at t = (n+1) dt )
            if correct_currents :
                fld.correct_currents()
            
            # Get the fields E and B on the spectral grid at t = (n+1) dt
            fld.push( ptcl_feedback )
            # Get the fields E and B on the interpolation grid at t = (n+1) dt
            fld.spect2interp('E')
            fld.spect2interp('B')
    
            # Increment the global time
            self.time += self.dt

def progression_bar(i, Ntot, Nbars=60, char='-') :
    "Shows a progression bar with Nbars"
    nbars = int( (i+1)*1./Ntot*Nbars )
    sys.stdout.write('\r[' + nbars*char )
    sys.stdout.write((Nbars-nbars)*' ' + ']')
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
    
    # Leave the last cell empy
    xmax = x.max()
    dx = x[1] - x[0]
    if p_xmax > xmax - 0.5*dx :
        p_xmax = xmax - 0.5*dx
    # Do not load particles below the lower bound of the box
    xmin = x.min()
    if p_xmin < xmin - 0.5*dx :
        p_xmin = xmin - 0.5*dx
        
    # Find the gridpoints on which the particles should be loaded
    x_load = x[ ( x>p_xmin ) & ( x < p_xmax ) ]
    p_xmin = x_load.min() - 0.5*dx
    p_xmax = x_load.max() + 0.5*dx
    
    # Deduce the total number of particles
    Npx = len(x_load) * p_nx

    return( p_xmin, p_xmax, Npx )
    
