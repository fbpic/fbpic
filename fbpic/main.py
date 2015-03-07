"""
Fourier-Bessel Particle-In-Cell (FB-PIC) main file

This file steers and controls the simulation.
"""
import sys
from scipy.constants import m_e, e
from fbpic.fields import Fields
from fbpic.particles import Particles

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
                 p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, n_e ) :
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

        n_e : float (in particles per m^3)
           Density of the electrons
        """
        # Initialize the field structure
        self.fld = Fields(Nz, zmax, Nr, rmax, Nm, dt)

        # Determine the total number of particles along the z and r direction
        Npz = int( (p_zmax - p_zmin)*p_nz*Nz * 1./zmax )
        Npr = int( (p_rmax - p_rmin)*p_nr*Nr * 1./rmax )
        
        # Initialize the electrons
        # (using 4 macroparticles per cell along the azimuthal direction)
        self.ptcl = [
            Particles( q=-e, m=m_e, n=n_e, Npz=Npz, zmin=p_zmin, zmax=p_zmax,
                       Npr=Npr, rmin=p_rmin, rmax=p_rmax, Nptheta=4, dt=dt )
            ]
        
        # Do the initial charge deposition (at t=0) now
        for species in self.ptcl :
            species.deposit( self.fld.interp, 'rho' )
        self.fld.divide_by_volume('rho')
        # Bring it to the spectral space
        self.fld.interp2spect('rho_prev')

    def step(self, N=1) :
        """
        Takes N timesteps in the PIC cycle
    
        The structures fld (Fields object) and ptcl (Particles object)
        have to be defined at that point
    
        Parameter
        ---------
        N : int, optional
            The number of timesteps to take
        """
        # Shortcuts
        ptcl = self.ptcl
        fld = self.fld
        
        # Loop over timesteps
        for i_step in xrange(N) :

            # Show a progression bar
            progression_bar( i_step, N )
            
            # Gather the fields at t = n dt
            for species in ptcl :
                species.gather( fld.interp )
    
            # Push the particles' positions and velocities to t = (n+1/2) dt
            for species in ptcl :
                species.push_p()
                species.halfpush_x()
            # Get the current on the interpolation grid at t = (n+1/2) dt
            fld.erase('J')
            for species in ptcl :
                species.deposit( fld.interp, 'J' )
            fld.divide_by_volume('J')
            # Get the current on the spectral grid at t = (n+1/2) dt
            fld.interp2spect('J')

            # Push the particles' position to t = (n+1) dt
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
            fld.correct_currents()
            
            # Get the fields E and B on the spectral grid at t = (n+1) dt
            fld.push()
            # Get the fields E and B on the interpolation grid at t = (n+1) dt
            fld.spect2interp('E')
            fld.spect2interp('B')
    
            # Boundary conditions could potentially be implemented here, 
            # on the interpolation grid. This would impose
            # to then convert the fields back to the spectral space.




def progression_bar(i, Ntot, Nbars=60, char='-') :
    "Shows a progression bar with Nbars"
    nbars = int( i*1./Ntot*Nbars )
    sys.stdout.write('\r[' + nbars*char )
    sys.stdout.write((Nbars-nbars)*' ' + ']')
    sys.stdout.flush()
