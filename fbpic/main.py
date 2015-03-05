"""
Fourier-Bessel Particle-In-Cell (FB-PIC) main file

This file steers and controls the simulation.
"""

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
                 p_zmin, p_zmax, p_rmax, p_nz, p_nr, p_q, p_m ) :
        """
        Initializes the simulation structures

        Parameters
        ----------
        Nz : int
            The number of gridpoints in z

        zmax : float
            The size of the simulation box along z
            
        Nr : int
            The number of gridpoints in r

        rmax : float
            The size of the simulation box along r

        Nm : int
            The number of azimuthal modes

        dt : float
            The timestep of the simulation
        """
        # Initialize the field structure
        self.fld = Fields(Nz, zmax, Nr, rmax, Nm, dt)
        # Fill their values
        # ....
        # Convert to spectral space
        self.fld.interp2spec('E')
        self.fld.interp2spec('B')
        
        # Initialize the particle structure
        self.ptcl = [
            Particles(..., dt ),  # Electrons
            Particles(...)   # Ions
            ]
        

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
        for _ in xrange(N) :
            
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
            fld.correct_currents()

            # Push the particles' position to t = (n+1) dt
            for species in ptcl :
                species.halfpush_x()
            # Get the charge density on the interpolation grid at t = (n+1) dt
            fld.erase('rho')
            for species in ptcl :
                species.deposit( fld.interp, 'rho' )
            fld.divide_by_volume('J')
            # Get the charge density on the spectral grid at t = (n+1) dt
            fld.interp2spect('rho')
    
            # Get the fields E and B on the spectral grid at t = (n+1) dt
            fld.push()
            # Get the fields E and B on the interpolation grid at t = (n+1) dt
            fld.spect2interp('E')
            fld.spect2interp('B')
    
            # Boundary conditions could potentially be implemented here, 
            # on the interpolation grid. This would impose
            # to then convert the fields back to the spectral space.
