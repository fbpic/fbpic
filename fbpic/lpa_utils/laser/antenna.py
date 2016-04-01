"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the class LaserAntenna, which can be used to continuously
emit a laser during a simulation.
"""
import numpy as np
from scipy.constants import e, c, m_e, epsilon_0
# Classical radius of the electron
r_e = e**2/(4*np.pi*epsilon_0*m_e*c**2)
from .profiles import gaussian_profile

class LaserAntenna( object ):
    """
    TO BE COMPLETED

    EXPLAIN THE VIRTUAL PARTICLES

    By default only the positive particles are stored.
    The excursion of the negative particles is the opposite.
    But then both negative and positive particles deposit current

    Every operation is done on the CPU
    
    """

    def __init__( self, E0, w0, ctau, z0, zf, k0, 
                    theta_pol, z0_antenna, dr_grid, Nr_grid, 
                    npr=2, Nptheta=4, epsilon=0.01, boost=None ):
        """
        TO BE COMPLETED

        npr: int
           Number of virtual particles along the r axis, per cell

        Nptheta: int
           Number of virtual particles in the theta direction
           (Particles are distributed along a star-pattern with
           Nptheta arms in the transverse plane)

        epsilon: float
           Ratio between the maximum transverse excursion of any virtual
           particle of the laser antenna, and the transverse size of a cell
           (i.e. a virtual particle will not move by more than epsilon*dr)

        boost: a BoostConverter object or None
        
        """
        # Porportionality coefficient between the weight of a particle
        # and its transverse position (in cylindrical geometry, particles
        # that are further away from the axis have a larger weight)
        alpha_weights = np.pi / ( 2*npr*epsilon ) * dr_grid / r_e * e
        # Mobility coefficient: proportionality coefficient between the
        # velocity of the particles and the electric field to be emitted
        self.mobility_coef = np.pi * dr_grid**2 / ( 2*npr*alpha_weights ) \
            * epsilon_0 * c
        if boost is not None:
            self.mobility_coef = self.mobility_coef / boost.gamma0

        # Get total number of virtual particles
        Npr = Nr_grid * npr
        Ntot = Npr * Nptheta
        # Get the baseline radius and angles of the virtual particles
        r_reg = dr_grid/npr * ( np.arange( Npr ) + 0.5 )
        theta_reg = 2*np.pi/Nptheta * np.arange( Nptheta )
        rp, thetap = np.meshgrid( r_reg, theta_reg, copy=True)
        r0 = rp.flatten()
        theta0 = thetap.flatten()
        
        # Baseline position of the particles and weights
        self.x0 = r0 * np.cos( theta0 )
        self.y0 = r0 * np.sin( theta0 )
        self.w = alpha_weights * self.r0 / dr_grid
        # Excursion with respect to the baseline position
        self.excursion_x = np.zeros( Ntot )
        self.excursion_y = np.zeros( Ntot )
        # Particle velocities
        self.vx = np.zeros( Ntot )
        self.vy = np.zeros( Ntot )

        # Position and velocity of the antenna
        self.z_antenna = z0_antenna
        self.vz_antenna = 0.
        # If the simulation is performed in a boosted frame,
        # boost these quantities
        if boost is not None:
            self.z_antenna, = boost.static_length( [ self.z_antenna ] )
            self.vz_antenna, = boost.velocity( [ self.vz_antenna ] )

        # Record laser properties
        self.E0 = E0
        self.w0 = w0
        self.k0 = k0
        self.ctau = ctau
        self.z0 = zf
        self.zf = zf
        self.boost = boost
            
    def halfpush_x( self, dt ):
        """
        Push the position of the virtual particles in the antenna
        over half a timestep, using their current velocity
    
        Parameter
        ---------
        dt: float (seconds)
            The (full) timestep of the simulation
        """
        # Half timestep
        hdt = 0.5*dt

        # Push transverse particle positions (element-wise array operation)
        self.excursion_x += hdt * self.vx
        self.excursion_y += hdt * self.vy

        # Move position of the antenna
        self.z_antenna += hdt * self.vz_antenna

    def update_v( self, t ):
        """
        Update the particle velocities so that it corresponds to time t

        The updated value of the velocities is determined by calculating
        the electric field at the time t and at the position of the antenna
        and by multiplying this field by the mobility.

        Parameter
        ---------
        t: float (seconds)
            The time at which to calculate the velocities
        """
        # The electric field is calculated from its lab-frame expression.
        # Thus, in case of a boost, find the time and position in the lab-frame
        if self.boost is not None:
            gamma0 = self.boost.gamma0
            beta0 = self.boost.beta0
            z_lab = gamma0*( self.z_antenna - beta0*t )
            t_lab = gamma0*( t - beta0*self.z_antenna )
        else:
            z_lab = self.z_antenna
            t_lab = t

        # Calculate the electric field to be emitted (in the lab-frame)
        # Eu is the amplitude along the polarization direction
        r0 = np.sqrt( (self.x0 + self.excursion_x)**2 + \
                      (self.y0 + self.excursion_y)**2 )
        Eu = self.E0 * gaussian_profile( z_lab, r0, t_lab,
                        self.w0, self.ctau, self.z0, self.zf,
                        self.k0, boost=None, output_longitudinal_field=False )

        # Calculate the corresponding velocity. This takes into account
        # lab-frame to boosted-frame conversion, through a modification
        # of the mobility coefficient: see the __init__ function
        self.vx = ( self.mobility_coef * np.cos(self.theta_pol) ) * Eu
        self.vy = ( self.mobility_coef * np.sin(self.theta_pol) ) * Eu

    def deposit( self ):
        """
        TO BE COMPLETED
        """
        # Check if z_antenna is in the current physical domain        
        # Calculate the displacement of the particles
        
        pass
    
    
