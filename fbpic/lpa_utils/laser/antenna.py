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
                    npr=2, nptheta=4, epsilon=0.01, boost=None ):
        """
        TO BE COMPLETED

        npr: int
           Number of virtual particles along the r axis, per cell

        nptheta: int
           Number of virtual particles in the theta direction
           (Particles are distributed along a star-pattern with
           nptheta arms in the transverse plane)

        epsilon: float
           Ratio between the maximum transverse excursion of any virtual
           particle of the laser antenna, and the transverse size of a cell
           (i.e. a virtual particle will not move by more than epsilon*dr)

        boost: a BoostConverter object or None
        
        """
        # Porportionality coefficient between the weight of a particle
        # and its transverse position (in cylindrical geometry, particles
        # that are further away from the axis have a larger weight)
        alpha_weights = 2*np.pi / ( nptheta*npr*epsilon ) * dr_grid / r_e * e
        # Mobility coefficient: proportionality coefficient between the
        # velocity of the particles and the electric field to be emitted
        self.mobility_coef = 2*np.pi * \
          dr_grid**2 / ( nptheta*npr*alpha_weights ) * epsilon_0 * c
        if boost is not None:
            self.mobility_coef = self.mobility_coef / boost.gamma0

        # Get total number of virtual particles
        Npr = Nr_grid * npr
        Ntot = Npr * nptheta
        # Get the baseline radius and angles of the virtual particles
        r_reg = dr_grid/npr * ( np.arange( Npr ) + 0.5 )
        theta_reg = 2*np.pi/nptheta * np.arange( nptheta )
        rp, thetap = np.meshgrid( r_reg, theta_reg, copy=True)
        self.baseline_r = rp.flatten()
        theta0 = thetap.flatten()
        
        # Baseline position of the particles and weights
        self.baseline_x = self.baseline_r * np.cos( theta0 )
        self.baseline_y = self.baseline_r * np.sin( theta0 )
        self.baseline_z = z0_antenna
        self.w = alpha_weights * self.baseline_r / dr_grid
        # Excursion with respect to the baseline position
        # (No excursion in z: the particles do not oscillate in this direction)
        self.excursion_x = np.zeros( Ntot )
        self.excursion_y = np.zeros( Ntot )
        # Particle velocities
        self.vx = np.zeros( Ntot )
        self.vy = np.zeros( Ntot )
        self.vz = np.zeros( Ntot )
        # If the simulation is performed in a boosted frame,
        # boost these quantities
        if boost is not None:
            self.baseline_z, = boost.static_length( [ self.baseline_z ] )
            self.vz, = boost.velocity( [ self.vz ] )

        # Record laser properties
        self.E0 = E0
        self.w0 = w0
        self.k0 = k0
        self.ctau = ctau
        self.z0 = z0
        self.zf = zf
        self.theta_pol = theta_pol
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
        # Move the position of the antenna (element-wise array operation)
        self.baseline_z += hdt * self.vz

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
            z_lab = gamma0*( self.baseline_z - beta0*t )
            t_lab = gamma0*( t - beta0*self.z_antenna )
        else:
            z_lab = self.baseline_z
            t_lab = t

        # Calculate the electric field to be emitted (in the lab-frame)
        # Eu is the amplitude along the polarization direction
        # Note that we neglect the excursion of the particles when
        # calculating the electric field on the particles. This is because
        # the excursion is typically small and because virtual negative and
        # positive particles have opposite excursion which would require 
        # calling this function twice.
        Eu = self.E0 * gaussian_profile( z_lab, self.baseline_r, t_lab,
                        self.w0, self.ctau, self.z0, self.zf,
                        self.k0, boost=None, output_Ez_profile=False )

        # Calculate the corresponding velocity. This takes into account
        # lab-frame to boosted-frame conversion, through a modification
        # of the mobility coefficient: see the __init__ function
        self.vx = ( self.mobility_coef * np.cos(self.theta_pol) ) * Eu
        self.vy = ( self.mobility_coef * np.sin(self.theta_pol) ) * Eu

    def deposit( self, fld, fieldtype ):
        """
        TO BE COMPLETED
        """
        # Check if z_antenna is in the current physical domain        
        # Calculate the displacement of the particles
        
        pass
    
    
