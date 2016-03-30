"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the class LaserAntenna, which can be used to continuously
emit a laser during a simulation.
"""

class LaserAntenna(object):
    """
    TO BE COMPLETED
    """

    def __init__( self, interp ):
        """
        TO BE COMPLETED
        """

        # Initialize the virtual particles along an 8-branch star,
        # with n_las particle per cell
        dr_particles = interp.dr/nlas
        N_particles = interp.Nr*nlas
        r_particles = 0.5*dr_particles + dr_particles*np.arange( N_particles )

        self.x = 0
        self.y = 0
        self.ux = 0
        self.uy = 0
        self.inv_gamma = 0
        # Initialize position and "momentum" of the antenna
        self.z_antenna = 0
        self.uz_antenna = 0

        # Calculate the weight of the particles

        
        
    def deposit( self ):
        """
        TO BE COMPLETED
        """
        # Check if z_antenna is in the current physical domain

        # Calculate the displacement of the particles
        
        pass
    

    

    # See add_laser_work in em3dsolver.py

    
