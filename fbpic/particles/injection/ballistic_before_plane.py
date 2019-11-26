# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a class for particle injection "through a plane".
"""
from scipy.constants import c

class BallisticBeforePlane( object ):
    """
    Class that defines particle injection "though a plane".
    In practice, when using this injection method, particles
    move ballistically before crossing a given plane.

    This is useful when running boosted-frame simulation, whereby a
    relativistic particle beam is initialized in vacuum and later enters the
    plasma. In this case, the particle beam may feel its own space charge
    force for a long distance (in the boosted-frame), which may alter its
    properties. Imposing that particles move ballistically before a plane
    (which corresponds to the entrance of the plasma) ensures that the
    particles do not feel this space charge force.
    """

    def __init__(self, z_plane_lab, boost):
        """
        Initialize the parameters of the plane.

        Parameters
        ----------
        z_plane_lab: float (in meters)
            The (fixed) position of the plane, in the lab frame

        boost: a BoostConverter object, optional
            Defines the Lorentz boost of the simulation.
        """
        # Register the parameters of the plane
        self.z_plane_lab = z_plane_lab
        if boost is not None:
            self.inv_gamma_boost = 1./boost.gamma0
            self.beta_boost = boost.beta0
        else:
            self.inv_gamma_boost = 1.
            self.beta_boost = 0.


    def get_current_plane_position( self, t ):
        """
        Get the current position of the plane, in the frame of the simulation

        Parameters:
        -----------
        t: float (in seconds)
            The time in the frame of the simulation
        Returns:
        --------
        z_plane: float (in meters)
            The position of the plane at t
        """
        z_plane = self.inv_gamma_boost*self.z_plane_lab - self.beta_boost*c*t
        return( z_plane )
