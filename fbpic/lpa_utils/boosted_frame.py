# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the transformation required to perform a boosted frame simulation
"""
import numpy as np
from scipy.constants import c

class BoostConverter( object ):
    """
    Class that converts the parameters of a simulation
    from the lab frame to the boosted frame.
    """

    def __init__(self, gamma0):
        """
        Initialize the object by calculating the velocity of the
        boosted frame.

        Parameters
        ----------
        gamma0: float
            Lorentz factor of the boosted frame
        """
        self.gamma0 = gamma0
        self.beta0 = np.sqrt( 1 - 1./gamma0**2 )

    # Length and density
    # ------------------

    def static_length( self, lab_frame_vars ):
        """
        Converts a list of lengths that correspond to static objects
        (e.g. the plasma) from the lab frame to the boosted frame, i.e:
        L' = L / gamma0

        Parameters
        ----------
        lab_frame_vars: list of floats
            A list containing several length to be converted

        Returns
        -------
        A list with same number of elements, with the converted quantities
        """
        boosted_frame_vars = []
        for length in lab_frame_vars:
            boosted_frame_vars.append( length/self.gamma0 )

        return( boosted_frame_vars )

    def copropag_length( self, lab_frame_vars, beta_object=1. ):
        """
        Converts a list of lengths that correspond to copropagating objects
        (e.g. the laser) from the lab frame to the boosted frame, i.e:
        L' = L / [ gamma0*(1 - beta_object*beta0) ]

        Parameters
        ----------
        lab_frame_vars: list of floats
            A list containing several length to be converted

        beta_object: float, optional
            The normalized velocity of the object whose
            length is being converted

        Returns
        -------
        A list with the same number of elements, with the converted quantities
        """
        convert_factor = 1./( self.gamma0*(1. - self.beta0*beta_object) )
        boosted_frame_vars = []
        for length in lab_frame_vars:
            boosted_frame_vars.append( length * convert_factor )

        return( boosted_frame_vars )

    def static_density( self, lab_frame_vars ):
        """
        Converts a list of densities that correspond to static objects
        (e.g. the plasma) from the lab frame to the boosted frame, i.e:
        n' = n * gamma0

        Parameters
        ----------
        lab_frame_vars: list of floats
            A list containing several length to be converted

        Returns
        -------
        A list with the same number of elements, with the converted quantities
        """
        boosted_frame_vars = []
        for dens in lab_frame_vars:
            boosted_frame_vars.append( dens * self.gamma0 )

        return( boosted_frame_vars )

    def copropag_density( self, lab_frame_vars, beta_object=1. ):
        """
        Converts a list of densities that correspond to copropagating objects
        (e.g. an electron bunch) from the lab frame to the boosted frame, i.e:
        n' = n * [ gamma0*(1 - beta_object*beta0) ]

        Parameters
        ----------
        lab_frame_vars: list of floats
            A list containing several densities to be converted

        beta_object: float, optional
            The normalized velocity of the object whose
            density is being converted

        Returns
        -------
        A list with the same number of elements, with the converted quantities
        """
        convert_factor = self.gamma0*(1. - self.beta0*beta_object)
        boosted_frame_vars = []
        for dens in lab_frame_vars:
            boosted_frame_vars.append( dens * convert_factor )

        return( boosted_frame_vars )

    # Momentum and velocity
    # ---------------------

    def velocity( self, lab_frame_vars ):
        """
        Converts a list of velocities from the lab frame to the boosted frame:
        v' = ( v - c * beta0 )/( 1 - beta0*v/c )

        Parameters
        ----------
        lab_frame_vars: list of floats
            A list containing several velocities to be converted

        Returns
        -------
        A list with the same number of elements, with the converted quantities
        """
        boosted_frame_vars = []
        for v in lab_frame_vars:
            boosted_frame_vars.append( (v-c*self.beta0)/(1-v*self.beta0/c) )

        return( boosted_frame_vars )

    def longitudinal_momentum( self, lab_frame_vars ):
        """
        Converts a list of momenta from the lab frame to the boosted frame:
        u_z' = gamma0 * ( u_z - sqrt(1 + u_z**2) * beta0 )

        Warning: The above formula assumes that the corresponding
        particle has no transverse motion at all.

        Parameters
        ----------
        lab_frame_vars: list of floats
            A list containing several momenta to be converted

        Returns
        -------
        A list with the same number of elements, with the converted quantities
        """
        boosted_frame_vars = []
        for u_z in lab_frame_vars:
            g = np.sqrt( 1 + u_z**2 )
            boosted_frame_vars.append( self.gamma0*( u_z - g*self.beta0 ) )

        return( boosted_frame_vars )

    def gamma( self, lab_frame_vars ):
        """
        Converts a list of Lorentz factors from the lab frame to the
        boosted frame:
        gamma' = gamma0 * ( gamma - beta0 * sqrt( gamma**2 - 1 ) )

        Warning: The above formula assumes that the corresponding
        particle has no transverse motion at all.

        Parameters
        ----------
        lab_frame_vars: list of floats
            A list containing several Lorentz factors to be converted

        Returns
        -------
        A list with the same number of elements, with the converted quantities
        """
        boosted_frame_vars = []
        for g in lab_frame_vars:
            uz = np.sqrt( g**2 - 1 )
            boosted_frame_vars.append( self.gamma0*( g - self.beta0*uz ) )

        return( boosted_frame_vars )

    # Laser quantities
    # ----------------

    def wavenumber( self, lab_frame_vars ):
        """
        Converts a list of wavenumbers from the lab frame to the boosted frame:
        k' = k / ( gamma0 ( 1 + beta0 ) )

        Parameters
        ----------
        lab_frame_vars: list of floats
            A list containing several wavenumbers to be converted

        Returns
        -------
        A list with the same number of elements, with the converted quantities
        """
        boosted_frame_vars = []
        for k in lab_frame_vars:
            boosted_frame_vars.append( k /( self.gamma0*( 1 + self.beta0 ) ) )

        return( boosted_frame_vars )

    def boost_particle_arrays( self, x, y, z, ux, uy, uz, inv_gamma ):
        """
        Transforms particles to the boosted frame and propagates
        them to a fixed time t_boost = 0. without taking any electromagnetic
        effects into account. (This is useful for the initialization
        of a bunch distribution in the boosted frame.)

        Parameters
        ----------
        x, y, z: 1darray of float (in meter)
            The position of the particles in the lab frame
            (One element per macroparticle)
        ux, uy, uz: 1darray of floats (dimensionless)
            The momenta of the particles
            (One element per macroparticle)
        inv_gamma: 1darray of floats (dimensionless)
            The inverse of the Lorentz factor
            (One element per macroparticle)

        Returns
        -------
        The same arrays (in the same order) but in the boosted frame
        """
        # Apply a Lorentz boost to the particle distribution.
        # In the Lorentz boosted frame, particles will not be defined
        # at a single time t'. Therefore, move of all particles to
        # a single time t' = const. in the boosted frame.
        uz_boost = self.gamma0*self.beta0

        # Transform particle times and longitudinal positions
        # to the boosted frame. Assumes a lab time t = 0.
        t_boost = -uz_boost*z/c
        z_boost = self.gamma0*z

        # Get particle Lab velocities
        vx = ux*inv_gamma*c
        vy = uy*inv_gamma*c
        vz = uz*inv_gamma*c

        # Calculate boost factor for velocities
        boost_fact = 1./(1.-self.beta0*vz/c)

        # Transform velocities to boosted frame
        vx_boost = vx*boost_fact/self.gamma0
        vy_boost = vy*boost_fact/self.gamma0
        vz_boost = (vz-self.beta0*c)*boost_fact

        # Correct for shift in t_boost, which comes from transformation
        # into boosted frame. Particles are moved with transformed
        # velocities to a boosted time t'=0.
        new_x = x - t_boost * vx_boost
        new_y = y - t_boost * vy_boost
        new_z = z_boost - t_boost * vz_boost

        # Get final quantities
        new_inv_gamma = np.sqrt(1.-(vx_boost**2+vy_boost**2+vz_boost**2)/c**2)
        new_ux = vx_boost / (new_inv_gamma * c)
        new_uy = vy_boost / (new_inv_gamma * c)
        new_uz = vz_boost / (new_inv_gamma * c)

        return( new_x, new_y, new_z, new_ux, new_uy, new_uz, new_inv_gamma )

    # Miscellaneous
    # -------------

    def interaction_time( self, L_interact, l_window, v_window ):
        """
        Calculates the interaction time in the boosted frame:
        (Time it takes for the moving window to slide once across the
        total interaction length, i.e. the plasma)

        L_interact' = L_interact / gamma0
        l_window' = l_window * gamma0 * ( 1 + beta0 )
        v_window' = ( v_window - c * beta0 )/( 1 - beta0*v_window/c )
        v_plasma' = -beta0 * c

        T_interact' = L_interact' + l_window' / ( v_window' - v_plasma' )

        Parameters
        ----------
        L_interact: float (in meter)
            The total interaction length (typically the length of the plasma)
            in the lab frame.

        l_window: float (in meter)
            The length of the moving window in the lab frame.

        v_window: float (in meter/second)
            The velocity of the moving window in the lab frame.

        Returns
        -------
        T_interact : float (in seconds)
            The interaction time in the boosted frame.
        """
        # Boost lab quantities
        L_i, = self.static_length( [L_interact] )
        l_w, = self.copropag_length( [l_window] )
        v_w, = self.velocity( [v_window] )
        v_p = -self.beta0*c
        # Interaction time
        T_interact = (L_i+l_w)/(v_w-v_p)

        return( T_interact )
