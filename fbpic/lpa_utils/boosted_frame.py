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


    def curvature( self, lab_frame_vars ):
        """
        Converts a list of laser curvature from the lab frame
        to the boosted frame:
        R' = R / ( 1 + beta0 )

        Parameters
        ----------
        lab_frame_vars: list of floats
            A list containing several curvatures to be converted

        Returns
        -------
        A list with the same number of elements, with the converted quantities
        """
        boosted_frame_vars = []
        for R in lab_frame_vars:
            boosted_frame_vars.append( R /( 1 + self.beta0 ) )

        return( boosted_frame_vars )
