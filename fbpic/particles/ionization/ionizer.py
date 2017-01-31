# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure and methods associated with atomic ionization.

#TODO A few words on the implementation: all levels are kept in the same species
"""
import numpy as np
from numba import cuda
from .atomic_data import ionization_energies_dict

class Ionizer(object):
    """
    Class that contains the data associated with ionization (on the ions side)

    Main attributes
    ---------------
    - ionization_level: 1darray of int16 (one element per particle)
      which contains the ionization state of each particle
    - TODO: complete
    """
    def __init__( self, element, ionizable_species, target_species,
                    z_min, z_max ):
        """
        # TODO: complete

        Parameters
        ----------
        element: string
            The atomic symbol of the considered ionizable species
            (e.g. 'He', 'N' ;  do not use 'Helium' or 'Nitrogen')

        ionizable_species: an fbpic.Particles object
            This object is not modified or registered.
            It is only used in order to pass a number of additional argument.

        target_species: an fbpic.Particles object
            This object is not modified when creating the class, but
            it is modified when ionization occurs
            (i.e. more particles are created)
        """
        # Register a few parameters
        self.target_species = target_species
        self.use_cuda = ionizable_species.use_cuda

        # Initialize ionization-relevant meta-data
        self.initialize_ADK_parameters( element, z_max,
                                            ionizable_species.dt )

        # Initialize the required arrays
        Ntot = ionizable_species.Ntot
        self.ionization_level = np.ones( Ntot, dtype=np.int16 ) * z_min
        self.neutral_weight = ionizable_species.w/ionizable_species.q

        # Allocate arrays and register variables when using CUDA
        # TODO complete
        if self.use_cuda:
            pass

    def initialize_ADK_parameters( self, element, z_max, dt ):
        """
        # TODO complete
        """
        # Check whether the element string is valid
        if element in ionization_energies_dict:
            self.element = element
        else:
            raise ValueError("Unknown ionizable element %s.\n" %element + \
            "Please use atomic symbol (e.g. 'He') not full name (e.g. Helium)")
        # Get the array of energies
        energies = ionization_energies_dict[element]

        # Determine the maximum level of ionization
        if z_max is None:
            self.z_max = len(energies)
        else:
            self.z_max = min( z_max, len(energies) )

        # Calculate the ADK prefactor
#        EH = ionization_energies_dict['H'][0]


    def send_particles_to_gpu( self ):
        """
        Copy the ionization data to the GPU.
        """
        # TODO: complete: add all required arrays
        if self.use_cuda:
            self.ionization_level = cuda.to_device( self.ionization_level )
            self.neutral_weight = cuda.to_device( self.neutral_weight )

    def receive_particles_from_gpu( self ):
        """
        Receive the particles from the GPU.
        """
        # TODO: complete: add all required arrays
        if self.use_cuda:
            self.ionization_level = self.ionization_level.copy_to_host()
            self.neutral_weight = self.neutral_weight.copy_to_host()
