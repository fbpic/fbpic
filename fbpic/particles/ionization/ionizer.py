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
from scipy.constants import c, e, m_e, physical_constants
from scipy.special import

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
                    z_min, z_max, full_initialization=True ):
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
        self.z_min = z_min
        self.use_cuda = ionizable_species.use_cuda

        # Initialize ionization-relevant meta-data
        if full_initialization:
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

        See Chen, JCP 236 (2013), equation (2)
        """
        # Check whether the element string is valid
        if element in ionization_energies_dict:
            self.element = element
        else:
            raise ValueError("Unknown ionizable element %s.\n" %element + \
            "Please use atomic symbol (e.g. 'He') not full name (e.g. Helium)")
        # Get the array of energies
        Uion = ionization_energies_dict[element]

        # Determine the maximum level of ionization
        if z_max is None:
            self.z_max = len(Uion)
        else:
            self.z_max = min( z_max, len(Uion) )

        # Calculate the ADK prefactors (See Chen, JCP 236 (2013), equation (2))
        # - Scalars
        alpha = physical_constants['fine-structure constant'][0]
        r_e = physical_constants['classical electron radius'][0]
        wa = alpha**3 * c * r_e
        Ea = m_e*c**2/e * alpha**4/r_e
        # - Arrays (one element per ionization level)
        UH = ionization_energies_dict['H'][0]
        Z = np.arange( len(Uion) ) + 1
        n_eff = Z * np.sqrt( UH/Uion )
        l_eff = n_eff[0] - 1
        C2 = 2**(2*n_eff) / (n_eff * gamma(n_eff+l_eff+1) * gamma(n_eff-l_eff))
        # For now, we assume l=0, m=0
        self.adk_power = - (2*n_eff - 1)
        self.adk_prefactor = dt * wa * C2 * ( Uion/(2*UH) ) \
            * ( 2*(U_ion/UH)**(3./2)*Ea )**(2*n_eff - 1)
        self.adk_exp_prefactor = -2./3 * ( Uion/UH )**(3./2) * Ea

        # Prepare random number generator
        if self.use_cuda:
            self.adk_power = cuda.to_device( self.adk_power )
            self.adk_prefactor = cuda.to_device( self.adk_prefactor )
            self.adk_exp_prefactor = cuda.to_device( self.adk_exp_prefactor )
            self.prng = rand.PRNG()

    def handle_ionization( self,  ):
        """
        # TODO: Complete
        """
        Ntot = self.Ntot
        # Process particles in batches (of typically 10, 20 particles)
        N_batches = Ntot / self.batch_size + 1

        if self.use_cuda:
            # Create temporary arrays
            is_ionized = cuda.device_array( Ntot, dtype=np.int16 )
            n_ionized = cuda.device_array( N_batches, dtype=np.int64 )
            # Draw random numbers
            random_draw = cuda.device_array( Ntot, dtype=np.float16 )
            self.prng.uniform( random_draw )

            # Ionize the ions
            ionize_ions_cuda[ tpb, bpg ](
                ion.ux, ion.uy, ion.uz, ion.Ex,
                ion.Ey, ion.Ez, ion.Bx, ion.By, ion.Bz,
                )

            # Count the total number of electrons (operation performed
            # on the CPU, as this is typically difficult on the GPU)
            n_ionized = n_ionized.copy_to_host()
            cumulative_n_ionized = np.zeros( len(n_ionized)+1, dtype=np.int64 )
            np.cumsum( n_ionized, out=cumulative_n_ionized[1:] )

            # Reallocate the electron species
            elec = self.target_species
            old_Ntot = elec.Ntot
            new_Ntot = old_Ntot + cumulative_n_ionized[-1]
            # Copy old particles (existing kernel?)
            cumulative_n_ionized = cuda.to_device( cumulative_n_ionized )

            # Copy new particles (one thread per batch)
            tpb, bpg = ...


def copy_ionized_electrons_cuda( N_batch, elec_old_Ntot, ion_Ntot,
    elec_x, elec_y, elec_z, elec_ux, elec_uy, elec_uz, elec_w,
    ion_x, ion_y, ion_z, ion_ux, ion_uy, ion_uz, ion_neutral_weight ):

    # Select the current batch
    ibatch = cuda.
    if ibatch < N_batch:
        copy_batch_cuda( i_batch, elec_old_Ntot, ion_Ntot,
            elec_x, elec_y, elec_z, elec_ux, elec_uy, elec_uz, elec_w,
            ion_x, ion_y, ion_z, ion_ux, ion_uy, ion_uz, ion_neutral_weight )


def copy_ionized_electrons_cuda( N_batch, elec_old_Ntot, ion_Ntot,
    elec_x, elec_y, elec_z, elec_ux, elec_uy, elec_uz, elec_w,
    ion_x, ion_y, ion_z, ion_ux, ion_uy, ion_uz, ion_neutral_weight ):

    # Select the current batch
    ibatch = cuda.
    if ibatch < N_batch:
        copy_ionized_electrons_batch_cuda( i_batch, elec_old_Ntot, ion_Ntot,
            elec_x, elec_y, elec_z, elec_ux, elec_uy, elec_uz, elec_w,
            ion_x, ion_y, ion_z, ion_ux, ion_uy, ion_uz, ion_neutral_weight )

def ionize_ions_cuda():

    ibatch = cuda.

    if ibatch < N_batch:
        # Set the number of ionized particles in the batch to 0
        n_ionized[ibatch] = 0

        # Loop through the batch
        N_max = min( (ibatch+1)*N_batches, Ntot )
        for ip in range( ibatch*N_batches, N_max ):

            # Skip the ionization routine, if the maximal ionization level
            # has already been reached for this macroparticle
            level = ionization_level[ip]
            if level >= z_max:
                continue

            # Calculate the amplitude of the electric field,
            # in the frame of the electrons
            E = get_E_amplitude_cuda( ux[ip], uy[ip], uz[ip],
                    Ex[ip], Ey[ip], Ez[ip], c*Bx[ip], c*By[ip], c*Bz[ip] )
            # Get ADK rate
            p = get_ionisation_probability_cuda( E, adk_prefactor[level],
                                power[level], exp_prefactor[level] )
            # Ionize particles
            if random_draw < p:
                # Set the corresponding flag and update particle count
                is_ionized[ip] = 1
                n_ionized[ibatch] += 1
                # Update the ionization level and the corresponding weight
                ionization_level[ip] += 1
                w[ip] = e * ionization_level[ip] * neutral_weight[ip]
            else:
                is_ionized[ip] = 0

    def send_to_gpu( self ):
        """
        Copy the ionization data to the GPU.
        """
        # TODO: complete: add all required arrays
        if self.use_cuda:
            self.ionization_level = cuda.to_device( self.ionization_level )
            self.neutral_weight = cuda.to_device( self.neutral_weight )


    def receive_from_gpu( self ):
        """
        Receive the ionization data from the GPU.
        """
        # TODO: complete: add all required arrays
        if self.use_cuda:
            self.ionization_level = self.ionization_level.copy_to_host()
            self.neutral_weight = self.neutral_weight.copy_to_host()
