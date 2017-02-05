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
from scipy.constants import c, e, m_e, physical_constants
from scipy.special import gamma
from .atomic_data import ionization_energies_dict
from .numba_methods import ionize_ions_numba, copy_ionized_electrons_numba
try:
    from fbpic.cuda_utils import cuda_tpb_bpg_1d
    from .cuda_methods import ionize_ions_cuda, \
        copy_ionized_electrons_cuda, copy_particle_data_cuda
    cuda_installed = True
except ImportError:
    cuda_installed = False

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
        # Process ionized particles into batches
        self.batch_size = 10

        # Initialize ionization-relevant meta-data
        if full_initialization:
            self.initialize_ADK_parameters( element, z_max,
                                            ionizable_species.dt )

        # Initialize the required arrays
        Ntot = ionizable_species.Ntot
        self.ionization_level = np.ones( Ntot, dtype=np.uint64 ) * z_min
        self.neutral_weight = ionizable_species.w/ionizable_species.q

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
        wa = alpha**3 * c / r_e
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
            * ( 2*(Uion/UH)**(3./2)*Ea )**(2*n_eff - 1)
        self.adk_exp_prefactor = -2./3 * ( Uion/UH )**(3./2) * Ea

        # Prepare random number generator
        if self.use_cuda:
            self.prng = cuda.rand.PRNG()

    def handle_ionization_gpu( self, ion ):
        """
        # TODO: Complete
        """
        # Process particles in batches (of typically 10, 20 particles)
        N_batch = int( ion.Ntot / self.batch_size ) + 1

        # Create temporary arrays
        is_ionized = cuda.device_array( ion.Ntot, dtype=np.int16 )
        n_ionized = cuda.device_array( N_batch, dtype=np.int64 )
        # Draw random numbers
        random_draw = cuda.device_array( ion.Ntot, dtype=np.float32 )
        self.prng.uniform( random_draw )

        # Ionize the ions (one thread per batch)
        batch_grid_1d, batch_block_1d = cuda_tpb_bpg_1d( N_batch )
        ionize_ions_cuda[ batch_grid_1d, batch_block_1d ](
            N_batch, self.batch_size, ion.Ntot, self.z_max,
            n_ionized, is_ionized, self.ionization_level, random_draw,
            self.adk_prefactor, self.adk_power, self.adk_exp_prefactor,
            ion.ux, ion.uy, ion.uz,
            ion.Ex, ion.Ey, ion.Ez,
            ion.Bx, ion.By, ion.Bz,
            ion.w, self.neutral_weight )

        # Count the total number of electrons (operation performed
        # on the CPU, as this is typically difficult on the GPU)
        n_ionized = n_ionized.copy_to_host()
        cumulative_n_ionized = np.zeros( len(n_ionized)+1, dtype=np.int64 )
        np.cumsum( n_ionized, out=cumulative_n_ionized[1:] )
        # If no new particle was created, skip the rest of this function
        if cumulative_n_ionized[-1] == 0:
            return

        # Reallocate the electron species, in order to
        # accomodate the electrons produced by ionization
        elec = self.target_species
        old_Ntot = elec.Ntot
        new_Ntot = old_Ntot + cumulative_n_ionized[-1]
        # Iterate over particle attributes and copy the old electrons
        # (one thread per particle)
        ptcl_grid_1d, ptcl_block_1d = cuda_tpb_bpg_1d( old_Ntot )
        for attr in ['x', 'y', 'z', 'ux', 'uy', 'uz', 'w', 'inv_gamma',
                        'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
            old_array = getattr(elec, attr)
            new_array = cuda.device_array( new_Ntot, dtype=np.float64 )
            copy_particle_data_cuda[ ptcl_grid_1d, ptcl_block_1d ](
                old_Ntot, old_array, new_array )
            setattr( elec, attr, new_array )
        # Allocate the auxiliary arrays
        self.cell_idx = cuda.device_array( new_Ntot, dtype=np.int32)
        self.sorted_idx = cuda.device_array( new_Ntot, dtype=np.uint32)
        self.sorting_buffer = cuda.device_array( new_Ntot, dtype=np.float64 )
        # Modify the total number of electrons
        elec.Ntot = new_Ntot
        # TODO: Generate particle ids on the GPU

        # Copy the new electrons from ionization (one thread per batch)
        copy_ionized_electrons_cuda[ batch_grid_1d, batch_block_1d ](
            N_batch, self.batch_size, old_Ntot, ion.Ntot,
            cumulative_n_ionized, is_ionized,
            elec.x, elec.y, elec.z, elec.inv_gamma,
            elec.ux, elec.uy, elec.uz, elec.w,
            elec.Ex, elec.Ey, elec.Ez, elec.Bx, elec.By, elec.Bz,
            ion.x, ion.y, ion.z, ion.inv_gamma,
            ion.ux, ion.uy, ion.uz, self.neutral_weight,
            ion.Ex, ion.Ey, ion.Ez, ion.Bx, ion.By, ion.Bz )
        elec.sorted = False

    def handle_ionization_cpu( self, ion ):
        """
        # TODO: Complete
        """
        # Process particles in batches (of typically 10, 20 particles)
        N_batch = int( ion.Ntot / self.batch_size ) + 1

        # Create temporary arrays
        is_ionized = np.empty( ion.Ntot, dtype=np.int16 )
        n_ionized = np.empty( N_batch, dtype=np.int64 )
        # Draw random numbers
        random_draw = np.random.rand( ion.Ntot )

        # Ionize the ions (one thread per batch)
        ionize_ions_numba(
            N_batch, self.batch_size, ion.Ntot, self.z_max,
            n_ionized, is_ionized, self.ionization_level, random_draw,
            self.adk_prefactor, self.adk_power, self.adk_exp_prefactor,
            ion.ux, ion.uy, ion.uz,
            ion.Ex, ion.Ey, ion.Ez,
            ion.Bx, ion.By, ion.Bz,
            ion.w, self.neutral_weight )

        # Count the total number of electrons
        cumulative_n_ionized = np.zeros( len(n_ionized)+1, dtype=np.int64 )
        np.cumsum( n_ionized, out=cumulative_n_ionized[1:] )
        # If no new particle was created, skip the rest of this function
        if cumulative_n_ionized[-1] == 0:
            return

        # Reallocate the electron species, in order to
        # accomodate the electrons produced by ionization
        elec = self.target_species
        old_Ntot = elec.Ntot
        new_Ntot = old_Ntot + cumulative_n_ionized[-1]
        # Iterate over particle attributes and copy the old electrons
        # (one thread per particle)
        for attr in ['x', 'y', 'z', 'ux', 'uy', 'uz', 'w', 'inv_gamma',
                            'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
            old_array = getattr(elec, attr)
            new_array = np.empty( new_Ntot, dtype=np.float64 )
            new_array[:old_Ntot] = old_array
            setattr( elec, attr, new_array )
        # Allocate the auxiliary arrays
        self.cell_idx = np.empty( new_Ntot, dtype=np.int32)
        self.sorted_idx = np.empty( new_Ntot, dtype=np.uint32)
        self.sorting_buffer = np.empty( new_Ntot, dtype=np.float64 )
        # Modify the total number of electrons
        elec.Ntot = new_Ntot
        # TODO: Generate particle ids on the GPU

        # Copy the new electrons from ionization (one thread per batch)
        copy_ionized_electrons_numba(
            N_batch, self.batch_size, old_Ntot, ion.Ntot,
            cumulative_n_ionized, is_ionized,
            elec.x, elec.y, elec.z, elec.inv_gamma,
            elec.ux, elec.uy, elec.uz, elec.w,
            elec.Ex, elec.Ey, elec.Ez, elec.Bx, elec.By, elec.Bz,
            ion.x, ion.y, ion.z, ion.inv_gamma,
            ion.ux, ion.uy, ion.uz, self.neutral_weight,
            ion.Ex, ion.Ey, ion.Ez, ion.Bx, ion.By, ion.Bz )

    def send_to_gpu( self ):
        """
        Copy the ionization data to the GPU.
        """
        # TODO: complete: add all required arrays
        if self.use_cuda:
            self.ionization_level = cuda.to_device( self.ionization_level )
            self.neutral_weight = cuda.to_device( self.neutral_weight )
            # Small-size arrays with ADK parameters
            self.adk_power = cuda.to_device( self.adk_power )
            self.adk_prefactor = cuda.to_device( self.adk_prefactor )
            self.adk_exp_prefactor = cuda.to_device( self.adk_exp_prefactor )

    def receive_from_gpu( self ):
        """
        Receive the ionization data from the GPU.
        """
        # TODO: complete: add all required arrays
        if self.use_cuda:
            self.ionization_level = self.ionization_level.copy_to_host()
            self.neutral_weight = self.neutral_weight.copy_to_host()
            # Small-size arrays with ADK parameters
            self.adk_power = self.adk_power.copy_to_host()
            self.adk_prefactor = self.adk_prefactor.copy_to_host()
            self.adk_exp_prefactor = self.adk_exp_prefactor.copy_to_host()
