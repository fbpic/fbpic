# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure and methods associated with atomic ionization.

The implemented ionization model is the ADK model. The implementation
is fully relativistic (i.e. it works in the boosted-frame as well).

Ionization is implemented by keeping ions at different ionization states in
the same Particles object, so that the number of macroparticles in this object
remains constant and arrays do not have to be reallocated (except when the
moving window creates new particles.) An array `ionization_level` keeps track
of the ionization state of each macroparticle.

On the other hand, the electrons generated by ionization do need to be added to
an existing Particles object, and this implies that the number of
macroparticles in the object does not remain constant and that the arrays are
reallocated.

In addition, at each PIC iteration, the number of new electrons need to
be counted in order to reallocate the array and copy the electrons to
the right indices. This basically involves a cumulative sum operation, which
difficult to implement on the GPU. For this reason, the corresponding array is
sent to the CPU, which performs the cumulative sum and sends the data back to
the GPU. In order, to limit the amount of data to be transfered, particles are
handled in batches of 10 particles, so that only the cumulative sum of the
number of particles in each batch need to be performed.
"""
import numpy as np
from scipy.constants import c, e, m_e, physical_constants
from scipy.special import gamma
from .read_atomic_data import get_ionization_energies
from .numba_methods import ionize_ions_numba, copy_ionized_electrons_numba
from ..cuda_numba_utils import allocate_empty, reallocate_and_copy_old, \
                                perform_cumsum_2d, generate_new_ids
from ...spin.numba_methods import copy_ionized_electron_spin_numba

# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
from fbpic.utils.printing import catch_gpu_memory_error
if cuda_installed:
    import cupy
    from fbpic.utils.cuda import cuda_tpb_bpg_1d
    from .cuda_methods import ionize_ions_cuda, copy_ionized_electrons_cuda
    from ...spin.cuda_methods import copy_ionized_electron_spin_cuda


class Ionizer(object):
    """
    Class that contains the data associated with ionization (on the ions side)
    and has method to calculate the ionization probability.

    The implemented ionization model is the ADK model. The implementation
    is fully relativistic (i.e. it works in the boosted-frame as well).

    This class also handles spin tracking. In case of an ion getting
    ionized, the first electron will have the same spin as the ion, whereas
    all subseqeuntly ionized electrons will have a random spin direction
    (generated via sphere point picking).

    Main attributes
    ---------------
    - ionization_level: 1darray of integers (one element per particle)
      which contains the ionization state of each particle
    - w_times_level: 1darray of floats (one element per particle)
      which contains the number of physical particle that correspond to each
      macroparticle, multiplied by the ionization level. (This is updated
      whenever further ionization happens, and is passed to the deposition
      kernel as the effective weight of the particles)
    """
    def __init__(self, element, ionizable_species, target_species,
                 level_start, level_max=None):
        """
        Initialize an Ionizer instance

        Parameters
        ----------
        element: string
            The atomic symbol of the considered ionizable species
            (e.g. 'He', 'N' ;  do not use 'Helium' or 'Nitrogen')

        ionizable_species: an fbpic.Particles object
            This object is not modified or registered.
            It is only used in order to pass a number of additional argument.

        target_species: a `Particles` object, or a dictionary of `Particles`
            Stores the electron macroparticles that are created in
            the ionization process.

            - If a single `Particles` object is passed, than electrons from all
            ionization levels are stored into this object.

            - If a dictionary is passed, then its keys should be integers
            (corresponding to the ionizable levels of `element`, starting
            at `level_start`), and its values should be `Particles` objects.
            In this case, the electrons from each distinct ionizable level
            will be stored into these separate objects. Note that using
            separate objects will typically require longer computing time.

            These objects are not modified when creating the class, but
            they are when ionization occurs (i.e. more particles are created)

        level_start: int
            The ionization level at which the macroparticles are initially
            (e.g. 0 for initially neutral atoms)

        level_max: int, optional
            If not None, defines the maximum ionization level that
            macroparticles can reach. Should not exceed the physical
            limit for the chosen element.
        """
        # Register a few parameters
        self.level_start = level_start
        self.level_max = level_max
        self.use_cuda = ionizable_species.use_cuda
        # Process ionized particles into batches
        self.batch_size = 10

        # Initialize ionization-relevant meta-data
        self.initialize_ADK_parameters( element, ionizable_species.dt )

        # Initialize the required arrays
        Ntot = ionizable_species.Ntot
        self.ionization_level = np.ones( Ntot, dtype=np.uint64 ) * level_start
        self.w_times_level = ionizable_species.w * self.ionization_level

        # Check if electrons from different ionization levels should
        # be stored into separate species
        if type(target_species) is dict:
            # When passing a dictionary
            # Check that the keys are the right integers
            for level in range(self.level_start, self.level_max):
                if level not in target_species.keys():
                    raise ValueError(
                    'When passing a dictionary for `target_species`, its keys '
                    'should be\nthe integers corresponding to the ionizable '
                    'levels.\n (i.e. the integers from %d to %d'
                    'for %s with level_start=%d.)' %(self.level_start,
                    self.level_max, element, self.level_start))
                # Check that the dictionary contains Particles objects
                assert isinstance(target_species[level], type(ionizable_species))
            # Convert to a list internally: the dictionary input is
            # just for less error-prone user input.
            self.target_species = [ target_species[level] \
                for level in range(self.level_start, self.level_max) ]
            self.store_electrons_per_level = True
        elif isinstance(target_species, type(ionizable_species)):
            # When passing a single Particles object
            self.target_species = [target_species]  # List of one element
            self.store_electrons_per_level = False
        else:
            raise ValueError(
                "Unexpected type for target_species: %s\n"
                "Please pass a `Particles` object, or a dictionary"
                %type(target_species))

        # Check that the target species are indeed electrons
        for species in self.target_species:
            assert species.q == -e
            assert species.m == m_e
            # Another sanity check: if spin tracking is enabled for
            # the parent ion, make sure it is activated for electrons, too
            if species.spin_tracker is None:
                species.activate_spin_tracking(
                    anom=physical_constants['electron mag. mom. anomaly'][0])

    def initialize_ADK_parameters( self, element, dt ):
        """
        Initialize parameters needed for the calculation of ADK ionization rate

        Parameters
        ----------
        element: string
            The atomic symbol of the considered ionizable species
            (e.g. 'He', 'N' ;  do not use 'Helium' or 'Nitrogen')

        dt: float (in seconds)
            The timestep of the simulation. (The calculated ionization
            probability is a probability *per timestep*.)

        See Chen, JCP 236 (2013), equation (2) for the ionization rate formula
        """
        # Get the array of energies
        Uion = get_ionization_energies( element )
        # Check whether the element string was valid
        if Uion is None:
            raise ValueError("Unknown ionizable element %s.\n" %element + \
            "Please use atomic symbol (e.g. 'He') not full name (e.g. Helium)")
        else:
            self.element = element

        # Determine and set the maximum level of ionization
        if self.level_max is None:
            self.level_max = len(Uion)
        else:
            assert type(self.level_max) is int, "level_max must be integer"
            if self.level_max>len(Uion):
                raise ValueError("Chosen level_max for {}".format(element) + \
                                 " cannot exceed {}".format(len(Uion)))

        # Calculate the ADK prefactors (See Chen, JCP 236 (2013), equation (2))
        # - Scalars
        alpha = physical_constants['fine-structure constant'][0]
        r_e = physical_constants['classical electron radius'][0]
        wa = alpha**3 * c / r_e
        Ea = m_e*c**2/e * alpha**4/r_e
        # - Arrays (one element per ionization level)
        UH = get_ionization_energies('H')[0]
        Z = np.arange( len(Uion) ) + 1
        n_eff = Z * np.sqrt( UH/Uion )
        l_eff = n_eff[0] - 1
        C2 = 2**(2*n_eff) / (n_eff * gamma(n_eff+l_eff+1) * gamma(n_eff-l_eff))
        # For now, we assume l=0, m=0
        self.adk_power = - (2*n_eff - 1)
        self.adk_prefactor = dt * wa * C2 * ( Uion/(2*UH) ) \
            * ( 2*(Uion/UH)**(3./2)*Ea )**(2*n_eff - 1)
        self.adk_exp_prefactor = -2./3 * ( Uion/UH )**(3./2) * Ea

    @catch_gpu_memory_error
    def handle_ionization( self, ion ):
        """
        Handle ionization, either on CPU or GPU

        - For each ion macroparticle, decide whether it is going to
          be further ionized during this timestep, based on the ADK rate.
        - Add the electrons created from ionization to the `target_species`

        Parameters:
        -----------
        ion: an fbpic.Particles object
            The ionizable species, from which new electrons are created.
        """
        # Skip this function if there are no ions
        if ion.Ntot == 0:
            return
        
        # Process particles in batches (of typically 10, 20 particles)
        N_batch = int( ion.Ntot / self.batch_size ) + 1
        # Short-cuts
        use_cuda = self.use_cuda

        # Set the number of levels that should be distinguished
        if self.store_electrons_per_level:
            n_levels = self.level_max - self.level_start
        else:
            n_levels = 1

        # Create temporary arrays (on CPU or GPU, depending on `use_cuda`)
        ionized_from = allocate_empty( ion.Ntot, use_cuda, dtype=np.int16 )
        n_ionized = allocate_empty( (n_levels, N_batch), use_cuda,
                                    dtype=np.int64 )
        # Draw random numbers
        if self.use_cuda:
            random_draw = cupy.random.rand( ion.Ntot, dtype=cupy.float32 )
        else:
            random_draw = np.random.rand( ion.Ntot )

        # Determine the ions that are ionized, and count them in each batch
        # (one thread per batch on GPU; parallel loop over batches on CPU)
        if use_cuda:
            batch_grid_1d, batch_block_1d = cuda_tpb_bpg_1d( N_batch )
            ionize_ions_cuda[ batch_grid_1d, batch_block_1d ](
                N_batch, self.batch_size, ion.Ntot,
                self.level_start, self.level_max, n_levels,
                n_ionized, ionized_from, self.ionization_level, random_draw,
                self.adk_prefactor, self.adk_power, self.adk_exp_prefactor,
                ion.ux, ion.uy, ion.uz, ion.Ex, ion.Ey, ion.Ez,
                ion.Bx, ion.By, ion.Bz, ion.w, self.w_times_level )
        else:
            ionize_ions_numba(
                N_batch, self.batch_size, ion.Ntot,
                self.level_start, self.level_max, n_levels,
                n_ionized, ionized_from, self.ionization_level, random_draw,
                self.adk_prefactor, self.adk_power, self.adk_exp_prefactor,
                ion.ux, ion.uy, ion.uz, ion.Ex, ion.Ey, ion.Ez,
                ion.Bx, ion.By, ion.Bz, ion.w, self.w_times_level )

        # Count the total number of new electrons 
        cumulative_n_ionized = perform_cumsum_2d( n_ionized, use_cuda )
        # If no new particle was created, skip the rest of this function
        if use_cuda:
            if cupy.all( cumulative_n_ionized[:,-1] == 0 ):
                return
        else:
            if np.all( cumulative_n_ionized[:,-1] == 0 ):
                return

        # Loop over the electron species associated to each level
        # (when store_electrons_per_level is False, there is a single species)
        # Reallocate electron species (on CPU or GPU depending on `use_cuda`),
        # to accomodate the electrons produced by ionization,
        # and copy the old electrons to the new arrays
        assert len(self.target_species) == n_levels
        for i_level, elec in enumerate(self.target_species):
            old_Ntot = elec.Ntot
            # Cast to int transfers the data from the GPU if needed
            new_Ntot = old_Ntot + int( cumulative_n_ionized[i_level,-1] )
            reallocate_and_copy_old( elec, use_cuda, old_Ntot, new_Ntot )
            # Create the new electrons from ionization (one thread per batch)
            if use_cuda:
                copy_ionized_electrons_cuda[ batch_grid_1d, batch_block_1d ](
                    N_batch, self.batch_size, old_Ntot, ion.Ntot,
                    cumulative_n_ionized, ionized_from,
                    i_level, self.store_electrons_per_level,
                    elec.x, elec.y, elec.z, elec.inv_gamma,
                    elec.ux, elec.uy, elec.uz, elec.w,
                    elec.Ex, elec.Ey, elec.Ez, elec.Bx, elec.By, elec.Bz,
                    ion.x, ion.y, ion.z, ion.inv_gamma,
                    ion.ux, ion.uy, ion.uz, ion.w,
                    ion.Ex, ion.Ey, ion.Ez, ion.Bx, ion.By, ion.Bz )
                # Mark the new electrons as unsorted
                elec.sorted = False
            else:
                copy_ionized_electrons_numba(
                    N_batch, self.batch_size, old_Ntot, ion.Ntot,
                    cumulative_n_ionized, ionized_from,
                    i_level, self.store_electrons_per_level,
                    elec.x, elec.y, elec.z, elec.inv_gamma,
                    elec.ux, elec.uy, elec.uz, elec.w,
                    elec.Ex, elec.Ey, elec.Ez, elec.Bx, elec.By, elec.Bz,
                    ion.x, ion.y, ion.z, ion.inv_gamma,
                    ion.ux, ion.uy, ion.uz, ion.w,
                    ion.Ex, ion.Ey, ion.Ez, ion.Bx, ion.By, ion.Bz )

            # If the electrons are tracked, generate new ids
            # (on GPU or GPU depending on `use_cuda`)
            generate_new_ids( elec, old_Ntot, new_Ntot )

            # If spin tracking is enabled, generate also new spins
            # (on GPU or GPU depending on `use_cuda`)
            if ion.spin_tracker is not None:
                if use_cuda:
                    # Generate a set of random spins here
                    #rand_sx = allocate_empty(new_Ntot-old_Ntot, True, np.float64)
                    #rand_sy = allocate_empty(new_Ntot - old_Ntot, True, np.float64)
                    #rand_sz = allocate_empty(new_Ntot - old_Ntot, True, np.float64)
                    rand_sx, rand_sy, rand_sz = ion.spin_tracker.generate_ionized_spins_gpu(new_Ntot-old_Ntot)
                    copy_ionized_electron_spin_cuda[ batch_grid_1d,
                                                     batch_block_1d ](
                        N_batch, self.batch_size, old_Ntot,
                        ion.Ntot, self.store_electrons_per_level,
                        cumulative_n_ionized, i_level, ionized_from,
                        elec.spin_tracker.sx, elec.spin_tracker.sy,
                        elec.spin_tracker.sz, ion.spin_tracker.sx,
                        ion.spin_tracker.sy, ion.spin_tracker.sz,
                        rand_sx, rand_sy, rand_sz)
                else:
                    copy_ionized_electron_spin_numba(
                        N_batch, self.batch_size, old_Ntot, new_Ntot,
                        ion.Ntot, self.store_electrons_per_level,
                        cumulative_n_ionized, i_level, ionized_from,
                        elec.spin_tracker.sx, elec.spin_tracker.sy,
                        elec.spin_tracker.sz, ion.spin_tracker.sx,
                        ion.spin_tracker.sy, ion.spin_tracker.sz)


    def send_to_gpu( self ):
        """
        Copy the ionization data to the GPU.
        """
        if self.use_cuda:
            # Arrays with one element per macroparticles
            self.ionization_level = cupy.asarray( self.ionization_level )
            self.w_times_level = cupy.asarray( self.w_times_level )
            # Small-size arrays with ADK parameters
            # (One element per ionization level)
            self.adk_power = cupy.asarray( self.adk_power )
            self.adk_prefactor = cupy.asarray( self.adk_prefactor )
            self.adk_exp_prefactor = cupy.asarray( self.adk_exp_prefactor )

    def receive_from_gpu( self ):
        """
        Receive the ionization data from the GPU.
        """
        if self.use_cuda:
            # Arrays with one element per macroparticles
            self.ionization_level = self.ionization_level.get()
            self.w_times_level = self.w_times_level.get()
            # Small-size arrays with ADK parameters
            # (One element per ionization level)
            self.adk_power = self.adk_power.get()
            self.adk_prefactor = self.adk_prefactor.get()
            self.adk_exp_prefactor = self.adk_exp_prefactor.get()
