# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
# TODO: Describe file
"""
import numpy as np
from numba import cuda
from .numba_methods import determine_scatterings_numba, \
                            scatter_photons_electrons_numba
from ..cuda_numba_utils import allocate_empty, reallocate_and_copy_old, \
                                perform_cumsum, generate_new_ids
# Check if CUDA is available, then import CUDA functions
from fbpic.cuda_utils import cuda_installed
if cuda_installed:
    from accelerate.cuda.rand import PRNG
    from fbpic.cuda_utils import cuda_tpb_bpg_1d
    from .cuda_methods import determine_scatterings_cuda, \
                            scatter_photons_electrons_cuda

class ComtonScatterer(object):
    """
    # TODO: Describe method for ComptonScattering
    # Reference Dave Grote
    """
    def __init__(self, source_species, target_species, photon_lab_n,
            photon_lab_px, photon_lab_py, photon_lab_pz, boost=None):
        """
        Initialize a ComptonScatterer:
        Scattering on a uniform, monoenergetic, unidirectional flux of photons.

        # TODO : describe main attributes

        Parameters
        ----------
        source_species: an fbpic Particles object
            The species that stores the electrons

        target species: an fbpic Particles object
            The species that will store the produced photons

        photon_lab_n: float (m^-3)
            The density of photons *in the lab frame*

        photon_lab_px, photon_lab_py, photon_lab_pz: float (kg.m.s^-1)
            The momenta of the photons *in the lab frame*

        boost: an fbpic BoostConverter object
            Contains information on the boosted-Lorentz frame,
            when running in the boosted frame.
        """
        # Register the photons species
        assert target_species.q == 0
        self.target_species = target_species

        # Register the parameters of the incident photon flux
        # **in the frame of the simulation**
        if boost is None:
            # Lab frame simulation: register directly the parameters
            self.photon_n = photon_lab_n
            self.photon_px = photon_lab_px
            self.photon_py = photon_lab_py
            self.photon_pz = photon_lab_pz
        else:
            # Boosted-frame simulation: convert photon parameters
            # to the boosted-frame, and store them as such
            photon_lab_p = np.sqrt( photon_lab_px**2 \
                                + photon_lab_py**2 + photon_lab_pz**2)
            # All the quantities below are in the frame of the simulation
            self.photon_px = photon_lab_px
            self.photon_py = photon_lab_py
            self.photon_pz = boost.gamma0 * \
                    (photon_lab_pz - boost.beta0*photon_lab_p)
            self.photon_n = photon_lab_n * \
                    boost.gamma0 * (1 - boost.beta0*photon_pz/photon_p)
        # Get additional useful parameters (precalculated, because constant)
        self.photon_p = np.sqrt( self.photon_px**2 \
                                + self.photon_py**2 + self.photon_pz**2 )
        self.photon_beta_x = self.photon_px/self.photon_p
        self.photon_beta_y = self.photon_py/self.photon_p
        self.photon_beta_z = self.photon_pz/self.photon_p

        # Register a few other parameters
        self.batch_size = 10
        self.use_cuda = source_species.use_cuda
        # Prepare random number generator
        if self.use_cuda:
            self.prng = PRNG()

    def handle_scattering( self, elec ):
        """
        Handle Compton scattering, either on CPU or GPU

        - For each electron, decide whether it is going to be produce a new
          photon, based on the integrated Klein-Nishina formula
        - Add the photons created from Compton scattering to `target_species`

        Parameters:
        -----------
        elec: an fbpic.Particles object
            The electrons species, from which new photons will be created
        """
        # Process particles in batches (of typically 10, 20 particles)
        N_batch = int( elec.Ntot / self.batch_size ) + 1
        # Short-cut for use_cuda
        use_cuda = self.use_cuda

        # Create temporary arrays (on CPU or GPU, depending on `use_cuda`)
        does_scatter = allocate_empty( elec.Ntot, use_cuda, dtype=np.int16 )
        n_scatters = allocate_empty( N_batch, use_cuda, dtype=np.int64 )
        # Draw random numbers
        if self.use_cuda:
            random_draw = allocate_empty(elec.Ntot, use_cuda, dtype=np.float32)
            self.prng.uniform( random_draw )
        else:
            random_draw = np.random.rand( elec.Ntot )

        # Determine the electrons that scatter, and count them in each batch
        # (one thread per batch on GPU; parallel loop over batches on CPU)
        if use_cuda:
            batch_grid_1d, batch_block_1d = cuda_tpb_bpg_1d( N_batch )
            determine_scatterings_cuda[ batch_grid_1d, batch_block_1d ](
                N_batch, self.batch_size, elec.Ntot,
                does_scatter, n_scatters, random_draw,
                elec.ux, elec.uy, elec.uz, elec.inv_gamma,
                self.photon_n, self.photon_p, self.photon_beta_x, self.photon_py, self.photon_pz )
        else:
            determine_scatterings_numba(
                N_batch, self.batch_size, elec.Ntot,
                does_scatter, n_scatters, random_draw,
                elec.ux, elec.uy, elec.uz, elec.inv_gamma,
                self.photon_n, self.photon_px, self.photon_py, self.photon_pz )

        # Count the total number of new photons (operation always performed
        # on the CPU, as this is typically difficult on the GPU)
        if use_cuda:
            n_scatters = n_scatters.copy_to_host()
        cumulative_n_scatters = perform_cumsum( n_scatters )
        # If no new particle was created, skip the rest of this function
        if cumulative_n_scatters[-1] == 0:
            return

        # Reallocate photons species (on CPU or GPU depending on `use_cuda`),
        # to accomodate the photons produced by Compton scattering,
        # and copy the old photons to the new arrays
        photons = self.target_species
        old_Ntot = photons.Ntot
        new_Ntot = old_Ntot + cumulative_n_scatters[-1]
        reallocate_and_copy_old( photons, use_cuda, old_Ntot, new_Ntot )

        # Create the new photons from ionization (with a random angle)
        # and add recoil momentum to the electrons
        if use_cuda:
            cumulative_n_scatters = cuda.to_device( cumulative_n_scatters )
            scatter_photons_electrons_cuda[ batch_grid_1d, batch_block_1d ](
                N_batch, self.batch_size, old_Ntot, elec.Ntot,
                cumulative_n_scatters, does_scatter,
                photons.x, photons.y, photons.z, photons.inv_gamma,
                photons.ux, photons.uy, photons.uz, photons.w,
                photons.Ex, photons.Ey, photons.Ez,
                photons.Bx, photons.By, photons.Bz,
                elec.x, elec.y, elec.z, elec.inv_gamma,
                elec.ux, elec.uy, elec.uz, elec.w,
                elec.Ex, elec.Ey, elec.Ez, elec.Bx, elec.By, elec.Bz )
            photons.sorted = False
        else:
            scatter_photons_electrons_numba(
                N_batch, self.batch_size, old_Ntot, elec.Ntot,
                cumulative_n_scatters, does_scatter,
                photons.x, photons.y, photons.z, photons.inv_gamma,
                photons.ux, photons.uy, photons.uz, photons.w,
                photons.Ex, photons.Ey, photons.Ez,
                photons.Bx, photons.By, photons.Bz,
                elec.x, elec.y, elec.z, elec.inv_gamma,
                elec.ux, elec.uy, elec.uz, elec.w,
                elec.Ex, elec.Ey, elec.Ez, elec.Bx, elec.By, elec.Bz )

        # If the photons are tracked, generate new ids
        # (on GPU or GPU depending on `use_cuda`)
        generate_new_ids( photons, old_Ntot, new_Ntot )
