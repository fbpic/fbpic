# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
# TODO: Describe file
"""
import numpy as np
from numba import cuda
from scipy.constants import c, h
from .numba_methods import get_photon_density_gaussian_numba, \
    determine_scatterings_numba, scatter_photons_electrons_numba
from ..cuda_numba_utils import allocate_empty, reallocate_and_copy_old, \
                                perform_cumsum, generate_new_ids
# Check if CUDA is available, then import CUDA functions
from fbpic.cuda_utils import cuda_installed
if cuda_installed:
    from fbpic.cuda_utils import cuda_tpb_bpg_1d
    from numba.cuda.random import create_xoroshiro128p_states
    from .cuda_methods import get_photon_density_gaussian_cuda, \
        determine_scatterings_cuda, scatter_photons_electrons_cuda

class ComptonScatterer(object):
    """
    Simulate Compton scattering for a Gaussian laser

    # Insist: for now, this assumes that the laser propagates along
    # the z axis

    # TODO: Describe method for ComptonScattering
    # Reference Dave Grote
    """
    def __init__( self, source_species, target_species, laser_energy,
        laser_wavelength, laser_waist, laser_ctau, laser_initial_z0,
        ratio_w_electron_photon, boost ):
        """
        Initialize a ComptonScatterer:
        Scattering on a uniform, monoenergetic, unidirectional flux of photons.

        # TODO : describe parameters

        Parameters
        ----------
        source_species: an fbpic Particles object
            The species that stores the electrons

        target species: an fbpic Particles object
            The species that will store the produced photons

        """
        # Register the photons species
        assert target_species.q == 0
        self.target_species = target_species
        self.ratio_w_electron_photon = ratio_w_electron_photon
        self.inv_ratio_w_elec_photon = 1./ratio_w_electron_photon

        # Register parameters of the simulation boosted-frame
        if boost is not None:
            self.gamma_boost = boost.gamma0
            self.beta_boost = boost.beta0
        else:
            self.gamma_boost = 1.
            self.beta_boost = 0.

        # Register the parameters of the incident photon flux
        # **in the frame of the simulation**
        # For now: this assumes that the photon flux is along z.
        photon_lab_px = 0.
        photon_lab_py = 0.
        photon_lab_pz = - h/laser_wavelength
        photon_lab_p = np.sqrt(
            photon_lab_px**2 + photon_lab_py**2 + photon_lab_pz**2 )
        # Register photon parameters in the simulation frame
        self.photon_px = photon_lab_px
        self.photon_py = photon_lab_py
        self.photon_pz = self.gamma_boost * \
            (photon_lab_pz - self.beta_boost*photon_lab_p)
        self.photon_p = np.sqrt(
            self.photon_px**2 + self.photon_py**2 + self.photon_pz**2 )
        self.photon_beta_x = self.photon_px/self.photon_p
        self.photon_beta_y = self.photon_py/self.photon_p
        self.photon_beta_z = self.photon_pz/self.photon_p

        # Register laser parameters
        self.laser_initial_z0 = laser_initial_z0
        # Precalculate parameters for more efficient calculations
        self.inv_laser_waist2 = 1./laser_waist**2
        self.inv_laser_ctau2 = 1./laser_ctau**2
        # Peak of the photon density, in the lab frame
        effective_volume = (np.pi/2.)**(3./2) * laser_waist**2 * laser_ctau
        photon_energy = photon_lab_p*c
        self.photon_n_lab_peak = laser_energy /(effective_volume*photon_energy)

        # Register a few other parameters
        self.batch_size = 10
        self.use_cuda = source_species.use_cuda

    def handle_scattering( self, elec, t ):
        """
        Handle Compton scattering, either on CPU or GPU

        - For each electron, decide whether it is going to be produce a new
          photon, based on the integrated Klein-Nishina formula
        - Add the photons created from Compton scattering to `target_species`

        Parameters:
        -----------
        elec: an fbpic.Particles object
            The electrons species, from which new photons will be created

        t: float
            The simulation time
        """
        # Process particles in batches (of typically 10, 20 particles)
        N_batch = int( elec.Ntot / self.batch_size ) + 1
        # Short-cut for use_cuda
        use_cuda = self.use_cuda

        # Create temporary arrays (on CPU or GPU, depending on `use_cuda`)
        nscatter_per_batch = allocate_empty(N_batch, use_cuda, dtype=np.int64)
        nscatter_per_elec = allocate_empty(elec.Ntot, use_cuda, dtype=np.int64)
        photon_n = allocate_empty(elec.Ntot, use_cuda, dtype=np.float64)
        # Prepare random numbers
        if self.use_cuda:
            seed = np.random.randint( 256 )
            random_states = create_xoroshiro128p_states( N_batch, seed )


        # For each electron, calculate the local density of photons
        # *in the frame of the simulation*
        if use_cuda:
            bpg, tpg = cuda_tpb_bpg_1d( elec.Ntot )
            get_photon_density_gaussian_cuda[ bpg, tpg ]( photon_n, elec.Ntot,
                elec.x, elec.y, elec.z, c*t, self.photon_n_lab_peak,
                self.inv_laser_waist2, self.inv_laser_ctau2,
                self.laser_initial_z0, self.gamma_boost, self.beta_boost  )
        else:
            get_photon_density_gaussian_numba( photon_n, elec.Ntot,
                elec.x, elec.y, elec.z, c*t, self.photon_n_lab_peak,
                self.inv_laser_waist2, self.inv_laser_ctau2,
                self.laser_initial_z0, self.gamma_boost, self.beta_boost  )

        # Determine the electrons that scatter, and count them in each batch
        # (one thread per batch on GPU; parallel loop over batches on CPU)
        if use_cuda:
            batch_grid_1d, batch_block_1d = cuda_tpb_bpg_1d( N_batch )
            determine_scatterings_cuda[ batch_grid_1d, batch_block_1d ](
                N_batch, self.batch_size, elec.Ntot,
                nscatter_per_elec, nscatter_per_batch, random_states,
                elec.dt, elec.ux, elec.uy, elec.uz, elec.inv_gamma,
                self.ratio_w_electron_photon, photon_n, self.photon_p,
                self.photon_beta_x, self.photon_beta_y, self.photon_beta_z )
        else:
            determine_scatterings_numba(
                N_batch, self.batch_size, elec.Ntot,
                nscatter_per_elec, nscatter_per_batch,
                elec.dt, elec.ux, elec.uy, elec.uz, elec.inv_gamma,
                self.ratio_w_electron_photon, photon_n, self.photon_p,
                self.photon_beta_x, self.photon_beta_y, self.photon_beta_z )

        # Count the total number of new photons (operation always performed
        # on the CPU, as this is typically difficult on the GPU)
        if use_cuda:
            nscatter_per_batch = nscatter_per_batch.copy_to_host()
        cumul_nscatter_per_batch = perform_cumsum( nscatter_per_batch )
        # If no new particle was created, skip the rest of this function
        if cumul_nscatter_per_batch[-1] == 0:
            return

        # Reallocate photons species (on CPU or GPU depending on `use_cuda`),
        # to accomodate the photons produced by Compton scattering,
        # and copy the old photons to the new arrays
        photons = self.target_species
        old_Ntot = photons.Ntot
        new_Ntot = old_Ntot + cumul_nscatter_per_batch[-1]
        reallocate_and_copy_old( photons, use_cuda, old_Ntot, new_Ntot )

        # Create the new photons from ionization (with a random
        # scattering angle) and add recoil momentum to the electrons
        if use_cuda:
            cumul_nscatter_per_batch = cuda.to_device(cumul_nscatter_per_batch)
            scatter_photons_electrons_cuda[ batch_grid_1d, batch_block_1d ](
                N_batch, self.batch_size, old_Ntot, elec.Ntot,
                cumul_nscatter_per_batch, nscatter_per_elec, random_states,
                self.photon_p, self.photon_px, self.photon_py, self.photon_pz,
                photons.x, photons.y, photons.z, photons.inv_gamma,
                photons.ux, photons.uy, photons.uz, photons.w,
                elec.x, elec.y, elec.z, elec.inv_gamma, elec.ux, elec.uy,
                elec.uz, elec.w, self.inv_ratio_w_elec_photon )
            photons.sorted = False
        else:
            scatter_photons_electrons_numba(
                N_batch, self.batch_size, old_Ntot, elec.Ntot,
                cumul_nscatter_per_batch, nscatter_per_elec,
                self.photon_p, self.photon_px, self.photon_py, self.photon_pz,
                photons.x, photons.y, photons.z, photons.inv_gamma,
                photons.ux, photons.uy, photons.uz, photons.w,
                elec.x, elec.y, elec.z, elec.inv_gamma, elec.ux, elec.uy,
                elec.uz, elec.w, self.inv_ratio_w_elec_photon )

        # If the photons are tracked, generate new ids
        # (on GPU or GPU depending on `use_cuda`)
        generate_new_ids( photons, old_Ntot, new_Ntot )
