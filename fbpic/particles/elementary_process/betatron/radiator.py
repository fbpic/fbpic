# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
"""

import numpy as np
from scipy.constants import c, e, m_e, physical_constants
from scipy.special import kv
from scipy.integrate import quad
from scipy.interpolate import interp1d

#from .numba_methods import ionize_ions_numba, copy_ionized_electrons_numba

# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
from fbpic.utils.printing import catch_gpu_memory_error
if cuda_installed:
    import cupy
    from fbpic.utils.cuda import cuda_tpb_bpg_1d
    #from .cuda_methods import ionize_ions_cuda, copy_ionized_electrons_cuda

class Radiator(object):
    """
    Class that contains the data associated with betatron radiation.
    """
    def __init__(self, radiating_species, omega_axis,
                 theta_axis, phi_axis, gamma_cutoff=10.0):
        """
        Initialize an Ionizer instance

        Parameters
        ----------
        radiating_species: an fbpic.Particles object
            This object is not modified or registered.

        omega_axis: tuple
            Parameters for the frequency axis provided as
            (omega_min, omega_max, N_omega), where omega_min and
            omega_max are floats in (1/s) and N_omega in integer

        theta_axis: tuple
            Parameters for the elevation angle axis provided as
            (theta_min, theta_max, N_theta), where theta_min and
            theta_max are floats in (rad) and N_theta in integer

        phi_axis: tuple
            Parameters for the azimuthal angle axis provided as
            (phi_min, phi_max, N_phi), where phi_min and
            phi_max are floats in (rad) and N_phi in integer

        gamma_cutoff: float
            Minimal gamma factor of particles for which radiation
            is calculated
        """
        # Register a few parameters
        self.beta_omega_min, self.beta_omega_max, self.beta_N_omega = omega_axis
        self.beta_theta_min, self.beta_theta_max, self.beta_N_theta = omega_theta
        self.beta_phi_min, self.beta_phi_max, self.beta_N_phi = omega_phi
        self.gamma_cutoff = gamma_cutoff
        self.Larmore_factor = 2 * e_cgs**2 / 3 / с_cgs * 1e-7 * radiating_species.dt

        self.use_cuda = radiating_species.use_cuda
        # Process radiating particles into batches
        self.batch_size = 10

        # Initialize radiation-relevant meta-data
        self.initialize_S_function()
        self.beta_omega = np.linspace(
            self.beta_omega_min, self.beta_omega_max, self.beta_N_omega
        )
        self.beta_theta = np.linspace(
            self.beta_omega_theta, self.beta_omega_theta, self.beta_N_theta
        )
        self.beta_frequency = np.linspace(
            self.beta_phi_min, self.beta_phi_max, self.beta_N_phi
        )

        self.radiation_data = np.zeros(
            (self.beta_N_phi, self.beta_N_theta, self.beta_N_omega),
            dtype=np.double
        )

    def initialize_S_function( self, x_max=32, nSamples=65536 ):
        """
        Initialize spectral shape function
        """

        k_53 = lambda x : kv(5./3, x)
        S0 = lambda x : 9 * 3**0.5 / 8 / np.pi * x \
                        * quad(k_53, x, np.inf)[0]
        S0 = np.vectorize(S0)
        x_ax = np.r_[0 : x_max : nSamples*1j]
        dx_ax = x_ax[1] - x_ax[0]

        self.S_func_data = S0(x_ax)
        self.S_func_dx = x_ax[1] - x_ax[0]
        self.S_func_x_max = x_max

    @catch_gpu_memory_error
    def handle_radiation( self, eon ):
        """
        Handle radiation, either on CPU or GPU

        Parameters:
        -----------
        eon: an fbpic.Particles object
            The radiating species
        """
        # Skip this function if there are no ions
        if eon.Ntot == 0:
            return

        # Process particles in batches (of typically 10, 20 particles)
        N_batch = int( ion.Ntot / self.batch_size ) + 1
        # Short-cuts
        use_cuda = self.use_cuda

        # Determine the ions that are ionized, and count them in each batch
        # (one thread per batch on GPU; parallel loop over batches on CPU)
        if use_cuda:
            batch_grid_1d, batch_block_1d = cuda_tpb_bpg_1d( N_batch )
            gather_betatron_cuda[ batch_grid_1d, batch_block_1d ](
                N_batch, self.batch_size, eon.Ntot,
                eon.ux, eon.uy, eon.uz, eon.Ex, eon.Ey, eon.Ez,
                eon.Bx, eon.By, eon.Bz, eon.w, self.Larmore_factor,
                self.gamma_cutoff, self.radiation_data )
        else:
            gather_betatron_numba(
                N_batch, self.batch_size, eon.Ntot,
                eon.ux, eon.uy, eon.uz, eon.Ex, eon.Ey, eon.Ez,
                eon.Bx, eon.By, eon.Bz, eon.w, self.radiation_data )

    def send_to_gpu( self ):
        """
        Copy the ionization data to the GPU.
        """
        if self.use_cuda:
            # Arrays with one element per macroparticles
            self.radiation_data = cupy.asarray( self.radiation_data)
            self.S_func_data = cupy.asarray( self.S_func_data)

    def receive_from_gpu( self ):
        """
        Receive the ionization data from the GPU.
        """
        if self.use_cuda:
            self.radiation_data = self.radiation_data.get()

