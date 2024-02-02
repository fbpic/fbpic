# Copyright 2023, FBPIC contributors
# Authors: Igor A Andriyash, Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
"""

import numpy as np
from scipy.constants import m_e, c, e, epsilon_0, hbar
from scipy.special import kv
from scipy.integrate import quad
from numba.core.errors import NumbaPerformanceWarning
from scipy.integrate import IntegrationWarning
import warnings

from ..cuda_numba_utils import allocate_empty
from .numba_methods import gather_synchrotron_numba

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=IntegrationWarning)

# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
from fbpic.utils.printing import catch_gpu_memory_error
if cuda_installed:
    import cupy
    from fbpic.utils.cuda import cuda_tpb_bpg_1d
    from .cuda_methods import gather_synchrotron_cuda
    from numba.cuda.random import create_xoroshiro128p_states

class SynchrotronRadiator(object):
    """
    Class for the synchrotron radiation calculation.
    """
    def __init__(self, radiating_species, photon_energy_axis,
                 theta_x_axis, theta_y_axis, gamma_cutoff,
                 radiation_reaction, x_max, nSamples):
        """
        Initialize a Radiator instance

        Parameters
        ----------
        radiating_species: an fbpic.Particles object
            This object is not modified

        photon_energy_axis: tuple
            Parameters for the photon energy axis provided as
            `(photon_energy_min, photon_energy_max, N_photon_energy)`, where
            `photon_energy_min` and `photon_energy_max` are floats in Joules
            and `N_photon_energy` is integer

        theta_x_axis: tuple
            Parameters for the x-elevation angle axis provided as
            `(theta_x_min, theta_x_max, N_theta_x)`, where `theta_x_min`
            and `theta_x_max` are floats in (rad) and `N_theta_x` is integer

        theta_y_axis: tuple
            Parameters for the y-elevation angle axis provided as
            `(theta_y_min, theta_y_max, N_theta_y)`, where `theta_y_min`
            and `theta_y_max` are floats in radians and `N_theta_y` is integer

        gamma_cutoff: float
            Minimal particle gamma factor for which radiation is calculated

        radiation_reaction: bool
            Whether to consider radiation reaction on the electrons

        x_max: float
            Extent of the sampling used for the spectral profile function

        nSamples: integer
            number of sampling points for the spectral profile function
        """
        # Register a few parameters
        self.use_cuda = radiating_species.use_cuda
        self.eon = radiating_species
        self.dt = radiating_species.dt
        self.gamma_cutoff_inv = 1. / gamma_cutoff
        self.radiation_reaction = radiation_reaction

        self.omega_min = photon_energy_axis[0] / hbar
        self.omega_max = photon_energy_axis[1] / hbar
        self.N_omega = photon_energy_axis[2]

        self.theta_x_min = theta_x_axis[0]
        self.theta_x_max = theta_x_axis[1]
        self.N_theta_x = theta_x_axis[2]

        self.theta_y_min = theta_y_axis[0]
        self.theta_y_max = theta_y_axis[1]
        self.N_theta_y = theta_y_axis[2]

        # Create the photon frequency axis
        self.omega_ax = np.linspace(
            self.omega_min, self.omega_max, self.N_omega
        )
        self.d_omega = self.omega_ax[1] - self.omega_ax[0]

        # Create the angular axes
        self.d_theta_x = (self.theta_x_max - self.theta_x_min) \
            / (self.N_theta_x - 1)
        self.d_theta_y = (self.theta_y_max - self.theta_y_min) \
            / (self.N_theta_y - 1)

        self.Larmore_factor_density = e**2 * self.dt \
            / ( 6 * np.pi * epsilon_0 * c * hbar * \
                self.d_theta_x * self.d_theta_y )

        self.Larmore_factor_momentum = e**2 * self.dt \
            / ( 6 * np.pi * epsilon_0 * m_e * c**3 )

        # Calculate sampling of the spectral profile function
        self.initialize_S_function( x_max=x_max, nSamples=nSamples )

        # Initialize radiation data
        self.radiation_data = np.zeros(
            (self.N_theta_x, self.N_theta_y, self.N_omega),
            dtype=np.double
        )

        # send the radiation-relevant data to GPU
        self.send_to_gpu()

        # Process radiating particles into batches
        self.batch_size = 10

    def initialize_S_function( self, x_max, nSamples ):
        """
        Initialize spectral profile function

        Parameters
        ----------
        x_max: float
            Extent of the sampling used for the spectral profile function

        nSamples: integer
            number of sampling points for the spectral profile function
        """

        k_53 = lambda x : kv(5./3, x)
        S0 = lambda x : 9 * 3**0.5 / 8 / np.pi * x \
                        * quad(k_53, x, np.inf)[0]
        S0 =  np.vectorize(S0)
        x_ax = np.linspace(0, x_max, nSamples)
        self.S_func_data = S0(x_ax)
        self.S_func_dx = x_ax[1] - x_ax[0]

    @catch_gpu_memory_error
    def handle_radiation( self ):
        """
        Handle radiation, either on CPU or GPU
        """
        # Short-cuts
        eon = self.eon

        # Skip this function if there are no electrons
        if eon.Ntot == 0:
            return

        if self.use_cuda:
            # Process particles in batches (of typically 10, 20 particles)
            N_batch = int( eon.Ntot / self.batch_size ) + 1

            # Allocate a container for spectral profiles for the
            # particles in the batch
            spect_batch = allocate_empty( (N_batch, self.N_omega), self.use_cuda,
                                          dtype=np.double )

            # initialize states for random number generator
            seed = np.random.randint( 256 )
            rng_states_batch = create_xoroshiro128p_states(N_batch, seed)

            # run kernel for radiation calculation
            batch_grid_1d, batch_block_1d = cuda_tpb_bpg_1d( N_batch )
            gather_synchrotron_cuda[ batch_grid_1d, batch_block_1d ](
                N_batch, self.batch_size,  eon.Ntot,
                eon.ux, eon.uy, eon.uz, eon.Ex, eon.Ey, eon.Ez,
                eon.Bx, eon.By, eon.Bz, eon.w, eon.inv_gamma,
                self.Larmore_factor_density,
                self.Larmore_factor_momentum,
                self.gamma_cutoff_inv, self.radiation_reaction,
                self.omega_ax, self.S_func_dx, self.S_func_data,
                self.theta_x_min, self.theta_x_max, self.d_theta_x,
                self.theta_y_min, self.theta_y_max, self.d_theta_y,
                spect_batch, rng_states_batch, self.radiation_data)
        else:
            # Allocate array for the single particle spectral profile
            spect_loc = allocate_empty( (self.N_omega,), self.use_cuda,
                                        dtype=np.double )

            # radiation calculation (parallel loop over particle)
            gather_synchrotron_numba(
                eon.Ntot,
                eon.ux, eon.uy, eon.uz, eon.Ex, eon.Ey, eon.Ez,
                eon.Bx, eon.By, eon.Bz, eon.w, eon.inv_gamma,
                self.Larmore_factor_density,
                self.Larmore_factor_momentum,
                self.gamma_cutoff_inv, self.radiation_reaction,
                self.omega_ax, self.S_func_dx, self.S_func_data,
                self.theta_x_min, self.theta_x_max, self.d_theta_x,
                self.theta_y_min, self.theta_y_max, self.d_theta_y,
                spect_loc, self.radiation_data)

    def send_to_gpu( self ):
        """
        Copy relevant data to the GPU
        """
        if self.use_cuda:
            self.radiation_data = cupy.asarray( self.radiation_data )
            self.omega_ax = cupy.asarray( self.omega_ax )
            self.S_func_data = cupy.asarray( self.S_func_data )

    def receive_from_gpu( self ):
        """
        Receive relevant data from the GPU
        """
        if self.use_cuda:
            self.radiation_data = self.radiation_data.get()
            self.omega_ax = self.omega_ax.get()
            self.S_func_data = self.S_func_data.get()
