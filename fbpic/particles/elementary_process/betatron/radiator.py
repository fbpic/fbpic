# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
"""

import numpy as np
from scipy.constants import c, e, m_e, physical_constants, epsilon_0
from scipy.special import kv
from scipy.integrate import quad
from scipy.interpolate import interp1d

# e_cgs = 4.8032047e-10
# с_cgs = c * 1e2

#from .numba_methods import ionize_ions_numba, copy_ionized_electrons_numba

from ..cuda_numba_utils import allocate_empty

from .numba_methods import gather_betatron_numba

# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
from fbpic.utils.printing import catch_gpu_memory_error
if cuda_installed:
    import cupy
    from fbpic.utils.cuda import cuda_tpb_bpg_1d
    from .cuda_methods import gather_betatron_cuda

class Radiator(object):
    """
    Class that contains the data associated with betatron radiation.
    """
    def __init__(self, radiating_species, omega_axis,
                 theta_x_axis, theta_y_axis, gamma_cutoff,
                 x_max, nSamples):
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

        theta_x_axis: tuple
            Parameters for the x-elevation angle axis provided as
            (theta_x_min, theta_x_max, N_x_theta), where theta_x_min
            and theta_x_max are floats in (rad) and N_x_theta is an integer

        theta_y_axis: tuple
            Parameters for the y-elevation angle axis provided as
            (theta_y_min, theta_y_max, N_y_theta), where theta_y_min
            and theta_y_max are floats in (rad) and N_y_theta is an integer

        gamma_cutoff: float
            Minimal gamma factor of particles for which radiation
            is calculated
        """
        # Register a few parameters
        self.eon = radiating_species
        omega_min, omega_max, self.N_omega = omega_axis
        self.theta_x_min, self.theta_x_max, N_x_theta = theta_x_axis
        self.theta_y_min, self.theta_y_max, N_y_theta = theta_y_axis
        self.gamma_cutoff = gamma_cutoff
        self.Larmore_factor = e**2 / 6 / np.pi / epsilon_0 / c * self.eon.dt
        #2 * e_cgs**2 / 3 / с_cgs * 1e-7 * self.eon.dt

        self.use_cuda = self.eon.use_cuda
        # Process radiating particles into batches
        self.batch_size = 10

        # Initialize radiation-relevant meta-data
        self.initialize_S_function( x_max=x_max, nSamples=nSamples )
        self.omega_ax = np.linspace(omega_min, omega_max, self.N_omega)

        self.d_theta_x = (self.theta_x_max - self.theta_x_min) / N_x_theta
        self.d_theta_y = (self.theta_y_max - self.theta_y_min) / N_y_theta

        self.radiation_data = np.zeros(
            (N_x_theta, N_y_theta, self.N_omega), dtype=np.double
        )

        self.send_to_gpu()

    def initialize_S_function( self, x_max, nSamples ):
        """
        Initialize spectral shape function
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

        # Process particles in batches (of typically 10, 20 particles)
        N_batch = int( eon.Ntot / self.batch_size ) + 1

        spect_loc = allocate_empty( (N_batch, self.N_omega), self.use_cuda,
                                    dtype=np.double )

        # Determine the ions that are ionized, and count them in each batch
        # (one thread per batch on GPU; parallel loop over batches on CPU)
        if self.use_cuda:
            batch_grid_1d, batch_block_1d = cuda_tpb_bpg_1d( N_batch )
            gather_betatron_cuda[ batch_grid_1d, batch_block_1d ](
                N_batch, self.batch_size,  eon.Ntot,
                eon.ux, eon.uy, eon.uz, eon.Ex, eon.Ey, eon.Ez,
                eon.Bx, eon.By, eon.Bz, eon.w,
                self.Larmore_factor, self.gamma_cutoff,
                self.omega_ax, self.S_func_dx, self.S_func_data,
                self.theta_x_min, self.theta_x_max, self.d_theta_x,
                self.theta_y_min, self.theta_y_max, self.d_theta_y,
                spect_loc, self.radiation_data)
        else:
            gather_betatron_numba(
                eon.Ntot,
                eon.ux, eon.uy, eon.uz, eon.Ex, eon.Ey, eon.Ez,
                eon.Bx, eon.By, eon.Bz, eon.w,
                self.Larmore_factor, self.gamma_cutoff,
                self.omega_ax, self.S_func_dx, self.S_func_data,
                self.theta_x_min, self.theta_x_max, self.d_theta_x,
                self.theta_y_min, self.theta_y_max, self.d_theta_y,
                spect_loc, self.radiation_data)

    def send_to_gpu( self ):
        """
        Copy the ionization data to the GPU.
        """
        if self.use_cuda:
            # Arrays with one element per macroparticles
            self.radiation_data = cupy.asarray( self.radiation_data )
            self.omega_ax = cupy.asarray(self.S_func_data )
            self.S_func_data = cupy.asarray( self.S_func_data )

    def receive_from_gpu( self ):
        """
        Receive the ionization data from the GPU.
        """
        if self.use_cuda:
            self.radiation_data = self.radiation_data.get()

