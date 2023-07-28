# Copyright 2023, FBPIC contributors
# Authors: Kris Poder, Michael Quin, Matteo Tamburini
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure and methods associated with particle spin
tracking.
"""
import numpy as np
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
from .numba_methods import push_s_numba, push_s_ioniz_numba


if cuda_installed:
    import cupy
    from fbpic.utils.cuda import cuda_tpb_bpg_1d
    from .cuda_methods import push_s_gpu, push_s_ioniz_gpu


class SpinTracker(object):
    """
    Class that stores and tracks particle spin vector.
    """

    def __init__(self, species, dt, sx_m=0., sy_m=0., sz_m=0.,
                       anom=0., spin_distr='fixed'):
        """
        Initialize the SpinTracker class.

        The length of each particles spin vector is 1, and this
        does not change during the push. During initialization,
        the spin vectors will be randomly generated to have the
        average values along all axis specified by the user.

        Note it is therefore unphysical to specify averages that
        sum to $s_x^2+s_y^2+s_z^2>1$.

        Parameters
        ----------
        species: fbpic.Particles object

        dt: float
            Simulation timestep, in s

        sx_m: float
            The species-averaged average projection onto
            the x-axis

        sy_m: float
            The species-averaged average projection onto
            the y-axis

        sz_m: float
            The species-averaged average projection onto
            the z-axis

        anom: float
            The anomalous magnetic moment of the particle.
        """
        self.sx_m = sx_m
        self.sy_m = sy_m
        self.sz_m = sz_m
        self.anom = anom
        self.dt = dt
        self.spin_distr = spin_distr

        # Store the species we perform spin tracking for
        self.ptcl = species
        self.use_cuda = species.use_cuda

        # Register arrays for previous timestep spins
        self.ux_prev = None
        self.uy_prev = None
        self.uz_prev = None

        # Create the initial spin array
        self.sx, self.sy, self.sz = self.generate_new_spins(self.ptcl.Ntot)

    def store_previous_momenta(self):
        """
        Store the momenta at the previous half timestep, ie at
        t = (n-1/2) dt
        """
        if self.use_cuda:
            self.ux_prev = cupy.array(self.ptcl.ux, order='C')
            self.uy_prev = cupy.array(self.ptcl.uy, order='C')
            self.uz_prev = cupy.array(self.ptcl.uz, order='C')
        else:
            self.ux_prev = np.array(self.ptcl.ux, order='C')
            self.uy_prev = np.array(self.ptcl.uy, order='C')
            self.uz_prev = np.array(self.ptcl.uz, order='C')

    def push_s(self):
        """
        Advance particles' spin vector over one timestep according to the
        Bargmann-Michel-Telegdi equation, using a Boris-pusher like method.
        Reference: Wen, Tamburini & Keitel, PRL 122, 214801 (2019)

        At step n we expect the following quantities:
        Quantity                                            Step
        self.sx, self.sy, self.sz,                          n-1/2
        ux_prev, uy_prev, uz_prev                           n-1/2
        self.Ex, self.Ey, self.Ez,                          n
        self.Bx, self.By, self.Bz,                          n
        self.x, self.y, self.z                              n
        self.ux, self.uy, self.uz                           n+1/2

        NOTE!
        Functions have not been modified to work with differing charge in
        'ionizable' or 'plane' methods.
        """
        # No precession for neutral particle
        if self.ptcl.q == 0:
            return

        # GPU (CUDA) version
        if self.use_cuda:
            # Get the threads per block and the blocks per grid
            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( self.ptcl.Ntot )
            # Call the CUDA Kernel for the spin push
            if self.ptcl.ionizer is not None:
                # Ionizable species can have a charge that depends on the
                # macroparticle, and hence require a different function
                push_s_ioniz_gpu[dim_grid_1d, dim_block_1d](
                    self.sx, self.sy, self.sz,
                    self.ux_prev, self.uy_prev, self.uz_prev,
                    self.ptcl.ux, self.ptcl.uy, self.ptcl.uz,
                    self.ptcl.Ex, self.ptcl.Ey, self.ptcl.Ez,
                    self.ptcl.Bx, self.ptcl.By, self.ptcl.Bz,
                    self.ptcl.m, self.ptcl.Ntot, self.dt, self.anom,
                    self.ptcl.ionizer.ionization_level)
                # elif z_plane is not None:
                # ... no implemented effect on spin precession for x_plane...
            else:
                # Standard pusher
                push_s_gpu[dim_grid_1d, dim_block_1d](
                    self.sx, self.sy, self.sz,
                    self.ux_prev, self.uy_prev, self.uz_prev,
                    self.ptcl.ux, self.ptcl.uy, self.ptcl.uz,
                    self.ptcl.Ex, self.ptcl.Ey, self.ptcl.Ez,
                    self.ptcl.Bx, self.ptcl.By, self.ptcl.Bz,
                    self.ptcl.q, self.ptcl.m, self.ptcl.Ntot,
                    self.dt, self.anom)
        # CPU version
        else:
            if self.ptcl.ionizer is not None:
                # Ionizable species can have a charge that depends on the
                # macroparticle, and hence require a different function
                push_s_ioniz_numba(self.sx, self.sy, self.sz,
                                   self.ux_prev, self.uy_prev, self.uz_prev,
                                   self.ptcl.ux, self.ptcl.uy, self.ptcl.uz,
                                   self.ptcl.Ex, self.ptcl.Ey, self.ptcl.Ez,
                                   self.ptcl.Bx, self.ptcl.By, self.ptcl.Bz,
                                   self.ptcl.m, self.ptcl.Ntot,
                                   self.dt, self.anom,
                                   self.ptcl.ionizer.ionization_level)
                # elif z_plane is not None:
                # ... no implemented effect on spin precession for z_plane...
            else:
                # Standard spin pusher
                push_s_numba(self.sx, self.sy, self.sz,
                             self.ux_prev, self.uy_prev, self.uz_prev,
                             self.ptcl.ux, self.ptcl.uy, self.ptcl.uz,
                             self.ptcl.Ex, self.ptcl.Ey, self.ptcl.Ez,
                             self.ptcl.Bx, self.ptcl.By, self.ptcl.Bz,
                             self.ptcl.q, self.ptcl.m, self.ptcl.Ntot,
                             self.dt, self.anom)

    def generate_new_spins(self, Ntot):
        """
        Create new spin vectors for particles. This method
        generates a set of spin components, where the ensemble
        averages satisfy the user's requirement in terms of averages.
        """
        sx = np.ones(Ntot) * self.sx_m
        sy = np.ones(Ntot) * self.sy_m
        sz = np.ones(Ntot) * self.sz_m
        return sx, sy, sz

    def send_to_gpu(self):
        """
        Copy the spin data to the GPU.
        """
        self.sx = cupy.asarray(self.sx)
        self.sy = cupy.asarray(self.sy)
        self.sz = cupy.asarray(self.sz)
        # TODO: And the anomalous moment too???? or ux_prev?

    def receive_from_gpu(self):
        """
        Transfer the spin data from the GPU to the CPU
        """
        self.sx = self.sx.get()
        self.sy = self.sy.get()
        self.sz = self.sz.get()
        # TODO: And the anomalous moment too???? or ux_prev?
