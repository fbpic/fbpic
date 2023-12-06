# Copyright 2023, FBPIC contributors
# Author: Michael J. Quin, Kristjan Poder
# Scientific supervision: Matteo Tamburini
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
from .cuda_numba_utils import random_point_sphere_gpu, \
    random_point_sphere_cpu

if cuda_installed:
    import cupy
    from fbpic.utils.cuda import cuda_tpb_bpg_1d
    from .cuda_methods import push_s_gpu, push_s_ioniz_gpu


class SpinTracker(object):
    """
    Class that stores and tracks particle spin vector.
    """

    def __init__(self, species, dt, sx_m=0., sy_m=0., sz_m=1.,
                 anom=0., spin_distr='fixed'):
        """
        Initialize the SpinTracker class.

        The length of each particle's spin vector is 1, and this
        does not change during the push. During initialization,
        the spin vectors will be all set to the same value or
        randomly generated to have the average values along all
        axis specified by the user (see spin_distr below).

        .. math::
            \\frac{d\\boldsymbol{s}}{dt} = (\\boldsymbol{\\Omega}_T +
             \\boldsymbol{\\Omega}_a) \\boldsymbol{s}

        where

        .. math::
            \\boldsymbol{\\Omega}_T = \\frac{q}{m}\\left(
                 \\frac{\\boldsymbol{B}}{\\gamma} -
                 \\frac{\\boldsymbol{B}}{1+\\gamma}
                 \\times \\frac{\\boldsymbol{E}}{c} \\right)

        and

        .. math::
            \\boldsymbol{\\Omega}_a = a_e \\frac{q}{m}\\left(
                 \\boldsymbol{B} -
                 \\frac{\\gamma}{1+\\gamma}\\boldsymbol{\\beta}
                 (\\boldsymbol{\\beta}\\cdot\\boldsymbol{B}) -
                 \\boldsymbol{\\beta} \\times \\frac{\\boldsymbol{E}}{c} \\right)

        Here, :math:`a_e` is the anomalous magnetic moment of the particle,
        :math:`\\gamma` is the Lorentz factor of the particle,
        :math:`\\boldsymbol{\\beta}=\\boldsymbol{v}/c` is the normalised velocity

        The implementation of the push algorithm is detailed in
        https://arxiv.org/abs/2303.16966.

        Parameters
        ----------
        species: an fbpic Particles object
            The species with which the spin tracker is
            associated with

        dt: float (in second)
            Simulation timestep

        sx_m: float (dimensionless), optional
            The species-averaged average projection onto
            the x-axis

        sy_m: float (dimensionless), optional
            The species-averaged average projection onto
            the y-axis

        sz_m: float (dimensionless), optional
            The species-averaged average projection onto
            the z-axis

        anom: float, (dimensionless), optional
            The anomalous magnetic moment of the particle,
            given by :math:`a=(g-2)/2`, where :math:`g` is the
            particle's g-factor.

        spin_distr: str, optional
            If 'fixed', all particles will have a fixed spin value
            equal to s{x,y,z}_m.
            If 'rand', the spin vectors will be random, but with an
            ensemble average defined by one of the values of
            s{x,y,z}_m. The first non-zero mean component will be
            used, with order of preference being x,y,z, ie if sx_m!=0,
            the generated spins will have an ensemble averages of
            |sx|=sx_m, |sy|=0, |sz|=0, or if sz_m!=0, |sx|=0, |sy|=0
            and |sz|=sz_m.

        """
        self.sx_m = sx_m
        self.sy_m = sy_m
        self.sz_m = sz_m
        self.sm = np.sqrt(sx_m ** 2 + sy_m ** 2 + sz_m ** 2)
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
        if self.spin_distr == 'fixed':
            sx = np.ones(Ntot) * self.sx_m / self.sm
            sy = np.ones(Ntot) * self.sy_m / self.sm
            sz = np.ones(Ntot) * self.sz_m / self.sm
        else:
            # If the user passes a preferred spin avergae
            if self.sx_m != 0.:
                sx, sy, sz = make_random_spins(Ntot, self.sx_m)
            elif self.sy_m != 0.:
                sy, sx, sz = make_random_spins(Ntot, self.sy_m)
            elif self.sz_m != 0.:
                sz, sx, sy = make_random_spins(Ntot, self.sz_m)
            else:
                # If the user does not pass anything, all spins
                # are randomly oriented
                sx, sy, sz = random_point_sphere_cpu(Ntot)

        return sx, sy, sz

    def generate_ionized_spins_gpu(self, Ntot):
        return random_point_sphere_gpu(Ntot)

    def send_to_gpu(self):
        """
        Copy the spin data to the GPU.
        """
        self.sx = cupy.asarray(self.sx)
        self.sy = cupy.asarray(self.sy)
        self.sz = cupy.asarray(self.sz)

    def receive_from_gpu(self):
        """
        Transfer the spin data from the GPU to the CPU
        """
        self.sx = self.sx.get()
        self.sy = self.sy.get()
        self.sz = self.sz.get()


def make_random_spins(Ntot, s1_m):
    """
    Make a set of random spins. The component s1 will
    have an average value of s1_m, with its distribution
    width being up to 0.3. All spin components will be
    clipped to abs(1).
    """
    s1_th = min((0.3, 1 - abs(s1_m)))
    s1 = s1_m * np.ones(Ntot) + s1_th * np.random.normal(size=Ntot)
    s1[s1 > 1] = 1.  # keep within 1!
    s1[s1 < -1] = -1.
    s2 = 0.5 * np.random.normal(size=Ntot)
    s3 = 0.5 * np.random.normal(size=Ntot)

    ratio = np.sqrt((1 - s1 ** 2) / (s2 ** 2 + s3 ** 2))
    s2 *= ratio
    s3 *= ratio
    return s1, s2, s3
