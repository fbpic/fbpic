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
from scipy.optimize import fsolve
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
             \\boldsymbol{\\Omega}_a) \\times \\boldsymbol{s}

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
            The species-averaged average projection onto the x-axis

        sy_m: float (dimensionless), optional
            The species-averaged average projection onto the y-axis

        sz_m: float (dimensionless), optional
            The species-averaged average projection onto the z-axis

        anom: float, (dimensionless), optional
            The anomalous magnetic moment of the particle,
            given by :math:`a=(g-2)/2`, where :math:`g` is the
            particle's g-factor.

        spin_distr: str, optional
            If 'fixed', all particles will have a fixed spin value
            equal to s{x,y,z}_m. Note the sum of squares of the components
            must add up to 1, otherwise an error will be raised.

            If 'rand', the spin vectors will be random, but with an
            ensemble average defined by one of the values of
            s{x,y,z}_m. Only one of s{x,y,z}_m may be given and an
            error will be raised if two or more components are specified.
            The generated spins will have an ensemble averages of
            <sx>=sx_m, <sy>=0, <sz>=0 if sx_m was specified. Note, if
            none of s{x,y,z}_m are passed, the spins will be randomly
            generated (by picking points on sphere of radius 1).

        """
        self.sx_m = sx_m
        self.sy_m = sy_m
        self.sz_m = sz_m
        self.sm = np.sqrt(sx_m ** 2 + sy_m ** 2 + sz_m ** 2)
        self.anom = anom
        self.dt = dt
        self.spin_distr = spin_distr
        self.alpha = None

        # Check the spin vector length is correct for the 'fixed' distribution
        if spin_distr == 'fixed' and self.sm != 1.:
            raise ValueError("When using 'fixed' spin distribution, the "
                             "squared sum of mean values must be equal to 1!")

        if abs(sx_m) > 1 or abs(sy_m) > 1 or abs(sz_m) > 1:
            raise ValueError("Mean spin projection onto axes must not "
                             "be greater than 1!")

        # Check that only one spin component is given for rand distribution
        if spin_distr == 'rand':
            assert len(np.nonzero((sx_m, sy_m, sz_m))[0]) == 1, \
                ('Only one non-zero spin-component may be '
                 'passed if spin_distr is "rand"')

        self.use_cuda = species.use_cuda

        # Register arrays for previous timestep spins
        self.ux_prev = None
        self.uy_prev = None
        self.uz_prev = None

        # Create the initial spin array
        self.sx, self.sy, self.sz = self.generate_new_spins(species.Ntot)

    def store_previous_momenta(self, species):
        """
        Store the momenta at the previous half timestep, ie at
        t = (n-1/2) dt
        
        Parameters
        ----------
        species: an fbpic.Species object
            The particle species whos spin is being tracked
        """
        if self.use_cuda:
            self.ux_prev = cupy.array(species.ux, order='C')
            self.uy_prev = cupy.array(species.uy, order='C')
            self.uz_prev = cupy.array(species.uz, order='C')
        else:
            self.ux_prev = np.array(species.ux, order='C')
            self.uy_prev = np.array(species.uy, order='C')
            self.uz_prev = np.array(species.uz, order='C')

    def push_s(self, species):
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
        
        Parameters
        ----------
        species: an fbpic.Species object
            The particle species whose spin is being tracked
        """
        # No precession for neutral particle
        if species.q == 0:
            return

        # GPU (CUDA) version
        if self.use_cuda:
            # Get the threads per block and the blocks per grid
            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( species.Ntot )
            # Call the CUDA Kernel for the spin push
            if species.ionizer is not None:
                # Ionizable species can have a charge that depends on the
                # macroparticle, and hence require a different function
                push_s_ioniz_gpu[dim_grid_1d, dim_block_1d](
                    self.sx, self.sy, self.sz,
                    self.ux_prev, self.uy_prev, self.uz_prev,
                    species.ux, species.uy, species.uz,
                    species.Ex, species.Ey, species.Ez,
                    species.Bx, species.By, species.Bz,
                    species.m, species.Ntot, self.dt, self.anom,
                    species.ionizer.ionization_level)
            else:
                # Standard pusher
                push_s_gpu[dim_grid_1d, dim_block_1d](
                    self.sx, self.sy, self.sz,
                    self.ux_prev, self.uy_prev, self.uz_prev,
                    species.ux, species.uy, species.uz,
                    species.Ex, species.Ey, species.Ez,
                    species.Bx, species.By, species.Bz,
                    species.q, species.m, species.Ntot,
                    self.dt, self.anom)
        # CPU version
        else:
            if species.ionizer is not None:
                # Ionizable species can have a charge that depends on the
                # macroparticle, and hence require a different function
                push_s_ioniz_numba(self.sx, self.sy, self.sz,
                                   self.ux_prev, self.uy_prev, self.uz_prev,
                                   species.ux, species.uy, species.uz,
                                   species.Ex, species.Ey, species.Ez,
                                   species.Bx, species.By, species.Bz,
                                   species.m, species.Ntot,
                                   self.dt, self.anom,
                                   species.ionizer.ionization_level)
            else:
                # Standard spin pusher
                push_s_numba(self.sx, self.sy, self.sz,
                             self.ux_prev, self.uy_prev, self.uz_prev,
                             species.ux, species.uy, species.uz,
                             species.Ex, species.Ey, species.Ez,
                             species.Bx, species.By, species.Bz,
                             species.q, species.m, species.Ntot,
                             self.dt, self.anom)

    def generate_new_spins(self, Ntot):
        """
        Create new spin vectors for particles. This method
        generates spin components, where the ensemble
        averages satisfy the distribution (spin_distr) and
        averages (sx_m, sy_m, sz_m) requested by the user.

        Parameters
        ----------
        Ntot: int
            Number of spin vectors to make
        """
        if self.spin_distr == 'fixed':
            sx = np.ones(Ntot) * self.sx_m
            sy = np.ones(Ntot) * self.sy_m
            sz = np.ones(Ntot) * self.sz_m
        else:
            # If the user passes a preferred spin avergae
            if self.sx_m != 0.:
                sy, sz, sx = self.make_random_spins(Ntot, self.sx_m)
            elif self.sy_m != 0.:
                sz, sx, sy = self.make_random_spins(Ntot, self.sy_m)
            elif self.sz_m != 0.:
                sx, sy, sz = self.make_random_spins(Ntot, self.sz_m)
            else:
                # If the user does not pass anything, all spins
                # are randomly oriented
                sx, sy, sz = self.make_random_spins(Ntot, 0.)

        return sx, sy, sz

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

    def make_random_spins(self, Ntot, s_m):
        """
        Generates a random distribution of points on the surface
        of a sphere of radius 1, such that the mean along z is `s_m`.
        For s_m=0, points are uniformly distributed
        on the surface of the sphere.

        Parameters
        ----------
        Ntot: int
            Number of spin vectors to generate

        s_m: float
            Mean value of z-projection of the generated spins. Note,
            if s_m = 0, the points will be randomly picked on the sphere.

        Returns
        -------
        x: np.array of length Ntot
            x-components of new particles.

        y: np.array of length Ntot
            y-components of new particles.

        z: np.array of length Ntot
            z-components of new particles.
        """
        if abs(s_m) == 1:
            z = s_m * np.ones(Ntot)  # z has a fixed value in this case
        elif s_m == 0:
            z = np.random.uniform(-1, 1, Ntot)  # random uniform
        else:
            # Generate z coord with a probability distribution
            # e^(-alpha z) between -1 and 1. First find the value of alpha
            # that correspond to s1_m using the fact that the mean of
            # the probability distribution e^(-alpha z)
            # is <z> = (1/alpha - 1/tanh(alpha))

            if self.alpha is None:
                sol = fsolve(lambda alpha: 1. / alpha -
                                           1. / np.tanh(alpha) - s_m, 0.5)
                self.alpha = sol[0]

            # Generate z with probability distribution e^(-alpha z)
            # by using the inverse CDF method
            u = np.random.uniform(0, 1, Ntot)
            z = -1 / self.alpha * np.log( np.exp(self.alpha)*(1-u) +
                                          np.exp(-self.alpha)*u )

        # azimuthal angle
        phi = np.random.uniform(0, 2 * np.pi, Ntot)
        # polar angle
        sin_theta = np.sin(np.arccos(z))
        # cartesian coords x and y
        x = sin_theta*np.cos(phi)
        y = sin_theta*np.sin(phi)
        return x, y, z
