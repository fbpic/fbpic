# Copyright 2023, FBPIC contributors
# Author: Michael J. Quin
# Scientific supervision: Matteo Tamburini
# Code optimization: Kristjan Poder
# License: 3-Clause-BSD-LBNL
"""
This file is for the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It carries out the spin-push related methods for all particles
on GPU using CUDA.
"""
from scipy.constants import e
from numba import cuda
from fbpic.utils.cuda import compile_cupy
# Import inline function
from .inline_functions import push_s_BMT, copy_ionized_electron_spin_batch
from .cuda_numba_utils import random_point_sphere_gpu

# Compile the inline function for GPU
push_s_BMT = cuda.jit( push_s_BMT, device=True, inline=True )
copy_ionized_electron_spin_batch = \
    cuda.jit(copy_ionized_electron_spin_batch, device=True, 
             inline=True)
random_point_sphere_gpu = cuda.jit(random_point_sphere_gpu, 
                                   device=True, inline=True)

@compile_cupy
def push_s_gpu(sx, sy, sz, ux_prev, uy_prev, uz_prev, ux, uy, uz,
               Ex, Ey, Ez, Bx, By, Bz, q, m, Ntot, dt, anom):
    """
    Advance particle's spin vector by dt as it precesses around fields.

    For a given timestep n, we expect the variables :
    s_prev, u_prev                      to be defined at time n-1/2
    E, B                                to be defined at time n
    s, u                                to be defined at time n+1/2
    """
    # Define constant pre-factor of Omega (a B-field like vector in
    # the B.M.T. equation.
    tauconst = dt * q / ( 2 * m )

    # Cuda 1D grid (position of current thread)
    ip = cuda.grid(1)

    # Loop over the particles
    if ip < Ntot:
        sx[ip], sy[ip], sz[ip] = push_s_BMT(sx[ip], sy[ip], sz[ip],
            ux_prev[ip], uy_prev[ip], uz_prev[ip], ux[ip], uy[ip], uz[ip],
            Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], tauconst, anom)


@compile_cupy
def push_s_ioniz_gpu(sx, sy, sz, ux_prev, uy_prev, uz_prev, ux, uy, uz,
                     Ex, Ey, Ez, Bx, By, Bz, m, Ntot, dt, anom, ionization_level):
    """
    Advance particle's spin vector by dt as it precesses around fields.
    This takes into account that the particles are ionizable, and thus
    that their charge is determined by `ionization_level`

    Parameters
    ----------
    ionization_level : 1darray of ints
        The number of electrons that each ion is missing
        (compared to a neutral atom)

    For the other parameters, see the docstring of push_p_gpu
    """
    # Cuda 1D grid (position of current thread)
    ip = cuda.grid(1)

    # Loop over the particles
    if ip < Ntot:
        if ionization_level[ip] != 0:
            # Set a few constants
            tauconst = ionization_level[ip] * dt * e / ( 2 * m )
            # push spin
            sx[ip], sy[ip], sz[ip] = push_s_BMT(sx[ip], sy[ip], sz[ip],
                ux_prev[ip], uy_prev[ip], uz_prev[ip], ux[ip], uy[ip], uz[ip],
                Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], tauconst, anom)


@compile_cupy
def copy_ionized_electron_spin_cuda(
        N_batch, batch_size, elec_old_Ntot, ion_Ntot,
        store_electrons_per_level,
        cumulative_n_ionized, i_level, ionized_from, elec_sx, elec_sy, elec_sz,
        ion_sx, ion_sy, ion_sz, rand_sx, rand_sy, rand_sz):
    """
    Generate spins for newly generated electrons.
    For the first ionized electron, the spin is copied
    from the parent ion's spin vector. For all subsequent
    ionized electrons, the spin vector is randomly oriented,
    ie sampled from sphere point picking.
    """
    # And now create the spins
    i_batch = cuda.grid(1)
    if i_batch < N_batch:
        copy_ionized_electron_spin_batch(
            i_batch, batch_size, elec_old_Ntot, ion_Ntot,
            cumulative_n_ionized, i_level, ionized_from,
            store_electrons_per_level,
            elec_sx, elec_sy, elec_sz,
            ion_sx, ion_sy, ion_sz,
            rand_sx, rand_sy, rand_sz)
