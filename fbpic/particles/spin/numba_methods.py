# Copyright 2023, FBPIC contributors
# Author: Michael J. Quin
# Scientific supervision: Matteo Tamburini
# Code optimization: Kristjan Poder
# License: 3-Clause-BSD-LBNL
"""
This file is for the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It carries out the spin-push related methods for all particles
on CPU.
"""
import numba
from fbpic.utils.threading import njit_parallel, prange
from scipy.constants import e
from .inline_functions import push_s_BMT, copy_ionized_electron_spin_batch
from .cuda_numba_utils import random_point_sphere_cpu


push_s_BMT = numba.njit(push_s_BMT)
copy_ionized_electron_spin_batch = \
    numba.njit(copy_ionized_electron_spin_batch)
random_point_sphere_cpu = numba.njit(random_point_sphere_cpu)


@njit_parallel
def push_s_numba(sx, sy, sz, ux_prev, uy_prev, uz_prev, ux, uy, uz,
                 Ex, Ey, Ez, Bx, By, Bz, q, m, Ntot, dt, anom):
    """
    Advance particle's spin vector as it precesses around fields.

    Fora given timestep n, we expect the variables :
    s_prev, u_prev                      to be defined at time n-1/2
    E, B                                to be defined at time n
    s, u                                to be defined at time n+1/2
    """
    # Define constant pre-factor of Omega (a B-field like vector in
    # the B.M.T. equation.
    tauconst = dt * q / ( 2 * m )

    # Loop over the particles
    for ip in prange(Ntot) :
        sx[ip], sy[ip], sz[ip] = push_s_BMT(sx[ip], sy[ip], sz[ip],
                                        ux_prev[ip], uy_prev[ip], uz_prev[ip],
                                        ux[ip], uy[ip], uz[ip],
                                        Ex[ip], Ey[ip], Ez[ip],
                                        Bx[ip], By[ip], Bz[ip],
                                        tauconst, anom)
    return sx, sy, sz


@njit_parallel
def push_s_ioniz_numba(sx, sy, sz, ux_prev, uy_prev, uz_prev, ux, uy, uz,
                 Ex, Ey, Ez, Bx, By, Bz, m, Ntot, dt, anom, ionization_level):
    """
    Advance particle's spin vector as it precesses around fields.

    Fora given timestep n, we expect the variables :
    s_prev, u_prev                      to be defined at time n-1/2
    E, B                                to be defined at time n
    s, u                                to be defined at time n+1/2

    Ionizable species can have a charge that depends on the
    macroparticle, and hence require a different function

    Macroparticle charge q = e * ionization_level

    """
    # Define constant pre-factor of Omega (a B-field like vector in
    # the B.M.T. equation.
    prefactor_tauconst = dt*e/(2*m)

    # Loop over the particles
    for ip in prange(Ntot):

        # For neutral macroparticles, skip this step
        if ionization_level[ip] == 0:
            continue

        # Calculate the charge dependent constant
        tauconst = prefactor_tauconst * ionization_level[ip]

        # push spin
        sx[ip], sy[ip], sz[ip] = push_s_BMT(sx[ip], sy[ip], sz[ip],
                                        ux_prev[ip], uy_prev[ip], uz_prev[ip],
                                        ux[ip], uy[ip], uz[ip],
                                        Ex[ip], Ey[ip], Ez[ip],
                                        Bx[ip], By[ip], Bz[ip],
                                        tauconst, anom)
    return sx, sy, sz


@njit_parallel
def copy_ionized_electron_spin_numba(
        N_batch, batch_size, elec_old_Ntot, elec_new_Ntot, ion_Ntot,
        store_electrons_per_level, cumulative_n_ionized, i_level,
        ionized_from, elec_sx, elec_sy, elec_sz, ion_sx, ion_sy, ion_sz):
    """
    Generate spins for newly generated electrons.
    For the first ionized electron, the spin is copied
    from the parent ion's spin vector. For all subsequent
    ionized electrons, the spin vector is randomly oriented,
    ie sampled from sphere point picking.
    """
    # First make a set of random spins to be copied, if needed
    rand_sx, rand_sy, rand_sz = random_point_sphere_cpu(elec_new_Ntot)

    for i_batch in prange( N_batch ):
        copy_ionized_electron_spin_batch(
            i_batch, batch_size, elec_old_Ntot, ion_Ntot,
            cumulative_n_ionized, i_level, ionized_from,
            store_electrons_per_level,
            elec_sx, elec_sy, elec_sz,
            ion_sx, ion_sy, ion_sz,
            rand_sx, rand_sy, rand_sz)

    return elec_sx, elec_sy, elec_sz
