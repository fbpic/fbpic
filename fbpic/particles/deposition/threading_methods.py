# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the deposition methods for rho and J for linear and cubic
order shapes on the CPU with threading.
"""
import numpy as np
import numba
from fbpic.utils.threading import njit_parallel, prange
import math
from scipy.constants import c
from fbpic.particles.deposition.particle_shapes import Sz_linear, \
    Sr_linear, Sz_cubic, Sr_cubic

# JIT-compilation of particle shapes
Sz_linear = numba.njit(Sz_linear)
Sr_linear = numba.njit(Sr_linear)
Sz_cubic = numba.njit(Sz_cubic)
Sr_cubic = numba.njit(Sr_cubic)

# -------------------------------
# Field deposition - linear - rho
# -------------------------------

@njit_parallel
def deposit_rho_numba_linear(x, y, z, w, q,
                           invdz, zmin, Nz,
                           invdr, rmin, Nr,
                           rho_global, Nm,
                           nthreads, ptcl_chunk_indices,
                           beta_n_m_0, beta_n_m_higher):
    """
    Deposition of the charge density rho using numba prange on the CPU.
    Iterates over the threads in parallel, while each thread iterates
    over a batch of particles. Intermediate results for each threads are
    stored in copies of the global grid. At the end of the parallel loop,
    the thread-local field arrays are combined (summed) to a global array.
    (This final reduction is *not* done in this function)

    Calculates the weighted amount of rho that is deposited to the
    4 cells surounding the particle based on its shape (linear).

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    w : 1d array of floats
        The weights of the particles
        (For ionizable atoms: weight times the ionization level)

    q : float
        Charge of the species
        (For ionizable atoms: this is always the elementary charge e)

    rho_global : 4darrays of complexs
        Global helper arrays of shape (nthreads, Nm, 2+Nz+2, 2+Nr+2) where the
        additional 2's in z and r correspond to deposition guard cells.
        This array stores the thread local charge density on the interpolation
        grid for each mode. (is modified by this function)

    Nm : int
        The number of azimuthal modes

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the considered direction

    Nz, Nr : int
        Number of gridpoints along the considered direction

    nthreads : int
        Number of CPU threads used with numba prange

    ptcl_chunk_indices : array of int, of size nthreads+1
        The indices (of the particle array) between which each thread
        should loop. (i.e. divisions of particle array between threads)

    beta_n_m_0 : 1darray of floats
        Ruyten-corrected particle shape factor coefficients for mode 0.

    beta_n_m_higher : 1darray of floats
        Ruyten-corrected particle shape factor coefficients for higher modes.
        Ignored when Nm == 1.
    """
    # Deposit the field per cell in parallel (for threads < number of cells)
    for i_thread in prange( nthreads ):

        # Allocate thread-local array
        rho_scal = np.zeros( Nm, dtype=np.complex128 )

        # Loop over all particles in thread chunk
        for i_ptcl in range( ptcl_chunk_indices[i_thread],
                             ptcl_chunk_indices[i_thread+1] ):

            # Position
            xj = x[i_ptcl]
            yj = y[i_ptcl]
            zj = z[i_ptcl]
            # Weights
            wj = q * w[i_ptcl]

            # Cylindrical conversion
            rj = math.sqrt(xj**2 + yj**2)
            # Avoid division by 0.
            if (rj != 0.):
                invr = 1./rj
                cos = xj*invr  # Cosine
                sin = yj*invr  # Sine
            else:
                cos = 1.
                sin = 0.
            # Calculate contribution from this particle to each mode
            rho_scal[0] = wj
            for m in range(1,Nm):
                rho_scal[m] = (cos + 1.j*sin)*rho_scal[m-1]

            # Positions of the particles, in the cell unit
            r_cell = invdr*(rj - rmin) - 0.5
            z_cell = invdz*(zj - zmin) - 0.5
            # Index of the lowest cell of `global_array` that gets modified
            # by this particle (note: `global_array` has 2 guard cells)
            # (`min` function avoids out-of-bounds access at high r)
            ir_cell = min( int(math.ceil(r_cell))+1, Nr+2 )
            iz_cell = int(math.ceil( z_cell )) + 1

            ir = min( int(math.ceil(r_cell)), Nr )

            # Add contribution of this particle to the global array
            for m in range(Nm):

                # Ruyten-corrected shape factor coefficient
                if m == 0:
                    bn = beta_n_m_0[ir]
                else:
                    bn = beta_n_m_higher[ir]

                rho_global[i_thread,m,iz_cell+0,ir_cell+0] += Sz_linear(z_cell, 0)*Sr_linear(r_cell, 0, (-1)**m, bn) * rho_scal[m]
                rho_global[i_thread,m,iz_cell+0,ir_cell+1] += Sz_linear(z_cell, 0)*Sr_linear(r_cell, 1, (-1)**m, bn) * rho_scal[m]
                rho_global[i_thread,m,iz_cell+1,ir_cell+0] += Sz_linear(z_cell, 1)*Sr_linear(r_cell, 0, (-1)**m, bn) * rho_scal[m]
                rho_global[i_thread,m,iz_cell+1,ir_cell+1] += Sz_linear(z_cell, 1)*Sr_linear(r_cell, 1, (-1)**m, bn) * rho_scal[m]

    return

# -------------------------------
# Field deposition - linear - J
# -------------------------------

@njit_parallel
def deposit_J_numba_linear(x, y, z, w, q,
                         ux, uy, uz, inv_gamma,
                         invdz, zmin, Nz,
                         invdr, rmin, Nr,
                         j_r_global, j_t_global, j_z_global, Nm,
                         nthreads, ptcl_chunk_indices,
                         beta_n_m_0, beta_n_m_higher):
    """
    Deposition of the current density J using numba prange on the CPU.
    Iterates over the threads in parallel, while each thread iterates
    over a batch of particles. Intermediate results for each threads are
    stored in copies of the global grid. At the end of the parallel loop,
    the thread-local field arrays are combined (summed) to the global array.
    (This final reduction is *not* done in this function)

    Calculates the weighted amount of J that is deposited to the
    4 cells surounding the particle based on its shape (linear).

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    w : 1d array of floats
        The weights of the particles
        (For ionizable atoms: weight times the ionization level)

    q : float
        Charge of the species
        (For ionizable atoms: this is always the elementary charge e)

    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    j_x_global : 4darrays of complexs
        Global helper arrays of shape (nthreads, Nm, 2+Nz+2, 2+Nr+2) where the
        additional 2's in z and r correspond to deposition guard cells.
        This array stores the thread local charge density on the interpolation
        grid for each mode. (is modified by this function)

    Nm : int
        The number of azimuthal modes

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the direction considered

    Nz, Nr : int
        Number of gridpoints along the considered direction

    nthreads : int
        Number of CPU threads used with numba prange

    ptcl_chunk_indices : array of int, of size nthreads+1
        The indices (of the particle array) between which each thread
        should loop. (i.e. divisions of particle array between threads)

    beta_n_m_0 : 1darray of floats
        Ruyten-corrected particle shape factor coefficients for mode 0.

    beta_n_m_higher : 1darray of floats
        Ruyten-corrected particle shape factor coefficients for higher modes.
        Ignored when Nm == 1.
    """
    # Deposit the field per cell in parallel (for threads < number of cells)
    for i_thread in prange( nthreads ):

        # Allocate thread-local array
        jr_scal = np.zeros( Nm, dtype=np.complex128 )
        jt_scal = np.zeros( Nm, dtype=np.complex128 )
        jz_scal = np.zeros( Nm, dtype=np.complex128 )

        # Loop over all particles in thread chunk
        for i_ptcl in range( ptcl_chunk_indices[i_thread],
                             ptcl_chunk_indices[i_thread+1] ):

            # Position
            xj = x[i_ptcl]
            yj = y[i_ptcl]
            zj = z[i_ptcl]
            # Velocity
            uxj = ux[i_ptcl]
            uyj = uy[i_ptcl]
            uzj = uz[i_ptcl]
            # Inverse gamma
            inv_gammaj = inv_gamma[i_ptcl]
            # Weights
            wj = q * w[i_ptcl]

            # Cylindrical conversion
            rj = math.sqrt(xj**2 + yj**2)
            # Avoid division by 0.
            if (rj != 0.):
                invr = 1./rj
                cos = xj*invr  # Cosine
                sin = yj*invr  # Sine
            else:
                cos = 1.
                sin = 0.
            # Calculate contribution from this particle to each mode
            jr_scal[0] = wj * c * inv_gammaj * (cos*uxj + sin*uyj)
            jt_scal[0] = wj * c * inv_gammaj * (cos*uyj - sin*uxj)
            jz_scal[0] = wj * c * inv_gammaj * uzj
            for m in range(1,Nm):
                jr_scal[m] = (cos + 1.j*sin) * jr_scal[m-1]
                jt_scal[m] = (cos + 1.j*sin) * jt_scal[m-1]
                jz_scal[m] = (cos + 1.j*sin) * jz_scal[m-1]

            # Positions of the particles, in the cell unit
            r_cell = invdr*(rj - rmin) - 0.5
            z_cell = invdz*(zj - zmin) - 0.5
            # Index of the lowest cell of `global_array` that gets modified
            # by this particle (note: `global_array` has 2 guard cells)
            # (`min` function avoids out-of-bounds access at high r)
            ir_cell = min( int(math.ceil(r_cell))+1, Nr+2 )
            iz_cell = int(math.ceil( z_cell )) + 1

            ir = min( int(math.ceil(r_cell)), Nr )

            # Add contribution of this particle to the global array
            for m in range(Nm):

                # Ruyten-corrected shape factor coefficient
                if m == 0:
                    bn = beta_n_m_0[ir]
                else:
                    bn = beta_n_m_higher[ir]

                j_r_global[i_thread,m,iz_cell+0,ir_cell+0] += Sz_linear(z_cell, 0)*Sr_linear(r_cell, 0, -(-1)**m, bn) * jr_scal[m]
                j_r_global[i_thread,m,iz_cell+0,ir_cell+1] += Sz_linear(z_cell, 0)*Sr_linear(r_cell, 1, -(-1)**m, bn) * jr_scal[m]
                j_r_global[i_thread,m,iz_cell+1,ir_cell+0] += Sz_linear(z_cell, 1)*Sr_linear(r_cell, 0, -(-1)**m, bn) * jr_scal[m]
                j_r_global[i_thread,m,iz_cell+1,ir_cell+1] += Sz_linear(z_cell, 1)*Sr_linear(r_cell, 1, -(-1)**m, bn) * jr_scal[m]

                j_t_global[i_thread,m,iz_cell+0,ir_cell+0] += Sz_linear(z_cell, 0)*Sr_linear(r_cell, 0, -(-1)**m, bn) * jt_scal[m]
                j_t_global[i_thread,m,iz_cell+0,ir_cell+1] += Sz_linear(z_cell, 0)*Sr_linear(r_cell, 1, -(-1)**m, bn) * jt_scal[m]
                j_t_global[i_thread,m,iz_cell+1,ir_cell+0] += Sz_linear(z_cell, 1)*Sr_linear(r_cell, 0, -(-1)**m, bn) * jt_scal[m]
                j_t_global[i_thread,m,iz_cell+1,ir_cell+1] += Sz_linear(z_cell, 1)*Sr_linear(r_cell, 1, -(-1)**m, bn) * jt_scal[m]

                j_z_global[i_thread,m,iz_cell+0,ir_cell+0] += Sz_linear(z_cell, 0)*Sr_linear(r_cell, 0, (-1)**m, bn) * jz_scal[m]
                j_z_global[i_thread,m,iz_cell+0,ir_cell+1] += Sz_linear(z_cell, 0)*Sr_linear(r_cell, 1, (-1)**m, bn) * jz_scal[m]
                j_z_global[i_thread,m,iz_cell+1,ir_cell+0] += Sz_linear(z_cell, 1)*Sr_linear(r_cell, 0, (-1)**m, bn) * jz_scal[m]
                j_z_global[i_thread,m,iz_cell+1,ir_cell+1] += Sz_linear(z_cell, 1)*Sr_linear(r_cell, 1, (-1)**m, bn) * jz_scal[m]


    return


# -------------------------------
# Field deposition - cubic - rho
# -------------------------------

@njit_parallel
def deposit_rho_numba_cubic(x, y, z, w, q,
                          invdz, zmin, Nz,
                          invdr, rmin, Nr,
                          rho_global, Nm,
                          nthreads, ptcl_chunk_indices,
                          beta_n_m_0, beta_n_m_higher):
    """
    Deposition of the charge density rho using numba prange on the CPU.
    Iterates over the threads in parallel, while each thread iterates
    over a batch of particles. Intermediate results for each threads are
    stored in copies of the global grid. At the end of the parallel loop,
    the thread-local field arrays are combined (summed) to the global array.
    (This final reduction is *not* done in this function)

    Calculates the weighted amount of rho that is deposited to the
    16 cells surounding the particle based on its shape (cubic).

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    w : 1d array of floats
        The weights of the particles
        (For ionizable atoms: weight times the ionization level)

    q : float
        Charge of the species
        (For ionizable atoms: this is always the elementary charge e)

    rho_global : 4darray of complexs
        Global helper arrays of shape (nthreads, Nm, 2+Nz+2, 2+Nr+2) where the
        additional 2's in z and r correspond to deposition guard cells.
        This array stores the thread local charge density on the interpolation
        grid for each mode. (is modified by this function)

    Nm : int
        The number of azimuthal modes

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the considered direction

    Nz, Nr : int
        Number of gridpoints along the considered direction

    nthreads : int
        Number of CPU threads used with numba prange

    ptcl_chunk_indices : array of int, of size nthreads+1
        The indices (of the particle array) between which each thread
        should loop. (i.e. divisions of particle array between threads)

    beta_n_m_0 : 1darray of floats
        Ruyten-corrected particle shape factor coefficients for mode 0.

    beta_n_m_higher : 1darray of floats
        Ruyten-corrected particle shape factor coefficients for higher modes.
        Ignored when Nm == 1.
    """
    # Deposit the field per cell in parallel (for threads < number of cells)
    for i_thread in prange( nthreads ):

        # Allocate thread-local array
        rho_scal = np.zeros( Nm, dtype=np.complex128 )

        # Loop over all particles in thread chunk
        for i_ptcl in range( ptcl_chunk_indices[i_thread],
                             ptcl_chunk_indices[i_thread+1] ):

            # Position
            xj = x[i_ptcl]
            yj = y[i_ptcl]
            zj = z[i_ptcl]
            # Weights
            wj = q * w[i_ptcl]

            # Cylindrical conversion
            rj = math.sqrt(xj**2 + yj**2)
            # Avoid division by 0.
            if (rj != 0.):
                invr = 1./rj
                cos = xj*invr  # Cosine
                sin = yj*invr  # Sine
            else:
                cos = 1.
                sin = 0.
            # Calculate contribution from this particle to each mode
            rho_scal[0] = wj
            for m in range(1,Nm):
                rho_scal[m] = (cos + 1.j*sin)*rho_scal[m-1]

            # Positions of the particles, in the cell unit
            r_cell = invdr*(rj - rmin) - 0.5
            z_cell = invdz*(zj - zmin) - 0.5
            # Index of the lowest cell of `global_array` that gets modified
            # by this particle (note: `global_array` has 2 guard cells)
            # (`min` function avoids out-of-bounds access at high r)
            ir_cell = min( int(math.ceil(r_cell)), Nr )
            iz_cell = int(math.ceil( z_cell ))

            ir = min( int(math.ceil(r_cell)), Nr )

            # Add contribution of this particle to the global array
            for m in range(Nm):

                # Ruyten-corrected shape factor coefficient
                if m == 0:
                    bn = beta_n_m_0[ir]
                else:
                    bn = beta_n_m_higher[ir]

                rho_global[i_thread,m,iz_cell+0,ir_cell+0] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 0, (-1)**m, bn)*rho_scal[m]
                rho_global[i_thread,m,iz_cell+0,ir_cell+1] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 1, (-1)**m, bn)*rho_scal[m]
                rho_global[i_thread,m,iz_cell+0,ir_cell+2] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 2, (-1)**m, bn)*rho_scal[m]
                rho_global[i_thread,m,iz_cell+0,ir_cell+3] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 3, (-1)**m, bn)*rho_scal[m]

                rho_global[i_thread,m,iz_cell+1,ir_cell+0] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 0, (-1)**m, bn)*rho_scal[m]
                rho_global[i_thread,m,iz_cell+1,ir_cell+1] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 1, (-1)**m, bn)*rho_scal[m]
                rho_global[i_thread,m,iz_cell+1,ir_cell+2] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 2, (-1)**m, bn)*rho_scal[m]
                rho_global[i_thread,m,iz_cell+1,ir_cell+3] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 3, (-1)**m, bn)*rho_scal[m]

                rho_global[i_thread,m,iz_cell+2,ir_cell+0] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 0, (-1)**m, bn)*rho_scal[m]
                rho_global[i_thread,m,iz_cell+2,ir_cell+1] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 1, (-1)**m, bn)*rho_scal[m]
                rho_global[i_thread,m,iz_cell+2,ir_cell+2] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 2, (-1)**m, bn)*rho_scal[m]
                rho_global[i_thread,m,iz_cell+2,ir_cell+3] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 3, (-1)**m, bn)*rho_scal[m]

                rho_global[i_thread,m,iz_cell+3,ir_cell+0] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 0, (-1)**m, bn)*rho_scal[m]
                rho_global[i_thread,m,iz_cell+3,ir_cell+1] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 1, (-1)**m, bn)*rho_scal[m]
                rho_global[i_thread,m,iz_cell+3,ir_cell+2] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 2, (-1)**m, bn)*rho_scal[m]
                rho_global[i_thread,m,iz_cell+3,ir_cell+3] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 3, (-1)**m, bn)*rho_scal[m]

    return

# -------------------------------
# Field deposition - cubic - J
# -------------------------------

@njit_parallel
def deposit_J_numba_cubic(x, y, z, w, q,
                        ux, uy, uz, inv_gamma,
                        invdz, zmin, Nz,
                        invdr, rmin, Nr,
                        j_r_global, j_t_global, j_z_global, Nm,
                        nthreads, ptcl_chunk_indices,
                        beta_n_m_0, beta_n_m_higher ):
    """
    Deposition of the current density J using numba prange on the CPU.
    Iterates over the threads in parallel, while each thread iterates
    over a batch of particles. Intermediate results for each threads are
    stored in copies of the global grid. At the end of the parallel loop,
    the thread-local field arrays are combined (summed) to the global array.
    (This final reduction is *not* done in this function)

    Calculates the weighted amount of J that is deposited to the
    16 cells surounding the particle based on its shape (cubic).

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    w : 1d array of floats
        The weights of the particles
        (For ionizable atoms: weight times the ionization level)

    q : float
        Charge of the species
        (For ionizable atoms: this is always the elementary charge e)

    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    j_x_global : 4darrays of complexs
        Global helper arrays of shape (nthreads, Nm, 2+Nz+2, 2+Nr+2) where the
        additional 2's in z and r correspond to deposition guard cells.
        This array stores the thread local current component in each
        direction (r, t, z) on the interpolation grid for each mode.
        (is modified by this function)

    Nm : int
        The number of azimuthal modes

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the direction considered

    Nz, Nr : int
        Number of gridpoints along the considered direction

    nthreads : int
        Number of CPU threads used with numba prange

    ptcl_chunk_indices : array of int, of size nthreads+1
        The indices (of the particle array) between which each thread
        should loop. (i.e. divisions of particle array between threads)

    beta_n_m_0 : 1darray of floats
        Ruyten-corrected particle shape factor coefficients for mode 0.

    beta_n_m_higher : 1darray of floats
        Ruyten-corrected particle shape factor coefficients for higher modes.
        Ignored when Nm == 1.
    """
    # Deposit the field per cell in parallel (for threads < number of cells)
    for i_thread in prange( nthreads ):

        # Allocate thread-local array
        jr_scal = np.zeros( Nm, dtype=np.complex128 )
        jt_scal = np.zeros( Nm, dtype=np.complex128 )
        jz_scal = np.zeros( Nm, dtype=np.complex128 )

        # Loop over all particles in thread chunk
        for i_ptcl in range( ptcl_chunk_indices[i_thread],
                             ptcl_chunk_indices[i_thread+1] ):

            # Position
            xj = x[i_ptcl]
            yj = y[i_ptcl]
            zj = z[i_ptcl]
            # Velocity
            uxj = ux[i_ptcl]
            uyj = uy[i_ptcl]
            uzj = uz[i_ptcl]
            # Inverse gamma
            inv_gammaj = inv_gamma[i_ptcl]
            # Weights
            wj = q * w[i_ptcl]

            # Cylindrical conversion
            rj = math.sqrt(xj**2 + yj**2)
            # Avoid division by 0.
            if (rj != 0.):
                invr = 1./rj
                cos = xj*invr  # Cosine
                sin = yj*invr  # Sine
            else:
                cos = 1.
                sin = 0.
            # Calculate contribution from this particle to each mode
            jr_scal[0] = wj * c * inv_gammaj * (cos*uxj + sin*uyj)
            jt_scal[0] = wj * c * inv_gammaj * (cos*uyj - sin*uxj)
            jz_scal[0] = wj * c * inv_gammaj * uzj
            for m in range(1,Nm):
                jr_scal[m] = (cos + 1.j*sin) * jr_scal[m-1]
                jt_scal[m] = (cos + 1.j*sin) * jt_scal[m-1]
                jz_scal[m] = (cos + 1.j*sin) * jz_scal[m-1]

            # Positions of the particles, in the cell unit
            r_cell = invdr*(rj - rmin) - 0.5
            z_cell = invdz*(zj - zmin) - 0.5
            # Index of the lowest cell of `global_array` that gets modified
            # by this particle (note: `global_array` has 2 guard cells)
            # (`min` function avoids out-of-bounds access at high r)
            ir_cell = min( int(math.ceil(r_cell)), Nr )
            iz_cell = int(math.ceil( z_cell ))

            ir = min( int(math.ceil(r_cell)), Nr )

            # Add contribution of this particle to the global array
            for m in range(Nm):

                # Ruyten-corrected shape factor coefficient
                if m == 0:
                    bn = beta_n_m_0[ir]
                else:
                    bn = beta_n_m_higher[ir]

                j_r_global[i_thread,m,iz_cell+0,ir_cell+0] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 0, -(-1)**m, bn)*jr_scal[m]
                j_r_global[i_thread,m,iz_cell+0,ir_cell+1] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 1, -(-1)**m, bn)*jr_scal[m]
                j_r_global[i_thread,m,iz_cell+0,ir_cell+2] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 2, -(-1)**m, bn)*jr_scal[m]
                j_r_global[i_thread,m,iz_cell+0,ir_cell+3] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 3, -(-1)**m, bn)*jr_scal[m]

                j_r_global[i_thread,m,iz_cell+1,ir_cell+0] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 0, -(-1)**m, bn)*jr_scal[m]
                j_r_global[i_thread,m,iz_cell+1,ir_cell+1] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 1, -(-1)**m, bn)*jr_scal[m]
                j_r_global[i_thread,m,iz_cell+1,ir_cell+2] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 2, -(-1)**m, bn)*jr_scal[m]
                j_r_global[i_thread,m,iz_cell+1,ir_cell+3] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 3, -(-1)**m, bn)*jr_scal[m]

                j_r_global[i_thread,m,iz_cell+2,ir_cell+0] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 0, -(-1)**m, bn)*jr_scal[m]
                j_r_global[i_thread,m,iz_cell+2,ir_cell+1] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 1, -(-1)**m, bn)*jr_scal[m]
                j_r_global[i_thread,m,iz_cell+2,ir_cell+2] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 2, -(-1)**m, bn)*jr_scal[m]
                j_r_global[i_thread,m,iz_cell+2,ir_cell+3] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 3, -(-1)**m, bn)*jr_scal[m]

                j_r_global[i_thread,m,iz_cell+3,ir_cell+0] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 0, -(-1)**m, bn)*jr_scal[m]
                j_r_global[i_thread,m,iz_cell+3,ir_cell+1] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 1, -(-1)**m, bn)*jr_scal[m]
                j_r_global[i_thread,m,iz_cell+3,ir_cell+2] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 2, -(-1)**m, bn)*jr_scal[m]
                j_r_global[i_thread,m,iz_cell+3,ir_cell+3] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 3, -(-1)**m, bn)*jr_scal[m]

                j_t_global[i_thread,m,iz_cell+0,ir_cell+0] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 0, -(-1)**m, bn)*jt_scal[m]
                j_t_global[i_thread,m,iz_cell+0,ir_cell+1] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 1, -(-1)**m, bn)*jt_scal[m]
                j_t_global[i_thread,m,iz_cell+0,ir_cell+2] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 2, -(-1)**m, bn)*jt_scal[m]
                j_t_global[i_thread,m,iz_cell+0,ir_cell+3] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 3, -(-1)**m, bn)*jt_scal[m]

                j_t_global[i_thread,m,iz_cell+1,ir_cell+0] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 0, -(-1)**m, bn)*jt_scal[m]
                j_t_global[i_thread,m,iz_cell+1,ir_cell+1] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 1, -(-1)**m, bn)*jt_scal[m]
                j_t_global[i_thread,m,iz_cell+1,ir_cell+2] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 2, -(-1)**m, bn)*jt_scal[m]
                j_t_global[i_thread,m,iz_cell+1,ir_cell+3] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 3, -(-1)**m, bn)*jt_scal[m]

                j_t_global[i_thread,m,iz_cell+2,ir_cell+0] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 0, -(-1)**m, bn)*jt_scal[m]
                j_t_global[i_thread,m,iz_cell+2,ir_cell+1] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 1, -(-1)**m, bn)*jt_scal[m]
                j_t_global[i_thread,m,iz_cell+2,ir_cell+2] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 2, -(-1)**m, bn)*jt_scal[m]
                j_t_global[i_thread,m,iz_cell+2,ir_cell+3] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 3, -(-1)**m, bn)*jt_scal[m]

                j_t_global[i_thread,m,iz_cell+3,ir_cell+0] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 0, -(-1)**m, bn)*jt_scal[m]
                j_t_global[i_thread,m,iz_cell+3,ir_cell+1] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 1, -(-1)**m, bn)*jt_scal[m]
                j_t_global[i_thread,m,iz_cell+3,ir_cell+2] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 2, -(-1)**m, bn)*jt_scal[m]
                j_t_global[i_thread,m,iz_cell+3,ir_cell+3] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 3, -(-1)**m, bn)*jt_scal[m]

                j_z_global[i_thread,m,iz_cell+0,ir_cell+0] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 0, (-1)**m, bn)*jz_scal[m]
                j_z_global[i_thread,m,iz_cell+0,ir_cell+1] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 1, (-1)**m, bn)*jz_scal[m]
                j_z_global[i_thread,m,iz_cell+0,ir_cell+2] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 2, (-1)**m, bn)*jz_scal[m]
                j_z_global[i_thread,m,iz_cell+0,ir_cell+3] += Sz_cubic(z_cell, 0)*Sr_cubic(r_cell, 3, (-1)**m, bn)*jz_scal[m]

                j_z_global[i_thread,m,iz_cell+1,ir_cell+0] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 0, (-1)**m, bn)*jz_scal[m]
                j_z_global[i_thread,m,iz_cell+1,ir_cell+1] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 1, (-1)**m, bn)*jz_scal[m]
                j_z_global[i_thread,m,iz_cell+1,ir_cell+2] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 2, (-1)**m, bn)*jz_scal[m]
                j_z_global[i_thread,m,iz_cell+1,ir_cell+3] += Sz_cubic(z_cell, 1)*Sr_cubic(r_cell, 3, (-1)**m, bn)*jz_scal[m]

                j_z_global[i_thread,m,iz_cell+2,ir_cell+0] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 0, (-1)**m, bn)*jz_scal[m]
                j_z_global[i_thread,m,iz_cell+2,ir_cell+1] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 1, (-1)**m, bn)*jz_scal[m]
                j_z_global[i_thread,m,iz_cell+2,ir_cell+2] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 2, (-1)**m, bn)*jz_scal[m]
                j_z_global[i_thread,m,iz_cell+2,ir_cell+3] += Sz_cubic(z_cell, 2)*Sr_cubic(r_cell, 3, (-1)**m, bn)*jz_scal[m]

                j_z_global[i_thread,m,iz_cell+3,ir_cell+0] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 0, (-1)**m, bn)*jz_scal[m]
                j_z_global[i_thread,m,iz_cell+3,ir_cell+1] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 1, (-1)**m, bn)*jz_scal[m]
                j_z_global[i_thread,m,iz_cell+3,ir_cell+2] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 2, (-1)**m, bn)*jz_scal[m]
                j_z_global[i_thread,m,iz_cell+3,ir_cell+3] += Sz_cubic(z_cell, 3)*Sr_cubic(r_cell, 3, (-1)**m, bn)*jz_scal[m]

    return
