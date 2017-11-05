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
from numba import int64
from fbpic.threading_utils import njit_parallel, prange
import math
from scipy.constants import c

# -------------------------------
# Particle shape Factor functions
# -------------------------------

# Linear shapes
@numba.njit
def Sz_linear(cell_position, index):
    iz = int64(math.floor(cell_position))
    if index == 0:
        return iz+1.-cell_position
    if index == 1:
        return cell_position - iz

@numba.njit
def Sr_linear(cell_position, index):
    flip_factor = 1.
    ir = int64(math.floor(cell_position))
    if index == 0:
        if ir < 0:
            flip_factor = -1.
        return flip_factor*(ir+1.-cell_position)
    if index == 1:
        return flip_factor*(cell_position - ir)

# Cubic shapes
@numba.njit
def Sz_cubic(cell_position, index):
    iz = int64(math.floor(cell_position)) - 1
    if index == 0:
        return (-1./6.)*((cell_position-iz)-2)**3
    if index == 1:
        return (1./6.)*(3*((cell_position-(iz+1))**3)-6*((cell_position-(iz+1))**2)+4)
    if index == 2:
        return (1./6.)*(3*(((iz+2)-cell_position)**3)-6*(((iz+2)-cell_position)**2)+4)
    if index == 3:
        return (-1./6.)*(((iz+3)-cell_position)-2)**3

@numba.njit
def Sr_cubic(cell_position, index):
    flip_factor = 1.
    ir = int64(math.floor(cell_position)) - 1
    if index == 0:
        if ir < 0:
            flip_factor = -1.
        return flip_factor*(-1./6.)*((cell_position-ir)-2)**3
    if index == 1:
        if ir+1 < 0:
            flip_factor = -1.
        return flip_factor*(1./6.)*(3*((cell_position-(ir+1))**3)-6*((cell_position-(ir+1))**2)+4)
    if index == 2:
        if ir+2 < 0:
            flip_factor = -1.
        return flip_factor*(1./6.)*(3*(((ir+2)-cell_position)**3)-6*(((ir+2)-cell_position)**2)+4)
    if index == 3:
        if ir+3 < 0:
            flip_factor = -1.
        return flip_factor*(-1./6.)*(((ir+3)-cell_position)-2)**3

# -------------------------------
# Field deposition - linear - rho
# -------------------------------

@njit_parallel
def deposit_rho_numba_linear(x, y, z, w, q,
                           invdz, zmin, Nz,
                           invdr, rmin, Nr,
                           rho_global, Nm,
                           nthreads, ptcl_chunk_indices):
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
    """
    # Deposit the field per cell in parallel (for threads < number of cells)
    for i_thread in prange( nthreads ):

        # Allocate thread-local array
        rho_scal = np.zeros( Nm, dtype=np.complex128 )

        # Loop over all particles in thread chunk
        for i_ptcl in range( ptcl_chunk_indices[i_thread],
                             ptcl_chunk_indices[i_thread+1] ):
            # Preliminary arrays for the cylindrical conversion
            # --------------------------------------------
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

            # Positions of the particles, in the cell unit
            r_cell = invdr*(rj - rmin) - 0.5
            z_cell = invdz*(zj - zmin) - 0.5

            # Calculate contribution from this particle to each mode
            rho_scal[0] = wj
            for m in range(1,Nm):
                rho_scal[m] = (cos + 1.j*sin)*rho_scal[m-1]

            # Original index of the uppper and lower cell
            ir_cell = int(math.floor( r_cell ))
            iz_cell = int(math.floor( z_cell ))

            # Declare local field array
            R_m0_00 = 0.
            R_m0_01 = 0.
            R_m0_10 = 0.
            R_m0_11 = 0.

            R_m1_00 = 0. + 0.j
            R_m1_01 = 0. + 0.j
            R_m1_10 = 0. + 0.j
            R_m1_11 = 0. + 0.j

            R_m0_00 += Sr_linear(r_cell, 0)*Sz_linear(z_cell, 0) * rho_scal[0]
            R_m0_01 += Sr_linear(r_cell, 0)*Sz_linear(z_cell, 1) * rho_scal[0]
            R_m1_00 += Sr_linear(r_cell, 0)*Sz_linear(z_cell, 0) * rho_scal[1]
            R_m1_01 += Sr_linear(r_cell, 0)*Sz_linear(z_cell, 1) * rho_scal[1]
            R_m0_10 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 0) * rho_scal[0]
            R_m0_11 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 1) * rho_scal[0]
            R_m1_10 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 0) * rho_scal[1]
            R_m1_11 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 1) * rho_scal[1]

            # Write ptcl fields to thread-local part of global deposition array
            rho_global[i_thread, 0, iz_cell+2, ir_cell+2] += R_m0_00
            rho_global[i_thread, 1, iz_cell+2, ir_cell+2] += R_m1_00

            rho_global[i_thread, 0, iz_cell+1+2, ir_cell+2] += R_m0_01
            rho_global[i_thread, 1, iz_cell+1+2, ir_cell+2] += R_m1_01

            rho_global[i_thread, 0, iz_cell+2, ir_cell+1+2] += R_m0_10
            rho_global[i_thread, 1, iz_cell+2, ir_cell+1+2] += R_m1_10

            rho_global[i_thread, 0, iz_cell+1+2, ir_cell+1+2] += R_m0_11
            rho_global[i_thread, 1, iz_cell+1+2, ir_cell+1+2] += R_m1_11

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
                         nthreads, ptcl_chunk_indices):
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
            # Preliminary arrays for the cylindrical conversion
            # --------------------------------------------
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

            # Get weights for the deposition
            # --------------------------------------------
            # Positions of the particles, in the cell unit
            r_cell = invdr*(rj - rmin) - 0.5
            z_cell = invdz*(zj - zmin) - 0.5

            # Calculate contribution from this particle to each mode
            jr_scal[0] = wj * c * inv_gammaj * (cos*uxj + sin*uyj)
            jt_scal[0] = wj * c * inv_gammaj * (cos*uyj - sin*uxj)
            jz_scal[0] = wj * c * inv_gammaj * uzj
            for m in range(1,Nm):
                jr_scal[m] = (cos + 1.j*sin) * jr_scal[m-1]
                jt_scal[m] = (cos + 1.j*sin) * jt_scal[m-1]
                jz_scal[m] = (cos + 1.j*sin) * jz_scal[m-1]

            # Original index of the uppper and lower cell
            ir_cell = int(math.floor( r_cell ))
            iz_cell = int(math.floor( z_cell ))

            # Treat the boundary conditions
            # guard cells in lower r
            if ir_cell < 0:
                ir_cell = 0
            # absorbing in upper r
            if ir_cell > Nr-1:
                ir_cell = Nr-1
            # periodic boundaries in z
            if iz_cell < 0:
                iz_cell += Nz
            if iz_cell > Nz-1:
                iz_cell -= Nz

            # Boundary Region Shifts
            ir_flip = int( math.floor(r_cell) )

            # Declare local field arrays
            J_r_m0_00 = 0.
            J_r_m1_00 = 0. + 0.j
            J_t_m0_00 = 0.
            J_t_m1_00 = 0. + 0.j
            J_z_m0_00 = 0.
            J_z_m1_00 = 0. + 0.j

            J_r_m0_01 = 0.
            J_r_m1_01 = 0. + 0.j
            J_t_m0_01 = 0.
            J_t_m1_01 = 0. + 0.j
            J_z_m0_01 = 0.
            J_z_m1_01 = 0. + 0.j

            J_r_m0_10 = 0.
            J_r_m1_10 = 0. + 0.j
            J_t_m0_10 = 0.
            J_t_m1_10 = 0. + 0.j
            J_z_m0_10 = 0.
            J_z_m1_10 = 0. + 0.j

            J_r_m0_11 = 0.
            J_r_m1_11 = 0. + 0.j
            J_t_m0_11 = 0.
            J_t_m1_11 = 0. + 0.j
            J_z_m0_11 = 0.
            J_z_m1_11 = 0. + 0.j

            J_r_m0_00 += Sr_linear(r_cell, 0)*Sz_linear(z_cell, 0) * jr_scal[0]
            J_t_m0_00 += Sr_linear(r_cell, 0)*Sz_linear(z_cell, 0) * jt_scal[0]
            J_z_m0_00 += Sr_linear(r_cell, 0)*Sz_linear(z_cell, 0) * jz_scal[0]
            J_r_m0_01 += Sr_linear(r_cell, 0)*Sz_linear(z_cell, 1) * jr_scal[0]
            J_t_m0_01 += Sr_linear(r_cell, 0)*Sz_linear(z_cell, 1) * jt_scal[0]
            J_z_m0_01 += Sr_linear(r_cell, 0)*Sz_linear(z_cell, 1) * jz_scal[0]
            J_r_m1_00 += Sr_linear(r_cell, 0)*Sz_linear(z_cell, 0) * jr_scal[1]
            J_t_m1_00 += Sr_linear(r_cell, 0)*Sz_linear(z_cell, 0) * jt_scal[1]
            J_z_m1_00 += Sr_linear(r_cell, 0)*Sz_linear(z_cell, 0) * jz_scal[1]
            J_r_m1_01 += Sr_linear(r_cell, 0)*Sz_linear(z_cell, 1) * jr_scal[1]
            J_t_m1_01 += Sr_linear(r_cell, 0)*Sz_linear(z_cell, 1) * jt_scal[1]
            J_z_m1_01 += Sr_linear(r_cell, 0)*Sz_linear(z_cell, 1) * jz_scal[1]

            # Take into account lower r flips
            if ir_flip == -1:
                J_r_m0_00 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 0) * jr_scal[0]
                J_t_m0_00 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 0) * jt_scal[0]
                J_z_m0_00 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 0) * jz_scal[0]
                J_r_m0_01 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 1) * jr_scal[0]
                J_t_m0_01 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 1) * jt_scal[0]
                J_z_m0_01 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 1) * jz_scal[0]
                J_r_m1_00 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 0) * jr_scal[1]
                J_t_m1_00 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 0) * jt_scal[1]
                J_z_m1_00 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 0) * jz_scal[1]
                J_r_m1_01 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 1) * jr_scal[1]
                J_t_m1_01 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 1) * jt_scal[1]
                J_z_m1_01 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 1) * jz_scal[1]
            else:
                J_r_m0_10 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 0) * jr_scal[0]
                J_t_m0_10 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 0) * jt_scal[0]
                J_z_m0_10 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 0) * jz_scal[0]
                J_r_m0_11 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 1) * jr_scal[0]
                J_t_m0_11 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 1) * jt_scal[0]
                J_z_m0_11 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 1) * jz_scal[0]
                J_r_m1_10 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 0) * jr_scal[1]
                J_t_m1_10 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 0) * jt_scal[1]
                J_z_m1_10 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 0) * jz_scal[1]
                J_r_m1_11 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 1) * jr_scal[1]
                J_t_m1_11 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 1) * jt_scal[1]
                J_z_m1_11 += Sr_linear(r_cell, 1)*Sz_linear(z_cell, 1) * jz_scal[1]

            # Cell shifts for the simulation boundaries
            shift_r = 0
            shift_z = 0
            if (ir_cell+1) > (Nr-1):
                shift_r = -1
            if (iz_cell+1) > Nz-1:
                shift_z -= Nz

            # Write ptcl fields to thread-local part of global deposition array
            j_r_global[i_thread, 0,iz_cell+2, ir_cell+2] += J_r_m0_00
            j_r_global[i_thread, 1,iz_cell+2, ir_cell+2] += J_r_m1_00

            j_r_global[i_thread, 0,iz_cell+1 + shift_z+2, ir_cell+2] += J_r_m0_01
            j_r_global[i_thread, 1,iz_cell+1 + shift_z+2, ir_cell+2] += J_r_m1_01

            j_r_global[i_thread, 0,iz_cell+2, ir_cell+1 + shift_r+2] += J_r_m0_10
            j_r_global[i_thread, 1,iz_cell+2, ir_cell+1 + shift_r+2] += J_r_m1_10

            j_r_global[i_thread, 0,iz_cell+1 + shift_z+2, ir_cell+1 + shift_r+2] += J_r_m0_11
            j_r_global[i_thread, 1,iz_cell+1 + shift_z+2, ir_cell+1 + shift_r+2] += J_r_m1_11

            j_t_global[i_thread, 0,iz_cell+2, ir_cell+2] += J_t_m0_00
            j_t_global[i_thread, 1,iz_cell+2, ir_cell+2] += J_t_m1_00

            j_t_global[i_thread, 0,iz_cell+1 + shift_z+2, ir_cell+2] += J_t_m0_01
            j_t_global[i_thread, 1,iz_cell+1 + shift_z+2, ir_cell+2] += J_t_m1_01

            j_t_global[i_thread, 0,iz_cell+2, ir_cell+1 + shift_r+2] += J_t_m0_10
            j_t_global[i_thread, 1,iz_cell+2, ir_cell+1 + shift_r+2] += J_t_m1_10

            j_t_global[i_thread, 0,iz_cell+1 + shift_z+2, ir_cell+1 + shift_r+2] += J_t_m0_11
            j_t_global[i_thread, 1,iz_cell+1 + shift_z+2, ir_cell+1 + shift_r+2] += J_t_m1_11

            j_z_global[i_thread, 0,iz_cell+2, ir_cell+2] += J_z_m0_00
            j_z_global[i_thread, 1,iz_cell+2, ir_cell+2] += J_z_m1_00

            j_z_global[i_thread, 0,iz_cell+1 + shift_z+2, ir_cell+2] += J_z_m0_01
            j_z_global[i_thread, 1,iz_cell+1 + shift_z+2, ir_cell+2] += J_z_m1_01

            j_z_global[i_thread, 0,iz_cell+2, ir_cell+1 + shift_r+2] += J_z_m0_10
            j_z_global[i_thread, 1,iz_cell+2, ir_cell+1 + shift_r+2] += J_z_m1_10

            j_z_global[i_thread, 0,iz_cell+1 + shift_z+2, ir_cell+1 + shift_r+2] += J_z_m0_11
            j_z_global[i_thread, 1,iz_cell+1 + shift_z+2, ir_cell+1 + shift_r+2] += J_z_m1_11

    return


# -------------------------------
# Field deposition - cubic - rho
# -------------------------------

@njit_parallel
def deposit_rho_numba_cubic(x, y, z, w, q,
                          invdz, zmin, Nz,
                          invdr, rmin, Nr,
                          rho_global, Nm,
                          nthreads, ptcl_chunk_indices):
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
    """
    # Deposit the field per cell in parallel (for threads < number of cells)
    for i_thread in prange( nthreads ):

        # Allocate thread-local array
        rho_scal = np.zeros( Nm, dtype=np.complex128 )

        # Loop over all particles in thread chunk
        for i_ptcl in range( ptcl_chunk_indices[i_thread],
                             ptcl_chunk_indices[i_thread+1] ):
            # Preliminary arrays for the cylindrical conversion
            # --------------------------------------------
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

            # Positions of the particles, in the cell unit
            r_cell = invdr*(rj - rmin) - 0.5
            z_cell = invdz*(zj - zmin) - 0.5

            # Calculate contribution from this particle to each mode
            rho_scal[0] = wj
            for m in range(1,Nm):
                rho_scal[m] = (cos + 1.j*sin)*rho_scal[m-1]

            # Original index of the uppper and lower cell
            ir_cell = int(math.floor( r_cell ))
            iz_cell = int(math.floor( z_cell ))

            # Declare the local field value for
            # all possible deposition directions,
            # depending on the shape order and per mode.
            R_m0_00 = 0.
            R_m1_00 = 0. + 0.j

            R_m0_01 = 0.
            R_m1_01 = 0. + 0.j

            R_m0_02 = 0.
            R_m1_02 = 0. + 0.j

            R_m0_03 = 0.
            R_m1_03 = 0. + 0.j

            R_m0_10 = 0.
            R_m1_10 = 0. + 0.j

            R_m0_11 = 0.
            R_m1_11 = 0. + 0.j

            R_m0_12 = 0.
            R_m1_12 = 0. + 0.j

            R_m0_13 = 0.
            R_m1_13 = 0. + 0.j

            R_m0_20 = 0.
            R_m1_20 = 0. + 0.j

            R_m0_21 = 0.
            R_m1_21 = 0. + 0.j

            R_m0_22 = 0.
            R_m1_22 = 0. + 0.j

            R_m0_23 = 0.
            R_m1_23 = 0. + 0.j

            R_m0_30 = 0.
            R_m1_30 = 0. + 0.j

            R_m0_31 = 0.
            R_m1_31 = 0. + 0.j

            R_m0_32 = 0.
            R_m1_32 = 0. + 0.j

            R_m0_33 = 0.
            R_m1_33 = 0. + 0.j

            R_m0_00 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*rho_scal[0]
            R_m1_00 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*rho_scal[1]
            R_m0_01 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*rho_scal[0]
            R_m1_01 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*rho_scal[1]
            R_m0_02 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*rho_scal[0]
            R_m1_02 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*rho_scal[1]
            R_m0_03 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*rho_scal[0]
            R_m1_03 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*rho_scal[1]

            R_m0_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*rho_scal[0]
            R_m1_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*rho_scal[1]
            R_m0_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*rho_scal[0]
            R_m1_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*rho_scal[1]
            R_m0_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*rho_scal[0]
            R_m1_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*rho_scal[1]
            R_m0_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*rho_scal[0]
            R_m1_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*rho_scal[1]

            R_m0_20 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*rho_scal[0]
            R_m1_20 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*rho_scal[1]
            R_m0_21 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*rho_scal[0]
            R_m1_21 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*rho_scal[1]
            R_m0_22 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*rho_scal[0]
            R_m1_22 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*rho_scal[1]
            R_m0_23 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*rho_scal[0]
            R_m1_23 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*rho_scal[1]

            R_m0_30 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*rho_scal[0]
            R_m1_30 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*rho_scal[1]
            R_m0_31 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*rho_scal[0]
            R_m1_31 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*rho_scal[1]
            R_m0_32 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*rho_scal[0]
            R_m1_32 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*rho_scal[1]
            R_m0_33 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*rho_scal[0]
            R_m1_33 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*rho_scal[1]

            # Write ptcl fields to thread-local part of global deposition array
            rho_global[i_thread, 0, iz_cell - 1 +2, ir_cell - 1 + 2] += R_m0_00
            rho_global[i_thread, 1, iz_cell - 1 +2, ir_cell - 1 + 2] += R_m1_00
            rho_global[i_thread, 0, iz_cell+2, ir_cell - 1 + 2] += R_m0_01
            rho_global[i_thread, 1, iz_cell+2, ir_cell - 1 + 2] += R_m1_01
            rho_global[i_thread, 0, iz_cell + 1 +2, ir_cell - 1 + 2] += R_m0_02
            rho_global[i_thread, 1, iz_cell + 1 +2, ir_cell - 1 + 2] += R_m1_02
            rho_global[i_thread, 0, iz_cell + 2 +2, ir_cell - 1 + 2] += R_m0_03
            rho_global[i_thread, 1, iz_cell + 2 +2, ir_cell - 1 + 2] += R_m1_03
            rho_global[i_thread, 0, iz_cell - 1 +2, ir_cell+2] += R_m0_10
            rho_global[i_thread, 1, iz_cell - 1 +2, ir_cell+2] += R_m1_10
            rho_global[i_thread, 0, iz_cell+2, ir_cell+2] += R_m0_11
            rho_global[i_thread, 1, iz_cell+2, ir_cell+2] += R_m1_11
            rho_global[i_thread, 0, iz_cell + 1 +2, ir_cell+2] += R_m0_12
            rho_global[i_thread, 1, iz_cell + 1 +2, ir_cell+2] += R_m1_12
            rho_global[i_thread, 0, iz_cell + 2 +2, ir_cell+2] += R_m0_13
            rho_global[i_thread, 1, iz_cell + 2 +2, ir_cell+2] += R_m1_13
            rho_global[i_thread, 0, iz_cell - 1 +2, ir_cell + 1 +2] += R_m0_20
            rho_global[i_thread, 1, iz_cell - 1 +2, ir_cell + 1 +2] += R_m1_20
            rho_global[i_thread, 0, iz_cell+2, ir_cell + 1 +2] += R_m0_21
            rho_global[i_thread, 1, iz_cell+2, ir_cell + 1 +2] += R_m1_21
            rho_global[i_thread, 0, iz_cell + 1 +2, ir_cell + 1 +2] += R_m0_22
            rho_global[i_thread, 1, iz_cell + 1 +2, ir_cell + 1 +2] += R_m1_22
            rho_global[i_thread, 0, iz_cell + 2 +2, ir_cell + 1 +2] += R_m0_23
            rho_global[i_thread, 1, iz_cell + 2 +2, ir_cell + 1 +2] += R_m1_23
            rho_global[i_thread, 0, iz_cell - 1 +2, ir_cell + 2 +2] += R_m0_30
            rho_global[i_thread, 1, iz_cell - 1 +2, ir_cell + 2 +2] += R_m1_30
            rho_global[i_thread, 0, iz_cell+2, ir_cell + 2 +2] += R_m0_31
            rho_global[i_thread, 1, iz_cell+2, ir_cell + 2 +2] += R_m1_31
            rho_global[i_thread, 0, iz_cell + 1 +2, ir_cell + 2 +2] += R_m0_32
            rho_global[i_thread, 1, iz_cell + 1 +2, ir_cell + 2 +2] += R_m1_32
            rho_global[i_thread, 0, iz_cell + 2 +2, ir_cell + 2 +2] += R_m0_33
            rho_global[i_thread, 1, iz_cell + 2 +2, ir_cell + 2 +2] += R_m1_33

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
                        nthreads, ptcl_chunk_indices):
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
            # Preliminary arrays for the cylindrical conversion
            # --------------------------------------------
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

            # Get weights for the deposition
            # --------------------------------------------
            # Positions of the particles, in the cell unit
            r_cell = invdr*(rj - rmin) - 0.5
            z_cell = invdz*(zj - zmin) - 0.5

            # Calculate contribution from this particle to each mode
            jr_scal[0] = wj * c * inv_gammaj * (cos*uxj + sin*uyj)
            jt_scal[0] = wj * c * inv_gammaj * (cos*uyj - sin*uxj)
            jz_scal[0] = wj * c * inv_gammaj * uzj
            for m in range(1,Nm):
                jr_scal[m] = (cos + 1.j*sin) * jr_scal[m-1]
                jt_scal[m] = (cos + 1.j*sin) * jt_scal[m-1]
                jz_scal[m] = (cos + 1.j*sin) * jz_scal[m-1]

            # Original index of the uppper and lower cell
            ir_cell = int(math.floor( r_cell ))
            iz_cell = int(math.floor( z_cell ))

            # Treat the boundary conditions
            # guard cells in lower r
            if ir_cell < 0:
                ir_cell = 0
            # absorbing in upper r
            if ir_cell > Nr-1:
                ir_cell = Nr-1
            # periodic boundaries in z
            if iz_cell < 0:
                iz_cell += Nz
            if iz_cell > Nz-1:
                iz_cell -= Nz

            # Compute values in local copies and consider boundaries
            ir_flip = int64(math.floor(r_cell)) - 1

            # Declare the local field value for
            # all possible deposition directions,
            # depending on the shape order and per mode for r,t and z.
            J_r_m0_00 = 0.
            J_t_m0_00 = 0.
            J_z_m0_00 = 0.
            J_r_m1_00 = 0. + 0.j
            J_t_m1_00 = 0. + 0.j
            J_z_m1_00 = 0. + 0.j

            J_r_m0_01 = 0.
            J_t_m0_01 = 0.
            J_z_m0_01 = 0.
            J_r_m1_01 = 0. + 0.j
            J_t_m1_01 = 0. + 0.j
            J_z_m1_01 = 0. + 0.j

            J_r_m0_02 = 0.
            J_t_m0_02 = 0.
            J_z_m0_02 = 0.
            J_r_m1_02 = 0. + 0.j
            J_t_m1_02 = 0. + 0.j
            J_z_m1_02 = 0. + 0.j

            J_r_m0_03 = 0.
            J_t_m0_03 = 0.
            J_z_m0_03 = 0.
            J_r_m1_03 = 0. + 0.j
            J_t_m1_03 = 0. + 0.j
            J_z_m1_03 = 0. + 0.j

            J_r_m0_10 = 0.
            J_t_m0_10 = 0.
            J_z_m0_10 = 0.
            J_r_m1_10 = 0. + 0.j
            J_t_m1_10 = 0. + 0.j
            J_z_m1_10 = 0. + 0.j

            J_r_m0_11 = 0.
            J_t_m0_11 = 0.
            J_z_m0_11 = 0.
            J_r_m1_11 = 0. + 0.j
            J_t_m1_11 = 0. + 0.j
            J_z_m1_11 = 0. + 0.j

            J_r_m0_12 = 0.
            J_t_m0_12 = 0.
            J_z_m0_12 = 0.
            J_r_m1_12 = 0. + 0.j
            J_t_m1_12 = 0. + 0.j
            J_z_m1_12 = 0. + 0.j

            J_r_m0_13 = 0.
            J_t_m0_13 = 0.
            J_z_m0_13 = 0.
            J_r_m1_13 = 0. + 0.j
            J_t_m1_13 = 0. + 0.j
            J_z_m1_13 = 0. + 0.j

            J_r_m0_20 = 0.
            J_t_m0_20 = 0.
            J_z_m0_20 = 0.
            J_r_m1_20 = 0. + 0.j
            J_t_m1_20 = 0. + 0.j
            J_z_m1_20 = 0. + 0.j

            J_r_m0_21 = 0.
            J_t_m0_21 = 0.
            J_z_m0_21 = 0.
            J_r_m1_21 = 0. + 0.j
            J_t_m1_21 = 0. + 0.j
            J_z_m1_21 = 0. + 0.j

            J_r_m0_22 = 0.
            J_t_m0_22 = 0.
            J_z_m0_22 = 0.
            J_r_m1_22 = 0. + 0.j
            J_t_m1_22 = 0. + 0.j
            J_z_m1_22 = 0. + 0.j

            J_r_m0_23 = 0.
            J_t_m0_23 = 0.
            J_z_m0_23 = 0.
            J_r_m1_23 = 0. + 0.j
            J_t_m1_23 = 0. + 0.j
            J_z_m1_23 = 0. + 0.j

            J_r_m0_30 = 0.
            J_t_m0_30 = 0.
            J_z_m0_30 = 0.
            J_r_m1_30 = 0. + 0.j
            J_t_m1_30 = 0. + 0.j
            J_z_m1_30 = 0. + 0.j

            J_r_m0_31 = 0.
            J_t_m0_31 = 0.
            J_z_m0_31 = 0.
            J_r_m1_31 = 0. + 0.j
            J_t_m1_31 = 0. + 0.j
            J_z_m1_31 = 0. + 0.j

            J_r_m0_32 = 0.
            J_t_m0_32 = 0.
            J_z_m0_32 = 0.
            J_r_m1_32 = 0. + 0.j
            J_t_m1_32 = 0. + 0.j
            J_z_m1_32 = 0. + 0.j

            J_r_m0_33 = 0.
            J_t_m0_33 = 0.
            J_z_m0_33 = 0.
            J_r_m1_33 = 0. + 0.j
            J_t_m1_33 = 0. + 0.j
            J_z_m1_33 = 0. + 0.j

            if (ir_flip == -2):
                J_r_m0_20 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*jr_scal[0]
                J_r_m1_20 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*jr_scal[1]
                J_r_m0_21 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jr_scal[0]
                J_r_m1_21 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jr_scal[1]
                J_r_m0_22 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jr_scal[0]
                J_r_m1_22 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jr_scal[1]
                J_r_m0_23 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jr_scal[0]
                J_r_m1_23 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jr_scal[1]

                J_r_m0_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jr_scal[0]
                J_r_m1_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jr_scal[1]
                J_r_m0_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jr_scal[0]
                J_r_m1_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jr_scal[1]
                J_r_m0_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jr_scal[0]
                J_r_m1_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jr_scal[1]
                J_r_m0_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jr_scal[0]
                J_r_m1_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jr_scal[1]

                J_r_m0_10 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jr_scal[0]
                J_r_m1_10 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jr_scal[1]
                J_r_m0_11 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jr_scal[0]
                J_r_m1_11 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jr_scal[1]
                J_r_m0_12 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jr_scal[0]
                J_r_m1_12 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jr_scal[1]
                J_r_m0_13 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jr_scal[0]
                J_r_m1_13 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jr_scal[1]

                J_r_m0_20 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jr_scal[0]
                J_r_m1_20 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jr_scal[1]
                J_r_m0_21 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jr_scal[0]
                J_r_m1_21 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jr_scal[1]
                J_r_m0_22 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jr_scal[0]
                J_r_m1_22 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jr_scal[1]
                J_r_m0_23 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jr_scal[0]
                J_r_m1_23 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jr_scal[1]
                J_r_m0_20 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*jr_scal[0]

                J_t_m1_20 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*jt_scal[1]
                J_t_m0_21 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jt_scal[0]
                J_t_m1_21 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jt_scal[1]
                J_t_m0_22 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jt_scal[0]
                J_t_m1_22 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jt_scal[1]
                J_t_m0_23 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jt_scal[0]
                J_t_m1_23 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jt_scal[1]

                J_t_m0_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jt_scal[0]
                J_t_m1_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jt_scal[1]
                J_t_m0_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jt_scal[0]
                J_t_m1_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jt_scal[1]
                J_t_m0_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jt_scal[0]
                J_t_m1_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jt_scal[1]
                J_t_m0_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jt_scal[0]
                J_t_m1_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jt_scal[1]

                J_t_m0_10 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jt_scal[0]
                J_t_m1_10 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jt_scal[1]
                J_t_m0_11 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jt_scal[0]
                J_t_m1_11 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jt_scal[1]
                J_t_m0_12 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jt_scal[0]
                J_t_m1_12 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jt_scal[1]
                J_t_m0_13 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jt_scal[0]
                J_t_m1_13 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jt_scal[1]

                J_t_m0_20 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jt_scal[0]
                J_t_m1_20 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jt_scal[1]
                J_t_m0_21 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jt_scal[0]
                J_t_m1_21 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jt_scal[1]
                J_t_m0_22 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jt_scal[0]
                J_t_m1_22 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jt_scal[1]
                J_t_m0_23 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jt_scal[0]
                J_t_m1_23 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jt_scal[1]

                J_z_m1_20 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*jz_scal[1]
                J_z_m0_21 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jz_scal[0]
                J_z_m1_21 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jz_scal[1]
                J_z_m0_22 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jz_scal[0]
                J_z_m1_22 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jz_scal[1]
                J_z_m0_23 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jz_scal[0]
                J_z_m1_23 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jz_scal[1]

                J_z_m0_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jz_scal[0]
                J_z_m1_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jz_scal[1]
                J_z_m0_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jz_scal[0]
                J_z_m1_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jz_scal[1]
                J_z_m0_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jz_scal[0]
                J_z_m1_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jz_scal[1]
                J_z_m0_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jz_scal[0]
                J_z_m1_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jz_scal[1]

                J_z_m0_10 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jz_scal[0]
                J_z_m1_10 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jz_scal[1]
                J_z_m0_11 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jz_scal[0]
                J_z_m1_11 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jz_scal[1]
                J_z_m0_12 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jz_scal[0]
                J_z_m1_12 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jz_scal[1]
                J_z_m0_13 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jz_scal[0]
                J_z_m1_13 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jz_scal[1]

                J_z_m0_20 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jz_scal[0]
                J_z_m1_20 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jz_scal[1]
                J_z_m0_21 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jz_scal[0]
                J_z_m1_21 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jz_scal[1]
                J_z_m0_22 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jz_scal[0]
                J_z_m1_22 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jz_scal[1]
                J_z_m0_23 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jz_scal[0]
                J_z_m1_23 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jz_scal[1]
            if (ir_flip == -1):
                J_r_m0_10 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*jr_scal[0]
                J_r_m1_10 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*jr_scal[1]
                J_r_m0_11 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jr_scal[0]
                J_r_m1_11 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jr_scal[1]
                J_r_m0_12 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jr_scal[0]
                J_r_m1_12 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jr_scal[1]
                J_r_m0_13 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jr_scal[0]
                J_r_m1_13 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jr_scal[1]

                J_r_m0_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jr_scal[0]
                J_r_m1_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jr_scal[1]
                J_r_m0_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jr_scal[0]
                J_r_m1_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jr_scal[1]
                J_r_m0_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jr_scal[0]
                J_r_m1_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jr_scal[1]
                J_r_m0_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jr_scal[0]
                J_r_m1_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jr_scal[1]

                J_r_m0_20 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jr_scal[0]
                J_r_m1_20 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jr_scal[1]
                J_r_m0_21 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jr_scal[0]
                J_r_m1_21 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jr_scal[1]
                J_r_m0_22 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jr_scal[0]
                J_r_m1_22 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jr_scal[1]
                J_r_m0_23 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jr_scal[0]
                J_r_m1_23 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jr_scal[1]

                J_r_m0_30 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jr_scal[0]
                J_r_m1_30 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jr_scal[1]
                J_r_m0_31 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jr_scal[0]
                J_r_m1_31 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jr_scal[1]
                J_r_m0_32 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jr_scal[0]
                J_r_m1_32 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jr_scal[1]
                J_r_m0_33 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jr_scal[0]
                J_r_m1_33 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jr_scal[1]

                J_t_m0_10 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*jt_scal[0]
                J_t_m1_10 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*jt_scal[1]
                J_t_m0_11 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jt_scal[0]
                J_t_m1_11 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jt_scal[1]
                J_t_m0_12 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jt_scal[0]
                J_t_m1_12 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jt_scal[1]
                J_t_m0_13 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jt_scal[0]
                J_t_m1_13 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jt_scal[1]

                J_t_m0_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jt_scal[0]
                J_t_m1_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jt_scal[1]
                J_t_m0_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jt_scal[0]
                J_t_m1_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jt_scal[1]
                J_t_m0_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jt_scal[0]
                J_t_m1_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jt_scal[1]
                J_t_m0_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jt_scal[0]
                J_t_m1_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jt_scal[1]

                J_t_m0_20 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jt_scal[0]
                J_t_m1_20 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jt_scal[1]
                J_t_m0_21 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jt_scal[0]
                J_t_m1_21 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jt_scal[1]
                J_t_m0_22 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jt_scal[0]
                J_t_m1_22 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jt_scal[1]
                J_t_m0_23 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jt_scal[0]
                J_t_m1_23 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jt_scal[1]

                J_t_m0_30 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jt_scal[0]
                J_t_m1_30 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jt_scal[1]
                J_t_m0_31 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jt_scal[0]
                J_t_m1_31 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jt_scal[1]
                J_t_m0_32 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jt_scal[0]
                J_t_m1_32 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jt_scal[1]
                J_t_m0_33 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jt_scal[0]
                J_t_m1_33 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jt_scal[1]

                J_z_m0_10 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*jz_scal[0]
                J_z_m1_10 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*jz_scal[1]
                J_z_m0_11 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jz_scal[0]
                J_z_m1_11 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jz_scal[1]
                J_z_m0_12 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jz_scal[0]
                J_z_m1_12 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jz_scal[1]
                J_z_m0_13 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jz_scal[0]
                J_z_m1_13 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jz_scal[1]

                J_z_m0_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jz_scal[0]
                J_z_m1_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jz_scal[1]
                J_z_m0_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jz_scal[0]
                J_z_m1_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jz_scal[1]
                J_z_m0_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jz_scal[0]
                J_z_m1_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jz_scal[1]
                J_z_m0_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jz_scal[0]
                J_z_m1_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jz_scal[1]

                J_z_m0_20 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jz_scal[0]
                J_z_m1_20 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jz_scal[1]
                J_z_m0_21 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jz_scal[0]
                J_z_m1_21 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jz_scal[1]
                J_z_m0_22 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jz_scal[0]
                J_z_m1_22 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jz_scal[1]
                J_z_m0_23 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jz_scal[0]
                J_z_m1_23 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jz_scal[1]

                J_z_m0_30 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jz_scal[0]
                J_z_m1_30 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jz_scal[1]
                J_z_m0_31 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jz_scal[0]
                J_z_m1_31 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jz_scal[1]
                J_z_m0_32 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jz_scal[0]
                J_z_m1_32 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jz_scal[1]
                J_z_m0_33 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jz_scal[0]
                J_z_m1_33 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jz_scal[1]
            if (ir_flip >= 0):
                J_r_m0_00 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*jr_scal[0]
                J_r_m1_00 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*jr_scal[1]
                J_r_m0_01 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jr_scal[0]
                J_r_m1_01 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jr_scal[1]
                J_r_m0_02 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jr_scal[0]
                J_r_m1_02 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jr_scal[1]
                J_r_m0_03 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jr_scal[0]
                J_r_m1_03 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jr_scal[1]

                J_r_m0_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jr_scal[0]
                J_r_m1_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jr_scal[1]
                J_r_m0_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jr_scal[0]
                J_r_m1_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jr_scal[1]
                J_r_m0_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jr_scal[0]
                J_r_m1_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jr_scal[1]
                J_r_m0_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jr_scal[0]
                J_r_m1_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jr_scal[1]

                J_r_m0_20 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jr_scal[0]
                J_r_m1_20 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jr_scal[1]
                J_r_m0_21 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jr_scal[0]
                J_r_m1_21 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jr_scal[1]
                J_r_m0_22 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jr_scal[0]
                J_r_m1_22 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jr_scal[1]
                J_r_m0_23 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jr_scal[0]
                J_r_m1_23 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jr_scal[1]

                J_r_m0_30 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jr_scal[0]
                J_r_m1_30 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jr_scal[1]
                J_r_m0_31 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jr_scal[0]
                J_r_m1_31 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jr_scal[1]
                J_r_m0_32 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jr_scal[0]
                J_r_m1_32 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jr_scal[1]
                J_r_m0_33 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jr_scal[0]
                J_r_m1_33 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jr_scal[1]

                J_t_m0_00 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*jt_scal[0]
                J_t_m1_00 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*jt_scal[1]
                J_t_m0_01 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jt_scal[0]
                J_t_m1_01 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jt_scal[1]
                J_t_m0_02 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jt_scal[0]
                J_t_m1_02 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jt_scal[1]
                J_t_m0_03 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jt_scal[0]
                J_t_m1_03 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jt_scal[1]

                J_t_m0_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jt_scal[0]
                J_t_m1_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jt_scal[1]
                J_t_m0_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jt_scal[0]
                J_t_m1_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jt_scal[1]
                J_t_m0_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jt_scal[0]
                J_t_m1_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jt_scal[1]
                J_t_m0_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jt_scal[0]
                J_t_m1_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jt_scal[1]

                J_t_m0_20 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jt_scal[0]
                J_t_m1_20 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jt_scal[1]
                J_t_m0_21 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jt_scal[0]
                J_t_m1_21 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jt_scal[1]
                J_t_m0_22 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jt_scal[0]
                J_t_m1_22 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jt_scal[1]
                J_t_m0_23 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jt_scal[0]
                J_t_m1_23 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jt_scal[1]

                J_t_m0_30 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jt_scal[0]
                J_t_m1_30 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jt_scal[1]
                J_t_m0_31 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jt_scal[0]
                J_t_m1_31 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jt_scal[1]
                J_t_m0_32 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jt_scal[0]
                J_t_m1_32 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jt_scal[1]
                J_t_m0_33 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jt_scal[0]
                J_t_m1_33 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jt_scal[1]

                J_z_m0_00 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*jz_scal[0]
                J_z_m1_00 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 0)*jz_scal[1]
                J_z_m0_01 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jz_scal[0]
                J_z_m1_01 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 1)*jz_scal[1]
                J_z_m0_02 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jz_scal[0]
                J_z_m1_02 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 2)*jz_scal[1]
                J_z_m0_03 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jz_scal[0]
                J_z_m1_03 += Sr_cubic(r_cell, 0)*Sz_cubic(z_cell, 3)*jz_scal[1]

                J_z_m0_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jz_scal[0]
                J_z_m1_10 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 0)*jz_scal[1]
                J_z_m0_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jz_scal[0]
                J_z_m1_11 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 1)*jz_scal[1]
                J_z_m0_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jz_scal[0]
                J_z_m1_12 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 2)*jz_scal[1]
                J_z_m0_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jz_scal[0]
                J_z_m1_13 += Sr_cubic(r_cell, 1)*Sz_cubic(z_cell, 3)*jz_scal[1]

                J_z_m0_20 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jz_scal[0]
                J_z_m1_20 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 0)*jz_scal[1]
                J_z_m0_21 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jz_scal[0]
                J_z_m1_21 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 1)*jz_scal[1]
                J_z_m0_22 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jz_scal[0]
                J_z_m1_22 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 2)*jz_scal[1]
                J_z_m0_23 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jz_scal[0]
                J_z_m1_23 += Sr_cubic(r_cell, 2)*Sz_cubic(z_cell, 3)*jz_scal[1]

                J_z_m0_30 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jz_scal[0]
                J_z_m1_30 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 0)*jz_scal[1]
                J_z_m0_31 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jz_scal[0]
                J_z_m1_31 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 1)*jz_scal[1]
                J_z_m0_32 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jz_scal[0]
                J_z_m1_32 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 2)*jz_scal[1]
                J_z_m0_33 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jz_scal[0]
                J_z_m1_33 += Sr_cubic(r_cell, 3)*Sz_cubic(z_cell, 3)*jz_scal[1]

            # Index Shifting since local copies are centered around
            # the current cell
            srl = 0         # shift r lower
            sru = 0         # shift r upper inner
            sru2 = 0        # shift r upper outer
            szl = 0         # shift z lower
            szu = 0         # shift z upper inner
            szu2 = 0        # shift z upper outer
            if (iz_cell-1) < 0:
                szl += Nz
            if (iz_cell) == (Nz - 1):
                szu -= Nz
                szu2 -= Nz
            if (iz_cell+1) == (Nz - 1):
                szu2 -= Nz
            if (ir_cell) >= (Nr - 1):
                sru = -1
                sru2 = -2
            if (ir_cell+1) == (Nr - 1):
                sru2 = -1
            if (ir_cell-1) < 0:
                srl = 1

            j_r_global[i_thread, 0, iz_cell - 1 + szl+2, ir_cell - 1 + srl+2] += J_r_m0_00
            j_r_global[i_thread, 1, iz_cell - 1 + szl+2, ir_cell - 1 + srl+2] += J_r_m1_00
            j_r_global[i_thread, 0, iz_cell+2, ir_cell - 1 + srl+2] += J_r_m0_01
            j_r_global[i_thread, 1, iz_cell+2, ir_cell - 1 + srl+2] += J_r_m1_01
            j_r_global[i_thread, 0, iz_cell + 1 + szu+2, ir_cell - 1 + srl+2] += J_r_m0_02
            j_r_global[i_thread, 1, iz_cell + 1 + szu+2, ir_cell - 1 + srl+2] += J_r_m1_02
            j_r_global[i_thread, 0, iz_cell + 2 + szu2+2, ir_cell - 1 + srl+2] += J_r_m0_03
            j_r_global[i_thread, 1, iz_cell + 2 + szu2+2, ir_cell - 1 + srl+2] += J_r_m1_03
            j_r_global[i_thread, 0, iz_cell - 1 + szl+2, ir_cell +2] += J_r_m0_10
            j_r_global[i_thread, 1, iz_cell - 1 + szl+2, ir_cell +2] += J_r_m1_10
            j_r_global[i_thread, 0, iz_cell+2, ir_cell+2] += J_r_m0_11
            j_r_global[i_thread, 1, iz_cell+2, ir_cell+2] += J_r_m1_11
            j_r_global[i_thread, 0, iz_cell + 1 + szu+2, ir_cell+2] += J_r_m0_12
            j_r_global[i_thread, 1, iz_cell + 1 + szu+2, ir_cell+2] += J_r_m1_12
            j_r_global[i_thread, 0, iz_cell + 2 + szu2+2, ir_cell+2] += J_r_m0_13
            j_r_global[i_thread, 1, iz_cell + 2 + szu2+2, ir_cell+2] += J_r_m1_13
            j_r_global[i_thread, 0, iz_cell - 1 + szl+2, ir_cell + 1 + sru+2] += J_r_m0_20
            j_r_global[i_thread, 1, iz_cell - 1 + szl+2, ir_cell + 1 + sru+2] += J_r_m1_20
            j_r_global[i_thread, 0, iz_cell+2, ir_cell + 1 + sru+2] += J_r_m0_21
            j_r_global[i_thread, 1, iz_cell+2, ir_cell + 1 + sru+2] += J_r_m1_21
            j_r_global[i_thread, 0, iz_cell + 1 + szu+2, ir_cell + 1 + sru+2] += J_r_m0_22
            j_r_global[i_thread, 1, iz_cell + 1 + szu+2, ir_cell + 1 + sru+2] += J_r_m1_22
            j_r_global[i_thread, 0, iz_cell + 2 + szu2+2, ir_cell + 1 + sru+2] += J_r_m0_23
            j_r_global[i_thread, 1, iz_cell + 2 + szu2+2, ir_cell + 1 + sru+2] += J_r_m1_23
            j_r_global[i_thread, 0, iz_cell - 1 + szl+2, ir_cell + 2 + sru2+2] += J_r_m0_30
            j_r_global[i_thread, 1, iz_cell - 1 + szl+2, ir_cell + 2 + sru2+2] += J_r_m1_30
            j_r_global[i_thread, 0, iz_cell+2, ir_cell + 2 + sru2+2] += J_r_m0_31
            j_r_global[i_thread, 1, iz_cell+2, ir_cell + 2 + sru2+2] += J_r_m1_31
            j_r_global[i_thread, 0, iz_cell + 1 + szu+2, ir_cell + 2 + sru2+2] += J_r_m0_32
            j_r_global[i_thread, 1, iz_cell + 1 + szu+2, ir_cell + 2 + sru2+2] += J_r_m1_32
            j_r_global[i_thread, 0, iz_cell + 2 + szu2+2, ir_cell + 2 + sru2+2] += J_r_m0_33
            j_r_global[i_thread, 1, iz_cell + 2 + szu2+2, ir_cell + 2 + sru2+2] += J_r_m1_33

            j_t_global[i_thread, 0, iz_cell - 1 + szl+2, ir_cell - 1 + srl+2] += J_t_m0_00
            j_t_global[i_thread, 1, iz_cell - 1 + szl+2, ir_cell - 1 + srl+2] += J_t_m1_00
            j_t_global[i_thread, 0, iz_cell+2, ir_cell - 1 + srl+2] += J_t_m0_01
            j_t_global[i_thread, 1, iz_cell+2, ir_cell - 1 + srl+2] += J_t_m1_01
            j_t_global[i_thread, 0, iz_cell + 1 + szu+2, ir_cell - 1 + srl+2] += J_t_m0_02
            j_t_global[i_thread, 1, iz_cell + 1 + szu+2, ir_cell - 1 + srl+2] += J_t_m1_02
            j_t_global[i_thread, 0, iz_cell + 2 + szu2+2, ir_cell - 1 + srl+2] += J_t_m0_03
            j_t_global[i_thread, 1, iz_cell + 2 + szu2+2, ir_cell - 1 + srl+2] += J_t_m1_03
            j_t_global[i_thread, 0, iz_cell - 1 + szl+2, ir_cell +2] += J_t_m0_10
            j_t_global[i_thread, 1, iz_cell - 1 + szl+2, ir_cell +2] += J_t_m1_10
            j_t_global[i_thread, 0, iz_cell+2, ir_cell+2] += J_t_m0_11
            j_t_global[i_thread, 1, iz_cell+2, ir_cell+2] += J_t_m1_11
            j_t_global[i_thread, 0, iz_cell + 1 + szu+2, ir_cell+2] += J_t_m0_12
            j_t_global[i_thread, 1, iz_cell + 1 + szu+2, ir_cell+2] += J_t_m1_12
            j_t_global[i_thread, 0, iz_cell + 2 + szu2+2, ir_cell+2] += J_t_m0_13
            j_t_global[i_thread, 1, iz_cell + 2 + szu2+2, ir_cell+2] += J_t_m1_13
            j_t_global[i_thread, 0, iz_cell - 1 + szl+2, ir_cell + 1 + sru+2] += J_t_m0_20
            j_t_global[i_thread, 1, iz_cell - 1 + szl+2, ir_cell + 1 + sru+2] += J_t_m1_20
            j_t_global[i_thread, 0, iz_cell+2, ir_cell + 1 + sru+2] += J_t_m0_21
            j_t_global[i_thread, 1, iz_cell+2, ir_cell + 1 + sru+2] += J_t_m1_21
            j_t_global[i_thread, 0, iz_cell + 1 + szu+2, ir_cell + 1 + sru+2] += J_t_m0_22
            j_t_global[i_thread, 1, iz_cell + 1 + szu+2, ir_cell + 1 + sru+2] += J_t_m1_22
            j_t_global[i_thread, 0, iz_cell + 2 + szu2+2, ir_cell + 1 + sru+2] += J_t_m0_23
            j_t_global[i_thread, 1, iz_cell + 2 + szu2+2, ir_cell + 1 + sru+2] += J_t_m1_23
            j_t_global[i_thread, 0, iz_cell - 1 + szl+2, ir_cell + 2 + sru2+2] += J_t_m0_30
            j_t_global[i_thread, 1, iz_cell - 1 + szl+2, ir_cell + 2 + sru2+2] += J_t_m1_30
            j_t_global[i_thread, 0, iz_cell+2, ir_cell + 2 + sru2+2] += J_t_m0_31
            j_t_global[i_thread, 1, iz_cell+2, ir_cell + 2 + sru2+2] += J_t_m1_31
            j_t_global[i_thread, 0, iz_cell + 1 + szu+2, ir_cell + 2 + sru2+2] += J_t_m0_32
            j_t_global[i_thread, 1, iz_cell + 1 + szu+2, ir_cell + 2 + sru2+2] += J_t_m1_32
            j_t_global[i_thread, 0, iz_cell + 2 + szu2+2, ir_cell + 2 + sru2+2] += J_t_m0_33
            j_t_global[i_thread, 1, iz_cell + 2 + szu2+2, ir_cell + 2 + sru2+2] += J_t_m1_33

            j_z_global[i_thread, 0, iz_cell - 1 + szl+2, ir_cell - 1 + srl+2] += J_z_m0_00
            j_z_global[i_thread, 1, iz_cell - 1 + szl+2, ir_cell - 1 + srl+2] += J_z_m1_00
            j_z_global[i_thread, 0, iz_cell+2, ir_cell - 1 + srl+2] += J_z_m0_01
            j_z_global[i_thread, 1, iz_cell+2, ir_cell - 1 + srl+2] += J_z_m1_01
            j_z_global[i_thread, 0, iz_cell + 1 + szu+2, ir_cell - 1 + srl+2] += J_z_m0_02
            j_z_global[i_thread, 1, iz_cell + 1 + szu+2, ir_cell - 1 + srl+2] += J_z_m1_02
            j_z_global[i_thread, 0, iz_cell + 2 + szu2+2, ir_cell - 1 + srl+2] += J_z_m0_03
            j_z_global[i_thread, 1, iz_cell + 2 + szu2+2, ir_cell - 1 + srl+2] += J_z_m1_03
            j_z_global[i_thread, 0, iz_cell - 1 + szl+2, ir_cell +2] += J_z_m0_10
            j_z_global[i_thread, 1, iz_cell - 1 + szl+2, ir_cell +2] += J_z_m1_10
            j_z_global[i_thread, 0, iz_cell+2, ir_cell+2] += J_z_m0_11
            j_z_global[i_thread, 1, iz_cell+2, ir_cell+2] += J_z_m1_11
            j_z_global[i_thread, 0, iz_cell + 1 + szu+2, ir_cell+2] += J_z_m0_12
            j_z_global[i_thread, 1, iz_cell + 1 + szu+2, ir_cell+2] += J_z_m1_12
            j_z_global[i_thread, 0, iz_cell + 2 + szu2+2, ir_cell+2] += J_z_m0_13
            j_z_global[i_thread, 1, iz_cell + 2 + szu2+2, ir_cell+2] += J_z_m1_13
            j_z_global[i_thread, 0, iz_cell - 1 + szl+2, ir_cell + 1 + sru+2] += J_z_m0_20
            j_z_global[i_thread, 1, iz_cell - 1 + szl+2, ir_cell + 1 + sru+2] += J_z_m1_20
            j_z_global[i_thread, 0, iz_cell+2, ir_cell + 1 + sru+2] += J_z_m0_21
            j_z_global[i_thread, 1, iz_cell+2, ir_cell + 1 + sru+2] += J_z_m1_21
            j_z_global[i_thread, 0, iz_cell + 1 + szu+2, ir_cell + 1 + sru+2] += J_z_m0_22
            j_z_global[i_thread, 1, iz_cell + 1 + szu+2, ir_cell + 1 + sru+2] += J_z_m1_22
            j_z_global[i_thread, 0, iz_cell + 2 + szu2+2, ir_cell + 1 + sru+2] += J_z_m0_23
            j_z_global[i_thread, 1, iz_cell + 2 + szu2+2, ir_cell + 1 + sru+2] += J_z_m1_23
            j_z_global[i_thread, 0, iz_cell - 1 + szl+2, ir_cell + 2 + sru2+2] += J_z_m0_30
            j_z_global[i_thread, 1, iz_cell - 1 + szl+2, ir_cell + 2 + sru2+2] += J_z_m1_30
            j_z_global[i_thread, 0, iz_cell+2, ir_cell + 2 + sru2+2] += J_z_m0_31
            j_z_global[i_thread, 1, iz_cell+2, ir_cell + 2 + sru2+2] += J_z_m1_31
            j_z_global[i_thread, 0, iz_cell + 1 + szu+2, ir_cell + 2 + sru2+2] += J_z_m0_32
            j_z_global[i_thread, 1, iz_cell + 1 + szu+2, ir_cell + 2 + sru2+2] += J_z_m1_32
            j_z_global[i_thread, 0, iz_cell + 2 + szu2+2, ir_cell + 2 + sru2+2] += J_z_m0_33
            j_z_global[i_thread, 1, iz_cell + 2 + szu2+2, ir_cell + 2 + sru2+2] += J_z_m1_33

    return

# -----------------------------------------------------------------------
# Parallel reduction of the global arrays for threads into a single array
# -----------------------------------------------------------------------

@numba.njit
def sum_reduce_2d_array( global_array, reduced_array, m ):
    """
    Sum the array `global_array` along its first axis and
    add it into `reduced_array`, and fold the deposition guard cells of
    global_array into the regular cells of reduced_array.

    Parameters:
    -----------
    global_array: 4darray of complexs
       Field array of shape (nthreads, Nm, 2+Nz+2, 2+Nr+2)
       where the additional 2's in z and r correspond to deposition guard cells
       that were used during the threaded deposition kernel.

    reduced array: 2darray of complex
      Field array of shape (Nz, Nr)

    m: int
       The azimuthal mode for which the reduction should be performed
    """
    # Extract size of each dimension
    Nreduce = global_array.shape[0]
    Nz, Nr = reduced_array.shape

    # Parallel loop over z in global_array (includes deposition guard cells)
    for iz_global in range(Nz+4):

        # Get index inside reduced_array
        iz = iz_global - 2
        if iz < 0:
            iz = iz + Nz
        elif iz >= Nz:
            iz = iz - Nz

        # Loop over the reduction dimension (slow dimension)
        for it in range( Nreduce ):

            # First fold the low-radius deposition guard cells in
            reduced_array[iz, 1] += global_array[it, m, iz_global, 0]
            reduced_array[iz, 0] += global_array[it, m, iz_global, 1]
            # Then loop over regular cells
            for ir in range( Nr ):
                reduced_array[iz, ir] +=  global_array[it, m, iz_global, ir+2]
            # Finally fold the high-radius guard cells in
            reduced_array[iz, Nr-1] += global_array[it, m, iz_global, Nr+2]
            reduced_array[iz, Nr-1] += global_array[it, m, iz_global, Nr+3]
