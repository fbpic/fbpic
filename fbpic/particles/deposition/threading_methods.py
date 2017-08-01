# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the deposition methods for rho and J for linear and cubic
order shapes on the CPU with threading.
"""
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
def z_shape_linear(cell_position, index):
    iz = int64(math.floor(cell_position))
    if index == 0:
        return iz+1.-cell_position
    if index == 1:
        return cell_position - iz

@numba.njit
def r_shape_linear(cell_position, index):
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
def z_shape_cubic(cell_position, index):
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
def r_shape_cubic(cell_position, index):
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
                           rho_m0_global, rho_m1_global,
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

    rho_m0_global, rho_m1_global : 3darrays of complexs (nthread, Nz, Nr)
        The global helper arrays to store the thread local charge densities
        on the interpolation grid for mode 0 and 1.
        (is modified by this function)

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
            exptheta_m0 = 1.
            exptheta_m1 = cos + 1.j*sin

            # Positions of the particles, in the cell unit
            r_cell = invdr*(rj - rmin) - 0.5
            z_cell = invdz*(zj - zmin) - 0.5

            # Calculate rho
            # --------------------------------------------
            # Mode 0
            R_m0_scal = wj * exptheta_m0
            # Mode 1
            R_m1_scal = wj * exptheta_m1

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

            # Declare local field array
            R_m0_00 = 0.
            R_m0_01 = 0.
            R_m0_10 = 0.
            R_m0_11 = 0.

            R_m1_00 = 0. + 0.j
            R_m1_01 = 0. + 0.j
            R_m1_10 = 0. + 0.j
            R_m1_11 = 0. + 0.j

            R_m0_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * R_m0_scal
            R_m0_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * R_m0_scal
            R_m1_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * R_m1_scal
            R_m1_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * R_m1_scal

            if ir_flip == -1:
                R_m0_00 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * R_m0_scal
                R_m0_01 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * R_m0_scal
                R_m1_00 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * R_m1_scal
                R_m1_01 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * R_m1_scal
            else:
                R_m0_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * R_m0_scal
                R_m0_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * R_m0_scal
                R_m1_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * R_m1_scal
                R_m1_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * R_m1_scal

            # Cell shifts for the simulation boundaries
            shift_r = 0
            shift_z = 0
            if ir_cell+1 > (Nr-1):
                shift_r = -1
            if iz_cell+1 > Nz-1:
                shift_z -= Nz

            # Write ptcl fields to thread-local part of global deposition array
            rho_m0_global[i_thread, iz_cell, ir_cell] += R_m0_00
            rho_m1_global[i_thread, iz_cell, ir_cell] += R_m1_00

            rho_m0_global[i_thread,iz_cell+1 + shift_z, ir_cell] += R_m0_01
            rho_m1_global[i_thread,iz_cell+1 + shift_z, ir_cell] += R_m1_01

            rho_m0_global[i_thread,iz_cell, ir_cell+1 + shift_r] += R_m0_10
            rho_m1_global[i_thread,iz_cell, ir_cell+1 + shift_r] += R_m1_10

            rho_m0_global[i_thread,iz_cell+1 + shift_z, ir_cell+1 + shift_r] += R_m0_11
            rho_m1_global[i_thread,iz_cell+1 + shift_z, ir_cell+1 + shift_r] += R_m1_11

    return

# -------------------------------
# Field deposition - linear - J
# -------------------------------

@njit_parallel
def deposit_J_numba_linear(x, y, z, w, q,
                         ux, uy, uz, inv_gamma,
                         invdz, zmin, Nz,
                         invdr, rmin, Nr,
                         j_r_m0_global, j_r_m1_global,
                         j_t_m0_global, j_t_m1_global,
                         j_z_m0_global, j_z_m1_global,
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

    j_x_m0_global, j_x_m1_global : 3darrays of complexs (nthread, Nz, Nr)
        The global helper arrays to store the thread local current component
        in each direction (r, t, z) on the interpolation grid for mode 0 and 1.
        (is modified by this function)

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
            exptheta_m0 = 1.
            exptheta_m1 = cos + 1.j*sin

            # Get weights for the deposition
            # --------------------------------------------
            # Positions of the particles, in the cell unit
            r_cell = invdr*(rj - rmin) - 0.5
            z_cell = invdz*(zj - zmin) - 0.5

            # Calculate the currents
            # --------------------------------------------
            # Mode 0
            J_r_m0_scal = wj * c * inv_gammaj*(cos*uxj + sin*uyj) * exptheta_m0
            J_t_m0_scal = wj * c * inv_gammaj*(cos*uyj - sin*uxj) * exptheta_m0
            J_z_m0_scal = wj * c * inv_gammaj*uzj * exptheta_m0
            # Mode 1
            J_r_m1_scal = wj * c * inv_gammaj*(cos*uxj + sin*uyj) * exptheta_m1
            J_t_m1_scal = wj * c * inv_gammaj*(cos*uyj - sin*uxj) * exptheta_m1
            J_z_m1_scal = wj * c * inv_gammaj*uzj * exptheta_m1

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

            J_r_m0_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_r_m0_scal
            J_t_m0_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_t_m0_scal
            J_z_m0_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_z_m0_scal
            J_r_m0_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_r_m0_scal
            J_t_m0_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_t_m0_scal
            J_z_m0_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_z_m0_scal
            J_r_m1_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_r_m1_scal
            J_t_m1_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_t_m1_scal
            J_z_m1_00 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_z_m1_scal
            J_r_m1_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_r_m1_scal
            J_t_m1_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_t_m1_scal
            J_z_m1_01 += r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_z_m1_scal

            # Take into account lower r flips
            if ir_flip == -1:
                J_r_m0_00 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_r_m0_scal
                J_t_m0_00 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_t_m0_scal
                J_z_m0_00 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_z_m0_scal
                J_r_m0_01 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_r_m0_scal
                J_t_m0_01 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_t_m0_scal
                J_z_m0_01 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_z_m0_scal
                J_r_m1_00 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_r_m1_scal
                J_t_m1_00 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_t_m1_scal
                J_z_m1_00 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_z_m1_scal
                J_r_m1_01 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_r_m1_scal
                J_t_m1_01 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_t_m1_scal
                J_z_m1_01 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_z_m1_scal
            else:
                J_r_m0_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_r_m0_scal
                J_t_m0_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_t_m0_scal
                J_z_m0_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_z_m0_scal
                J_r_m0_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_r_m0_scal
                J_t_m0_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_t_m0_scal
                J_z_m0_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_z_m0_scal
                J_r_m1_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_r_m1_scal
                J_t_m1_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_t_m1_scal
                J_z_m1_10 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_z_m1_scal
                J_r_m1_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_r_m1_scal
                J_t_m1_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_t_m1_scal
                J_z_m1_11 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_z_m1_scal

            # Cell shifts for the simulation boundaries
            shift_r = 0
            shift_z = 0
            if (ir_cell+1) > (Nr-1):
                shift_r = -1
            if (iz_cell+1) > Nz-1:
                shift_z -= Nz

            # Write ptcl fields to thread-local part of global deposition array
            j_r_m0_global[i_thread,iz_cell, ir_cell] += J_r_m0_00
            j_r_m1_global[i_thread,iz_cell, ir_cell] += J_r_m1_00

            j_r_m0_global[i_thread,iz_cell+1 + shift_z, ir_cell] += J_r_m0_01
            j_r_m1_global[i_thread,iz_cell+1 + shift_z, ir_cell] += J_r_m1_01

            j_r_m0_global[i_thread,iz_cell, ir_cell+1 + shift_r] += J_r_m0_10
            j_r_m1_global[i_thread,iz_cell, ir_cell+1 + shift_r] += J_r_m1_10

            j_r_m0_global[i_thread,iz_cell+1 + shift_z, ir_cell+1 + shift_r] += J_r_m0_11
            j_r_m1_global[i_thread,iz_cell+1 + shift_z, ir_cell+1 + shift_r] += J_r_m1_11

            j_t_m0_global[i_thread,iz_cell, ir_cell] += J_t_m0_00
            j_t_m1_global[i_thread,iz_cell, ir_cell] += J_t_m1_00

            j_t_m0_global[i_thread,iz_cell+1 + shift_z, ir_cell] += J_t_m0_01
            j_t_m1_global[i_thread,iz_cell+1 + shift_z, ir_cell] += J_t_m1_01

            j_t_m0_global[i_thread,iz_cell, ir_cell+1 + shift_r] += J_t_m0_10
            j_t_m1_global[i_thread,iz_cell, ir_cell+1 + shift_r] += J_t_m1_10

            j_t_m0_global[i_thread,iz_cell+1 + shift_z, ir_cell+1 + shift_r] += J_t_m0_11
            j_t_m1_global[i_thread,iz_cell+1 + shift_z, ir_cell+1 + shift_r] += J_t_m1_11

            j_z_m0_global[i_thread,iz_cell, ir_cell] += J_z_m0_00
            j_z_m1_global[i_thread,iz_cell, ir_cell] += J_z_m1_00

            j_z_m0_global[i_thread,iz_cell+1 + shift_z, ir_cell] += J_z_m0_01
            j_z_m1_global[i_thread,iz_cell+1 + shift_z, ir_cell] += J_z_m1_01

            j_z_m0_global[i_thread,iz_cell, ir_cell+1 + shift_r] += J_z_m0_10
            j_z_m1_global[i_thread,iz_cell, ir_cell+1 + shift_r] += J_z_m1_10

            j_z_m0_global[i_thread,iz_cell+1 + shift_z, ir_cell+1 + shift_r] += J_z_m0_11
            j_z_m1_global[i_thread,iz_cell+1 + shift_z, ir_cell+1 + shift_r] += J_z_m1_11

    return


# -------------------------------
# Field deposition - cubic - rho
# -------------------------------

@njit_parallel
def deposit_rho_numba_cubic(x, y, z, w, q,
                          invdz, zmin, Nz,
                          invdr, rmin, Nr,
                          rho_m0_global, rho_m1_global,
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

    rho_m0_global, rho_m1_global : 3darrays of complexs (nthread, Nz, Nr)
        The global helper arrays to store the thread local charge densities
        on the interpolation grid for mode 0 and 1.
        (is modified by this function)

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
            exptheta_m0 = 1.
            exptheta_m1 = cos + 1.j*sin

            # Positions of the particles, in the cell unit
            r_cell = invdr*(rj - rmin) - 0.5
            z_cell = invdz*(zj - zmin) - 0.5

            # Calculate rho
            # --------------------------------------------
            # Mode 0
            R_m0_scal = wj * exptheta_m0
            # Mode 1
            R_m1_scal = wj * exptheta_m1

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
            ir_flip = int( math.floor(r_cell) ) - 1

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

            if (ir_flip == -2):
                R_m0_20 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*R_m0_scal
                R_m1_20 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*R_m1_scal
                R_m0_21 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*R_m0_scal
                R_m1_21 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*R_m1_scal
                R_m0_22 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*R_m0_scal
                R_m1_22 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*R_m1_scal
                R_m0_23 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*R_m0_scal
                R_m1_23 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*R_m1_scal

                R_m0_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*R_m0_scal
                R_m1_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*R_m1_scal
                R_m0_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*R_m0_scal
                R_m1_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*R_m1_scal
                R_m0_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*R_m0_scal
                R_m1_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*R_m1_scal
                R_m0_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*R_m0_scal
                R_m1_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*R_m1_scal

                R_m0_10 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*R_m0_scal
                R_m1_10 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*R_m1_scal
                R_m0_11 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*R_m0_scal
                R_m1_11 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*R_m1_scal
                R_m0_12 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*R_m0_scal
                R_m1_12 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*R_m1_scal
                R_m0_13 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*R_m0_scal
                R_m1_13 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*R_m1_scal

                R_m0_20 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*R_m0_scal
                R_m1_20 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*R_m1_scal
                R_m0_21 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*R_m0_scal
                R_m1_21 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*R_m1_scal
                R_m0_22 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*R_m0_scal
                R_m1_22 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*R_m1_scal
                R_m0_23 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*R_m0_scal
                R_m1_23 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*R_m1_scal

            if (ir_flip == -1):
                R_m0_10 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*R_m0_scal
                R_m1_10 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*R_m1_scal
                R_m0_11 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*R_m0_scal
                R_m1_11 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*R_m1_scal
                R_m0_12 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*R_m0_scal
                R_m1_12 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*R_m1_scal
                R_m0_13 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*R_m0_scal
                R_m1_13 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*R_m1_scal

                R_m0_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*R_m0_scal
                R_m1_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*R_m1_scal
                R_m0_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*R_m0_scal
                R_m1_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*R_m1_scal
                R_m0_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*R_m0_scal
                R_m1_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*R_m1_scal
                R_m0_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*R_m0_scal
                R_m1_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*R_m1_scal

                R_m0_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*R_m0_scal
                R_m1_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*R_m1_scal
                R_m0_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*R_m0_scal
                R_m1_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*R_m1_scal
                R_m0_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*R_m0_scal
                R_m1_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*R_m1_scal
                R_m0_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*R_m0_scal
                R_m1_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*R_m1_scal

                R_m0_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*R_m0_scal
                R_m1_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*R_m1_scal
                R_m0_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*R_m0_scal
                R_m1_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*R_m1_scal
                R_m0_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*R_m0_scal
                R_m1_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*R_m1_scal
                R_m0_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*R_m0_scal
                R_m1_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*R_m1_scal
            if (ir_flip >= 0):
                R_m0_00 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*R_m0_scal
                R_m1_00 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*R_m1_scal
                R_m0_01 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*R_m0_scal
                R_m1_01 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*R_m1_scal
                R_m0_02 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*R_m0_scal
                R_m1_02 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*R_m1_scal
                R_m0_03 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*R_m0_scal
                R_m1_03 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*R_m1_scal

                R_m0_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*R_m0_scal
                R_m1_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*R_m1_scal
                R_m0_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*R_m0_scal
                R_m1_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*R_m1_scal
                R_m0_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*R_m0_scal
                R_m1_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*R_m1_scal
                R_m0_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*R_m0_scal
                R_m1_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*R_m1_scal

                R_m0_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*R_m0_scal
                R_m1_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*R_m1_scal
                R_m0_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*R_m0_scal
                R_m1_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*R_m1_scal
                R_m0_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*R_m0_scal
                R_m1_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*R_m1_scal
                R_m0_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*R_m0_scal
                R_m1_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*R_m1_scal

                R_m0_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*R_m0_scal
                R_m1_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*R_m1_scal
                R_m0_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*R_m0_scal
                R_m1_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*R_m1_scal
                R_m0_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*R_m0_scal
                R_m1_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*R_m1_scal
                R_m0_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*R_m0_scal
                R_m1_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*R_m1_scal

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

            # Write ptcl fields to thread-local part of global deposition array
            rho_m0_global[i_thread, iz_cell - 1 + szl, ir_cell - 1 + srl] += R_m0_00
            rho_m1_global[i_thread, iz_cell - 1 + szl, ir_cell - 1 + srl] += R_m1_00
            rho_m0_global[i_thread, iz_cell, ir_cell - 1 + srl] += R_m0_01
            rho_m1_global[i_thread, iz_cell, ir_cell - 1 + srl] += R_m1_01
            rho_m0_global[i_thread, iz_cell + 1 + szu, ir_cell - 1 + srl] += R_m0_02
            rho_m1_global[i_thread, iz_cell + 1 + szu, ir_cell - 1 + srl] += R_m1_02
            rho_m0_global[i_thread, iz_cell + 2 + szu2, ir_cell - 1 + srl] += R_m0_03
            rho_m1_global[i_thread, iz_cell + 2 + szu2, ir_cell - 1 + srl] += R_m1_03
            rho_m0_global[i_thread, iz_cell - 1 + szl, ir_cell] += R_m0_10
            rho_m1_global[i_thread, iz_cell - 1 + szl, ir_cell] += R_m1_10
            rho_m0_global[i_thread, iz_cell, ir_cell] += R_m0_11
            rho_m1_global[i_thread, iz_cell, ir_cell] += R_m1_11
            rho_m0_global[i_thread, iz_cell + 1 + szu, ir_cell] += R_m0_12
            rho_m1_global[i_thread, iz_cell + 1 + szu, ir_cell] += R_m1_12
            rho_m0_global[i_thread, iz_cell + 2 + szu2, ir_cell] += R_m0_13
            rho_m1_global[i_thread, iz_cell + 2 + szu2, ir_cell] += R_m1_13
            rho_m0_global[i_thread, iz_cell - 1 + szl, ir_cell + 1 + sru] += R_m0_20
            rho_m1_global[i_thread, iz_cell - 1 + szl, ir_cell + 1 + sru] += R_m1_20
            rho_m0_global[i_thread, iz_cell, ir_cell + 1 + sru] += R_m0_21
            rho_m1_global[i_thread, iz_cell, ir_cell + 1 + sru] += R_m1_21
            rho_m0_global[i_thread, iz_cell + 1 + szu, ir_cell + 1 + sru] += R_m0_22
            rho_m1_global[i_thread, iz_cell + 1 + szu, ir_cell + 1 + sru] += R_m1_22
            rho_m0_global[i_thread, iz_cell + 2 + szu2, ir_cell + 1 + sru] += R_m0_23
            rho_m1_global[i_thread, iz_cell + 2 + szu2, ir_cell + 1 + sru] += R_m1_23
            rho_m0_global[i_thread, iz_cell - 1 + szl, ir_cell + 2 + sru2] += R_m0_30
            rho_m1_global[i_thread, iz_cell - 1 + szl, ir_cell + 2 + sru2] += R_m1_30
            rho_m0_global[i_thread, iz_cell, ir_cell + 2 + sru2] += R_m0_31
            rho_m1_global[i_thread, iz_cell, ir_cell + 2 + sru2] += R_m1_31
            rho_m0_global[i_thread, iz_cell + 1 + szu, ir_cell + 2 + sru2] += R_m0_32
            rho_m1_global[i_thread, iz_cell + 1 + szu, ir_cell + 2 + sru2] += R_m1_32
            rho_m0_global[i_thread, iz_cell + 2 + szu2, ir_cell + 2 + sru2] += R_m0_33
            rho_m1_global[i_thread, iz_cell + 2 + szu2, ir_cell + 2 + sru2] += R_m1_33

    return

# -------------------------------
# Field deposition - cubic - J
# -------------------------------

@njit_parallel
def deposit_J_numba_cubic(x, y, z, w, q,
                        ux, uy, uz, inv_gamma,
                        invdz, zmin, Nz,
                        invdr, rmin, Nr,
                        j_r_m0_global, j_r_m1_global,
                        j_t_m0_global, j_t_m1_global,
                        j_z_m0_global, j_z_m1_global,
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

    j_x_m0_global, j_x_m1_global : 3darrays of complexs (nthread, Nz, Nr)
        The global helper arrays to store the thread local current component
        in each direction (r, t, z) on the interpolation grid for mode 0 and 1.
        (is modified by this function)

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
            exptheta_m0 = 1.
            exptheta_m1 = cos + 1.j*sin

            # Get weights for the deposition
            # --------------------------------------------
            # Positions of the particles, in the cell unit
            r_cell = invdr*(rj - rmin) - 0.5
            z_cell = invdz*(zj - zmin) - 0.5

            # Calculate the currents
            # --------------------------------------------
            # Mode 0
            J_r_m0_scal = wj * c * inv_gammaj*(cos*uxj + sin*uyj) * exptheta_m0
            J_t_m0_scal = wj * c * inv_gammaj*(cos*uyj - sin*uxj) * exptheta_m0
            J_z_m0_scal = wj * c * inv_gammaj*uzj * exptheta_m0
            # Mode 1
            J_r_m1_scal = wj * c * inv_gammaj*(cos*uxj + sin*uyj) * exptheta_m1
            J_t_m1_scal = wj * c * inv_gammaj*(cos*uyj - sin*uxj) * exptheta_m1
            J_z_m1_scal = wj * c * inv_gammaj*uzj * exptheta_m1

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
                J_r_m0_20 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_r_m0_scal
                J_r_m1_20 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_r_m1_scal
                J_r_m0_21 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_r_m0_scal
                J_r_m1_21 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_r_m1_scal
                J_r_m0_22 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_r_m0_scal
                J_r_m1_22 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_r_m1_scal
                J_r_m0_23 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_r_m0_scal
                J_r_m1_23 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_r_m1_scal

                J_r_m0_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_r_m0_scal
                J_r_m1_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_r_m1_scal
                J_r_m0_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_r_m0_scal
                J_r_m1_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_r_m1_scal
                J_r_m0_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_r_m0_scal
                J_r_m1_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_r_m1_scal
                J_r_m0_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_r_m0_scal
                J_r_m1_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_r_m1_scal

                J_r_m0_10 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_r_m0_scal
                J_r_m1_10 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_r_m1_scal
                J_r_m0_11 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_r_m0_scal
                J_r_m1_11 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_r_m1_scal
                J_r_m0_12 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_r_m0_scal
                J_r_m1_12 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_r_m1_scal
                J_r_m0_13 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_r_m0_scal
                J_r_m1_13 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_r_m1_scal

                J_r_m0_20 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_r_m0_scal
                J_r_m1_20 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_r_m1_scal
                J_r_m0_21 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_r_m0_scal
                J_r_m1_21 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_r_m1_scal
                J_r_m0_22 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_r_m0_scal
                J_r_m1_22 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_r_m1_scal
                J_r_m0_23 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_r_m0_scal
                J_r_m1_23 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_r_m1_scal
                J_r_m0_20 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_r_m0_scal

                J_t_m1_20 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_t_m1_scal
                J_t_m0_21 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_t_m0_scal
                J_t_m1_21 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_t_m1_scal
                J_t_m0_22 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_t_m0_scal
                J_t_m1_22 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_t_m1_scal
                J_t_m0_23 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_t_m0_scal
                J_t_m1_23 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_t_m1_scal

                J_t_m0_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_t_m0_scal
                J_t_m1_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_t_m1_scal
                J_t_m0_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_t_m0_scal
                J_t_m1_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_t_m1_scal
                J_t_m0_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_t_m0_scal
                J_t_m1_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_t_m1_scal
                J_t_m0_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_t_m0_scal
                J_t_m1_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_t_m1_scal

                J_t_m0_10 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_t_m0_scal
                J_t_m1_10 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_t_m1_scal
                J_t_m0_11 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_t_m0_scal
                J_t_m1_11 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_t_m1_scal
                J_t_m0_12 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_t_m0_scal
                J_t_m1_12 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_t_m1_scal
                J_t_m0_13 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_t_m0_scal
                J_t_m1_13 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_t_m1_scal

                J_t_m0_20 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_t_m0_scal
                J_t_m1_20 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_t_m1_scal
                J_t_m0_21 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_t_m0_scal
                J_t_m1_21 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_t_m1_scal
                J_t_m0_22 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_t_m0_scal
                J_t_m1_22 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_t_m1_scal
                J_t_m0_23 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_t_m0_scal
                J_t_m1_23 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_t_m1_scal

                J_z_m1_20 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_z_m1_scal
                J_z_m0_21 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_z_m0_scal
                J_z_m1_21 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_z_m1_scal
                J_z_m0_22 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_z_m0_scal
                J_z_m1_22 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_z_m1_scal
                J_z_m0_23 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_z_m0_scal
                J_z_m1_23 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_z_m1_scal

                J_z_m0_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_z_m0_scal
                J_z_m1_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_z_m1_scal
                J_z_m0_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_z_m0_scal
                J_z_m1_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_z_m1_scal
                J_z_m0_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_z_m0_scal
                J_z_m1_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_z_m1_scal
                J_z_m0_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_z_m0_scal
                J_z_m1_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_z_m1_scal

                J_z_m0_10 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_z_m0_scal
                J_z_m1_10 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_z_m1_scal
                J_z_m0_11 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_z_m0_scal
                J_z_m1_11 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_z_m1_scal
                J_z_m0_12 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_z_m0_scal
                J_z_m1_12 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_z_m1_scal
                J_z_m0_13 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_z_m0_scal
                J_z_m1_13 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_z_m1_scal

                J_z_m0_20 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_z_m0_scal
                J_z_m1_20 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_z_m1_scal
                J_z_m0_21 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_z_m0_scal
                J_z_m1_21 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_z_m1_scal
                J_z_m0_22 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_z_m0_scal
                J_z_m1_22 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_z_m1_scal
                J_z_m0_23 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_z_m0_scal
                J_z_m1_23 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_z_m1_scal
            if (ir_flip == -1):
                J_r_m0_10 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_r_m0_scal
                J_r_m1_10 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_r_m1_scal
                J_r_m0_11 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_r_m0_scal
                J_r_m1_11 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_r_m1_scal
                J_r_m0_12 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_r_m0_scal
                J_r_m1_12 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_r_m1_scal
                J_r_m0_13 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_r_m0_scal
                J_r_m1_13 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_r_m1_scal

                J_r_m0_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_r_m0_scal
                J_r_m1_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_r_m1_scal
                J_r_m0_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_r_m0_scal
                J_r_m1_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_r_m1_scal
                J_r_m0_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_r_m0_scal
                J_r_m1_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_r_m1_scal
                J_r_m0_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_r_m0_scal
                J_r_m1_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_r_m1_scal

                J_r_m0_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_r_m0_scal
                J_r_m1_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_r_m1_scal
                J_r_m0_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_r_m0_scal
                J_r_m1_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_r_m1_scal
                J_r_m0_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_r_m0_scal
                J_r_m1_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_r_m1_scal
                J_r_m0_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_r_m0_scal
                J_r_m1_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_r_m1_scal

                J_r_m0_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_r_m0_scal
                J_r_m1_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_r_m1_scal
                J_r_m0_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_r_m0_scal
                J_r_m1_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_r_m1_scal
                J_r_m0_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_r_m0_scal
                J_r_m1_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_r_m1_scal
                J_r_m0_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_r_m0_scal
                J_r_m1_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_r_m1_scal

                J_t_m0_10 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_t_m0_scal
                J_t_m1_10 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_t_m1_scal
                J_t_m0_11 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_t_m0_scal
                J_t_m1_11 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_t_m1_scal
                J_t_m0_12 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_t_m0_scal
                J_t_m1_12 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_t_m1_scal
                J_t_m0_13 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_t_m0_scal
                J_t_m1_13 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_t_m1_scal

                J_t_m0_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_t_m0_scal
                J_t_m1_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_t_m1_scal
                J_t_m0_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_t_m0_scal
                J_t_m1_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_t_m1_scal
                J_t_m0_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_t_m0_scal
                J_t_m1_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_t_m1_scal
                J_t_m0_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_t_m0_scal
                J_t_m1_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_t_m1_scal

                J_t_m0_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_t_m0_scal
                J_t_m1_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_t_m1_scal
                J_t_m0_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_t_m0_scal
                J_t_m1_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_t_m1_scal
                J_t_m0_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_t_m0_scal
                J_t_m1_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_t_m1_scal
                J_t_m0_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_t_m0_scal
                J_t_m1_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_t_m1_scal

                J_t_m0_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_t_m0_scal
                J_t_m1_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_t_m1_scal
                J_t_m0_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_t_m0_scal
                J_t_m1_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_t_m1_scal
                J_t_m0_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_t_m0_scal
                J_t_m1_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_t_m1_scal
                J_t_m0_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_t_m0_scal
                J_t_m1_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_t_m1_scal

                J_z_m0_10 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_z_m0_scal
                J_z_m1_10 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_z_m1_scal
                J_z_m0_11 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_z_m0_scal
                J_z_m1_11 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_z_m1_scal
                J_z_m0_12 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_z_m0_scal
                J_z_m1_12 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_z_m1_scal
                J_z_m0_13 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_z_m0_scal
                J_z_m1_13 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_z_m1_scal

                J_z_m0_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_z_m0_scal
                J_z_m1_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_z_m1_scal
                J_z_m0_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_z_m0_scal
                J_z_m1_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_z_m1_scal
                J_z_m0_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_z_m0_scal
                J_z_m1_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_z_m1_scal
                J_z_m0_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_z_m0_scal
                J_z_m1_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_z_m1_scal

                J_z_m0_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_z_m0_scal
                J_z_m1_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_z_m1_scal
                J_z_m0_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_z_m0_scal
                J_z_m1_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_z_m1_scal
                J_z_m0_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_z_m0_scal
                J_z_m1_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_z_m1_scal
                J_z_m0_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_z_m0_scal
                J_z_m1_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_z_m1_scal

                J_z_m0_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_z_m0_scal
                J_z_m1_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_z_m1_scal
                J_z_m0_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_z_m0_scal
                J_z_m1_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_z_m1_scal
                J_z_m0_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_z_m0_scal
                J_z_m1_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_z_m1_scal
                J_z_m0_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_z_m0_scal
                J_z_m1_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_z_m1_scal
            if (ir_flip >= 0):
                J_r_m0_00 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_r_m0_scal
                J_r_m1_00 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_r_m1_scal
                J_r_m0_01 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_r_m0_scal
                J_r_m1_01 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_r_m1_scal
                J_r_m0_02 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_r_m0_scal
                J_r_m1_02 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_r_m1_scal
                J_r_m0_03 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_r_m0_scal
                J_r_m1_03 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_r_m1_scal

                J_r_m0_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_r_m0_scal
                J_r_m1_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_r_m1_scal
                J_r_m0_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_r_m0_scal
                J_r_m1_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_r_m1_scal
                J_r_m0_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_r_m0_scal
                J_r_m1_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_r_m1_scal
                J_r_m0_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_r_m0_scal
                J_r_m1_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_r_m1_scal

                J_r_m0_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_r_m0_scal
                J_r_m1_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_r_m1_scal
                J_r_m0_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_r_m0_scal
                J_r_m1_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_r_m1_scal
                J_r_m0_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_r_m0_scal
                J_r_m1_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_r_m1_scal
                J_r_m0_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_r_m0_scal
                J_r_m1_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_r_m1_scal

                J_r_m0_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_r_m0_scal
                J_r_m1_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_r_m1_scal
                J_r_m0_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_r_m0_scal
                J_r_m1_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_r_m1_scal
                J_r_m0_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_r_m0_scal
                J_r_m1_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_r_m1_scal
                J_r_m0_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_r_m0_scal
                J_r_m1_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_r_m1_scal

                J_t_m0_00 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_t_m0_scal
                J_t_m1_00 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_t_m1_scal
                J_t_m0_01 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_t_m0_scal
                J_t_m1_01 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_t_m1_scal
                J_t_m0_02 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_t_m0_scal
                J_t_m1_02 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_t_m1_scal
                J_t_m0_03 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_t_m0_scal
                J_t_m1_03 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_t_m1_scal

                J_t_m0_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_t_m0_scal
                J_t_m1_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_t_m1_scal
                J_t_m0_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_t_m0_scal
                J_t_m1_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_t_m1_scal
                J_t_m0_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_t_m0_scal
                J_t_m1_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_t_m1_scal
                J_t_m0_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_t_m0_scal
                J_t_m1_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_t_m1_scal

                J_t_m0_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_t_m0_scal
                J_t_m1_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_t_m1_scal
                J_t_m0_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_t_m0_scal
                J_t_m1_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_t_m1_scal
                J_t_m0_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_t_m0_scal
                J_t_m1_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_t_m1_scal
                J_t_m0_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_t_m0_scal
                J_t_m1_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_t_m1_scal

                J_t_m0_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_t_m0_scal
                J_t_m1_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_t_m1_scal
                J_t_m0_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_t_m0_scal
                J_t_m1_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_t_m1_scal
                J_t_m0_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_t_m0_scal
                J_t_m1_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_t_m1_scal
                J_t_m0_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_t_m0_scal
                J_t_m1_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_t_m1_scal

                J_z_m0_00 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_z_m0_scal
                J_z_m1_00 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 0)*J_z_m1_scal
                J_z_m0_01 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_z_m0_scal
                J_z_m1_01 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 1)*J_z_m1_scal
                J_z_m0_02 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_z_m0_scal
                J_z_m1_02 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 2)*J_z_m1_scal
                J_z_m0_03 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_z_m0_scal
                J_z_m1_03 += r_shape_cubic(r_cell, 0)*z_shape_cubic(z_cell, 3)*J_z_m1_scal

                J_z_m0_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_z_m0_scal
                J_z_m1_10 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 0)*J_z_m1_scal
                J_z_m0_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_z_m0_scal
                J_z_m1_11 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 1)*J_z_m1_scal
                J_z_m0_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_z_m0_scal
                J_z_m1_12 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 2)*J_z_m1_scal
                J_z_m0_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_z_m0_scal
                J_z_m1_13 += r_shape_cubic(r_cell, 1)*z_shape_cubic(z_cell, 3)*J_z_m1_scal

                J_z_m0_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_z_m0_scal
                J_z_m1_20 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 0)*J_z_m1_scal
                J_z_m0_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_z_m0_scal
                J_z_m1_21 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 1)*J_z_m1_scal
                J_z_m0_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_z_m0_scal
                J_z_m1_22 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 2)*J_z_m1_scal
                J_z_m0_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_z_m0_scal
                J_z_m1_23 += r_shape_cubic(r_cell, 2)*z_shape_cubic(z_cell, 3)*J_z_m1_scal

                J_z_m0_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_z_m0_scal
                J_z_m1_30 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 0)*J_z_m1_scal
                J_z_m0_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_z_m0_scal
                J_z_m1_31 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 1)*J_z_m1_scal
                J_z_m0_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_z_m0_scal
                J_z_m1_32 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 2)*J_z_m1_scal
                J_z_m0_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_z_m0_scal
                J_z_m1_33 += r_shape_cubic(r_cell, 3)*z_shape_cubic(z_cell, 3)*J_z_m1_scal

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

            j_r_m0_global[i_thread, iz_cell - 1 + szl, ir_cell - 1 + srl] += J_r_m0_00
            j_r_m1_global[i_thread, iz_cell - 1 + szl, ir_cell - 1 + srl] += J_r_m1_00
            j_r_m0_global[i_thread, iz_cell, ir_cell - 1 + srl] += J_r_m0_01
            j_r_m1_global[i_thread, iz_cell, ir_cell - 1 + srl] += J_r_m1_01
            j_r_m0_global[i_thread, iz_cell + 1 + szu, ir_cell - 1 + srl] += J_r_m0_02
            j_r_m1_global[i_thread, iz_cell + 1 + szu, ir_cell - 1 + srl] += J_r_m1_02
            j_r_m0_global[i_thread, iz_cell + 2 + szu2, ir_cell - 1 + srl] += J_r_m0_03
            j_r_m1_global[i_thread, iz_cell + 2 + szu2, ir_cell - 1 + srl] += J_r_m1_03
            j_r_m0_global[i_thread, iz_cell - 1 + szl, ir_cell ] += J_r_m0_10
            j_r_m1_global[i_thread, iz_cell - 1 + szl, ir_cell ] += J_r_m1_10
            j_r_m0_global[i_thread, iz_cell, ir_cell] += J_r_m0_11
            j_r_m1_global[i_thread, iz_cell, ir_cell] += J_r_m1_11
            j_r_m0_global[i_thread, iz_cell + 1 + szu, ir_cell] += J_r_m0_12
            j_r_m1_global[i_thread, iz_cell + 1 + szu, ir_cell] += J_r_m1_12
            j_r_m0_global[i_thread, iz_cell + 2 + szu2, ir_cell] += J_r_m0_13
            j_r_m1_global[i_thread, iz_cell + 2 + szu2, ir_cell] += J_r_m1_13
            j_r_m0_global[i_thread, iz_cell - 1 + szl, ir_cell + 1 + sru] += J_r_m0_20
            j_r_m1_global[i_thread, iz_cell - 1 + szl, ir_cell + 1 + sru] += J_r_m1_20
            j_r_m0_global[i_thread, iz_cell, ir_cell + 1 + sru] += J_r_m0_21
            j_r_m1_global[i_thread, iz_cell, ir_cell + 1 + sru] += J_r_m1_21
            j_r_m0_global[i_thread, iz_cell + 1 + szu, ir_cell + 1 + sru] += J_r_m0_22
            j_r_m1_global[i_thread, iz_cell + 1 + szu, ir_cell + 1 + sru] += J_r_m1_22
            j_r_m0_global[i_thread, iz_cell + 2 + szu2, ir_cell + 1 + sru] += J_r_m0_23
            j_r_m1_global[i_thread, iz_cell + 2 + szu2, ir_cell + 1 + sru] += J_r_m1_23
            j_r_m0_global[i_thread, iz_cell - 1 + szl, ir_cell + 2 + sru2] += J_r_m0_30
            j_r_m1_global[i_thread, iz_cell - 1 + szl, ir_cell + 2 + sru2] += J_r_m1_30
            j_r_m0_global[i_thread, iz_cell, ir_cell + 2 + sru2] += J_r_m0_31
            j_r_m1_global[i_thread, iz_cell, ir_cell + 2 + sru2] += J_r_m1_31
            j_r_m0_global[i_thread, iz_cell + 1 + szu, ir_cell + 2 + sru2] += J_r_m0_32
            j_r_m1_global[i_thread, iz_cell + 1 + szu, ir_cell + 2 + sru2] += J_r_m1_32
            j_r_m0_global[i_thread, iz_cell + 2 + szu2, ir_cell + 2 + sru2] += J_r_m0_33
            j_r_m1_global[i_thread, iz_cell + 2 + szu2, ir_cell + 2 + sru2] += J_r_m1_33

            j_t_m0_global[i_thread, iz_cell - 1 + szl, ir_cell - 1 + srl] += J_t_m0_00
            j_t_m1_global[i_thread, iz_cell - 1 + szl, ir_cell - 1 + srl] += J_t_m1_00
            j_t_m0_global[i_thread, iz_cell, ir_cell - 1 + srl] += J_t_m0_01
            j_t_m1_global[i_thread, iz_cell, ir_cell - 1 + srl] += J_t_m1_01
            j_t_m0_global[i_thread, iz_cell + 1 + szu, ir_cell - 1 + srl] += J_t_m0_02
            j_t_m1_global[i_thread, iz_cell + 1 + szu, ir_cell - 1 + srl] += J_t_m1_02
            j_t_m0_global[i_thread, iz_cell + 2 + szu2, ir_cell - 1 + srl] += J_t_m0_03
            j_t_m1_global[i_thread, iz_cell + 2 + szu2, ir_cell - 1 + srl] += J_t_m1_03
            j_t_m0_global[i_thread, iz_cell - 1 + szl, ir_cell ] += J_t_m0_10
            j_t_m1_global[i_thread, iz_cell - 1 + szl, ir_cell ] += J_t_m1_10
            j_t_m0_global[i_thread, iz_cell, ir_cell] += J_t_m0_11
            j_t_m1_global[i_thread, iz_cell, ir_cell] += J_t_m1_11
            j_t_m0_global[i_thread, iz_cell + 1 + szu, ir_cell] += J_t_m0_12
            j_t_m1_global[i_thread, iz_cell + 1 + szu, ir_cell] += J_t_m1_12
            j_t_m0_global[i_thread, iz_cell + 2 + szu2, ir_cell] += J_t_m0_13
            j_t_m1_global[i_thread, iz_cell + 2 + szu2, ir_cell] += J_t_m1_13
            j_t_m0_global[i_thread, iz_cell - 1 + szl, ir_cell + 1 + sru] += J_t_m0_20
            j_t_m1_global[i_thread, iz_cell - 1 + szl, ir_cell + 1 + sru] += J_t_m1_20
            j_t_m0_global[i_thread, iz_cell, ir_cell + 1 + sru] += J_t_m0_21
            j_t_m1_global[i_thread, iz_cell, ir_cell + 1 + sru] += J_t_m1_21
            j_t_m0_global[i_thread, iz_cell + 1 + szu, ir_cell + 1 + sru] += J_t_m0_22
            j_t_m1_global[i_thread, iz_cell + 1 + szu, ir_cell + 1 + sru] += J_t_m1_22
            j_t_m0_global[i_thread, iz_cell + 2 + szu2, ir_cell + 1 + sru] += J_t_m0_23
            j_t_m1_global[i_thread, iz_cell + 2 + szu2, ir_cell + 1 + sru] += J_t_m1_23
            j_t_m0_global[i_thread, iz_cell - 1 + szl, ir_cell + 2 + sru2] += J_t_m0_30
            j_t_m1_global[i_thread, iz_cell - 1 + szl, ir_cell + 2 + sru2] += J_t_m1_30
            j_t_m0_global[i_thread, iz_cell, ir_cell + 2 + sru2] += J_t_m0_31
            j_t_m1_global[i_thread, iz_cell, ir_cell + 2 + sru2] += J_t_m1_31
            j_t_m0_global[i_thread, iz_cell + 1 + szu, ir_cell + 2 + sru2] += J_t_m0_32
            j_t_m1_global[i_thread, iz_cell + 1 + szu, ir_cell + 2 + sru2] += J_t_m1_32
            j_t_m0_global[i_thread, iz_cell + 2 + szu2, ir_cell + 2 + sru2] += J_t_m0_33
            j_t_m1_global[i_thread, iz_cell + 2 + szu2, ir_cell + 2 + sru2] += J_t_m1_33

            j_z_m0_global[i_thread, iz_cell - 1 + szl, ir_cell - 1 + srl] += J_z_m0_00
            j_z_m1_global[i_thread, iz_cell - 1 + szl, ir_cell - 1 + srl] += J_z_m1_00
            j_z_m0_global[i_thread, iz_cell, ir_cell - 1 + srl] += J_z_m0_01
            j_z_m1_global[i_thread, iz_cell, ir_cell - 1 + srl] += J_z_m1_01
            j_z_m0_global[i_thread, iz_cell + 1 + szu, ir_cell - 1 + srl] += J_z_m0_02
            j_z_m1_global[i_thread, iz_cell + 1 + szu, ir_cell - 1 + srl] += J_z_m1_02
            j_z_m0_global[i_thread, iz_cell + 2 + szu2, ir_cell - 1 + srl] += J_z_m0_03
            j_z_m1_global[i_thread, iz_cell + 2 + szu2, ir_cell - 1 + srl] += J_z_m1_03
            j_z_m0_global[i_thread, iz_cell - 1 + szl, ir_cell ] += J_z_m0_10
            j_z_m1_global[i_thread, iz_cell - 1 + szl, ir_cell ] += J_z_m1_10
            j_z_m0_global[i_thread, iz_cell, ir_cell] += J_z_m0_11
            j_z_m1_global[i_thread, iz_cell, ir_cell] += J_z_m1_11
            j_z_m0_global[i_thread, iz_cell + 1 + szu, ir_cell] += J_z_m0_12
            j_z_m1_global[i_thread, iz_cell + 1 + szu, ir_cell] += J_z_m1_12
            j_z_m0_global[i_thread, iz_cell + 2 + szu2, ir_cell] += J_z_m0_13
            j_z_m1_global[i_thread, iz_cell + 2 + szu2, ir_cell] += J_z_m1_13
            j_z_m0_global[i_thread, iz_cell - 1 + szl, ir_cell + 1 + sru] += J_z_m0_20
            j_z_m1_global[i_thread, iz_cell - 1 + szl, ir_cell + 1 + sru] += J_z_m1_20
            j_z_m0_global[i_thread, iz_cell, ir_cell + 1 + sru] += J_z_m0_21
            j_z_m1_global[i_thread, iz_cell, ir_cell + 1 + sru] += J_z_m1_21
            j_z_m0_global[i_thread, iz_cell + 1 + szu, ir_cell + 1 + sru] += J_z_m0_22
            j_z_m1_global[i_thread, iz_cell + 1 + szu, ir_cell + 1 + sru] += J_z_m1_22
            j_z_m0_global[i_thread, iz_cell + 2 + szu2, ir_cell + 1 + sru] += J_z_m0_23
            j_z_m1_global[i_thread, iz_cell + 2 + szu2, ir_cell + 1 + sru] += J_z_m1_23
            j_z_m0_global[i_thread, iz_cell - 1 + szl, ir_cell + 2 + sru2] += J_z_m0_30
            j_z_m1_global[i_thread, iz_cell - 1 + szl, ir_cell + 2 + sru2] += J_z_m1_30
            j_z_m0_global[i_thread, iz_cell, ir_cell + 2 + sru2] += J_z_m0_31
            j_z_m1_global[i_thread, iz_cell, ir_cell + 2 + sru2] += J_z_m1_31
            j_z_m0_global[i_thread, iz_cell + 1 + szu, ir_cell + 2 + sru2] += J_z_m0_32
            j_z_m1_global[i_thread, iz_cell + 1 + szu, ir_cell + 2 + sru2] += J_z_m1_32
            j_z_m0_global[i_thread, iz_cell + 2 + szu2, ir_cell + 2 + sru2] += J_z_m0_33
            j_z_m1_global[i_thread, iz_cell + 2 + szu2, ir_cell + 2 + sru2] += J_z_m1_33

    return

# -----------------------------------------------------------------------
# Parallel reduction of the global arrays for threads into a single array
# -----------------------------------------------------------------------

@njit_parallel
def sum_reduce_2d_array( global_array, reduced_array ):
    """
    Sum the array `global_array` along its first axis and
    add it into `reduced_array`.

    Parameters:
    -----------
    global_array: 3darray of complexs
       Field array whose first dimension corresponds to the
       reduction dimension (typically: the number of threads used
       during the current deposition)

    reduced array: 2darray of complexs
    """
    # Extract size of each dimension
    Nreduce, Nz, Nr = global_array.shape

    # Parallel loop over iz
    for iz in prange( Nz ):
        # Loop over the reduction dimension (slow dimension)
        for it in range( Nreduce ):
            # Loop over ir (fast dimension)
            for ir in range( Nr ):

                reduced_array[ iz, ir ] +=  global_array[ it, iz, ir ]
