# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the deposition methods for rho and J for linear order shapes
"""
from numba import cuda, int64
import math
from scipy.constants import c
import numpy as np


# Shape Factor functions to compute particle shapes.
@cuda.jit(device=True, inline=True)
def get_z_shape_linear(cell_position, index):
    iz = int64(math.floor(cell_position))
    if index == 0:
        return iz+1.-cell_position
    if index == 1:
        return cell_position - iz


@cuda.jit(device=True, inline=True)
def get_r_shape_linear(cell_position, index):
    flip_factor = 1.
    ir = int64(math.floor(cell_position))
    if index == 0:
        if ir < 0:
            flip_factor = -1.
        return flip_factor*(ir+1.-cell_position)
    if index == 1:
        return flip_factor*(cell_position - ir)


# -------------------------------
# Field deposition utility - rho
# -------------------------------

@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], \
                float64, float64, int32, \
                float64, float64, int32, \
                complex128[:,:], complex128[:,:], \
                int32[:], int32[:])')
def deposit_rho_gpu_linear(x, y, z, w,
                           invdz, zmin, Nz,
                           invdr, rmin, Nr,
                           rho_m0, rho_m1,
                           cell_idx, prefix_sum):
    """
    Deposition of the charge density rho using numba on the GPU.
    Iterates over the cells and over the particles per cell.
    Calculates the weighted amount of rho that is deposited to the
    4 cells surounding the particle based on its shape (linear).

    The particles are sorted by their cell index (the lower cell
    in r and z that they deposit to) and the deposited field
    is split into 4 variables (one for each possible direction,
    e.g. upper in z, lower in r) to maintain parallelism while
    avoiding any race conditions.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    w : 1d array of floats
        The weights of the particles

    rho_m0, rho_m1 : 2darrays of complexs
        The charge density on the interpolation grid for
        mode 0 and 1. (is modified by this function)

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the considered direction

    Nz, Nr : int
        Number of gridpoints along the considered direction

    cell_idx : 1darray of integers
        The cell index of the particle

    prefix_sum : 1darray of integers
        Represents the cumulative sum of
        the particles per cell
    """
    # Get the 1D CUDA grid
    i = cuda.grid(1)
    # Deposit the field per cell in parallel (for threads < number of cells)
    if i < prefix_sum.shape[0]:
        # Calculate the cell index in 2D from the 1D threadIdx
        iz_cell = int(i / Nr)
        ir_cell = int(i - iz_cell * Nr)
        # Calculate the inclusive offset for the current cell
        # It represents the number of particles contained in all other cells
        # with an index smaller than i + the total number of particles in the
        # current cell (inclusive).
        incl_offset = np.int32(prefix_sum[i])
        # Calculate the frequency per cell from the offset and the previous
        # offset (prefix_sum[i-1]).
        if i > 0:
            frequency_per_cell = np.int32(incl_offset - prefix_sum[i - 1])
        if i == 0:
            frequency_per_cell = np.int32(incl_offset)

        # Declare local field arrays
        R_m0_00 = 0.
        R_m0_01 = 0.
        R_m0_10 = 0.
        R_m0_11 = 0.

        R_m1_00 = 0. + 0.j
        R_m1_01 = 0. + 0.j
        R_m1_10 = 0. + 0.j
        R_m1_11 = 0. + 0.j

        for j in range(frequency_per_cell):
            # Get the particle index before the sorting
            # --------------------------------------------
            # (Since incl_offset is a cumulative sum of particle number,
            # and since python index starts at 0, one has to add -1)
            ptcl_idx = incl_offset-1-j

            # Preliminary arrays for the cylindrical conversion
            # --------------------------------------------
            # Position
            xj = x[ptcl_idx]
            yj = y[ptcl_idx]
            zj = z[ptcl_idx]
            # Weights
            wj = w[ptcl_idx]

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

            # Boundary Region Shifts
            ir_lower = int64(math.floor(r_cell))

            R_m0_00 += get_r_shape_linear(r_cell, 0)*get_z_shape_linear(z_cell, 0) * R_m0_scal
            R_m0_01 += get_r_shape_linear(r_cell, 0)*get_z_shape_linear(z_cell, 1) * R_m0_scal
            R_m1_00 += get_r_shape_linear(r_cell, 0)*get_z_shape_linear(z_cell, 0) * R_m1_scal
            R_m1_01 += get_r_shape_linear(r_cell, 0)*get_z_shape_linear(z_cell, 1) * R_m1_scal

            if ir_lower == -1:
                R_m0_00 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 0) * R_m0_scal
                R_m0_01 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 1) * R_m0_scal
                R_m1_00 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 0) * R_m1_scal
                R_m1_01 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 1) * R_m1_scal
            else:
                R_m0_10 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 0) * R_m0_scal
                R_m0_11 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 1) * R_m0_scal
                R_m1_10 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 0) * R_m1_scal
                R_m1_11 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 1) * R_m1_scal

        # Cell shifts for the simulation boundaries
        shift_r = 0
        shift_z = 0
        if ir_cell+1 > (Nr-1):
            shift_r = -1
        if iz_cell+1 > Nz-1:
            shift_z -= Nz
        cuda.atomic.add(rho_m0.real, (iz_cell, ir_cell), R_m0_00.real)
        cuda.atomic.add(rho_m1.real, (iz_cell, ir_cell), R_m1_00.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell, ir_cell), R_m1_00.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell+1 + shift_z, ir_cell), R_m0_01.real)
        cuda.atomic.add(rho_m1.real, (iz_cell+1 + shift_z, ir_cell), R_m1_01.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell+1 + shift_z, ir_cell), R_m1_01.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell, ir_cell+1 + shift_r), R_m0_10.real)
        cuda.atomic.add(rho_m1.real, (iz_cell, ir_cell+1 + shift_r), R_m1_10.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell, ir_cell+1 + shift_r), R_m1_10.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell+1 + shift_z, ir_cell+1 + shift_r), R_m0_11.real)
        cuda.atomic.add(rho_m1.real, (iz_cell+1 + shift_z, ir_cell+1 + shift_r), R_m1_11.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell+1 + shift_z, ir_cell+1 + shift_r), R_m1_11.imag)


# -------------------------------
# Field deposition utility - J
# -------------------------------

@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], \
                float64, float64, int32, \
                float64, float64, int32, \
                complex128[:,:], complex128[:,:], \
                complex128[:,:], complex128[:,:], \
                complex128[:,:], complex128[:,:],\
                int32[:], int32[:])')
def deposit_J_gpu_linear(x, y, z, w,
                         ux, uy, uz, inv_gamma,
                         invdz, zmin, Nz,
                         invdr, rmin, Nr,
                         j_r_m0, j_r_m1,
                         j_t_m0, j_t_m1,
                         j_z_m0, j_z_m1,
                         cell_idx, prefix_sum):
    """
    Deposition of the current J using numba on the GPU.
    Iterates over the cells and over the particles per cell.
    Calculates the weighted amount of J that is deposited to the
    4 cells surounding the particle based on its shape (linear).

    The particles are sorted by their cell index (the lower cell
    in r and z that they deposit to) and the deposited field
    is split into 4 variables (one for each possible direction,
    e.g. upper in z, lower in r) to maintain parallelism while
    avoiding any race conditions.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    w : 1d array of floats
        The weights of the particles

    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    j_r_m0, j_r_m1, j_t_m0, j_t_m1, j_z_m0, j_z_m1,: 2darrays of complexs
        The current component in each direction (r, t, z)
        on the interpolation grid for mode 0 and 1.
        (is modified by this function)

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the direction considered

    Nz, Nr : int
        Number of gridpoints along the considered direction

    cell_idx : 1darray of integers
        The cell index of the particle

    prefix_sum : 1darray of integers
        Represents the cumulative sum of
        the particles per cell
    """
    # Get the 1D CUDA grid
    i = cuda.grid(1)
    # Deposit the field per cell in parallel (for threads < number of cells)
    if i < prefix_sum.shape[0]:
        # Calculate the cell index in 2D from the 1D threadIdx
        iz_cell = int(i/Nr)
        ir_cell = int(i - iz_cell * Nr)
        # Calculate the inclusive offset for the current cell
        # It represents the number of particles contained in all other cells
        # with an index smaller than i + the total number of particles in the
        # current cell (inclusive).
        incl_offset = np.int32(prefix_sum[i])
        # Calculate the frequency per cell from the offset and the previous
        # offset (prefix_sum[i-1]).
        if i > 0:
            frequency_per_cell = np.int32(incl_offset - prefix_sum[i-1])
        if i == 0:
            frequency_per_cell = np.int32(incl_offset)

        # Declare the local field value for
        # all possible deposition directions,
        # depending on the shape order and per mode for r,t and z.

        J_r_m0_00 = 0.
        J_r_m1_00 = 0. + 0.j
        J_t_m0_00 = 0.# + 0.j
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


        # Loop over the number of particles per cell
        for j in range(frequency_per_cell):
            # Get the particle index
            # ----------------------
            # (Since incl_offset is a cumulative sum of particle number,
            # and since python index starts at 0, one has to add -1)
            ptcl_idx = incl_offset-1-j

            # Preliminary arrays for the cylindrical conversion
            # --------------------------------------------
            # Position
            xj = x[ptcl_idx]
            yj = y[ptcl_idx]
            zj = z[ptcl_idx]
            # Velocity
            uxj = ux[ptcl_idx]
            uyj = uy[ptcl_idx]
            uzj = uz[ptcl_idx]
            # Inverse gamma
            inv_gammaj = inv_gamma[ptcl_idx]
            # Weights
            wj = w[ptcl_idx]

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

            # Deposit on local copies at respective position
            ir_lower = int64(math.floor(r_cell))

            J_r_m0_00 += get_r_shape_linear(r_cell, 0)*get_z_shape_linear(z_cell, 0) * J_r_m0_scal
            J_t_m0_00 += get_r_shape_linear(r_cell, 0)*get_z_shape_linear(z_cell, 0) * J_t_m0_scal
            J_z_m0_00 += get_r_shape_linear(r_cell, 0)*get_z_shape_linear(z_cell, 0) * J_z_m0_scal
            J_r_m0_01 += get_r_shape_linear(r_cell, 0)*get_z_shape_linear(z_cell, 1) * J_r_m0_scal
            J_t_m0_01 += get_r_shape_linear(r_cell, 0)*get_z_shape_linear(z_cell, 1) * J_t_m0_scal
            J_z_m0_01 += get_r_shape_linear(r_cell, 0)*get_z_shape_linear(z_cell, 1) * J_z_m0_scal
            J_r_m1_00 += get_r_shape_linear(r_cell, 0)*get_z_shape_linear(z_cell, 0) * J_r_m1_scal
            J_t_m1_00 += get_r_shape_linear(r_cell, 0)*get_z_shape_linear(z_cell, 0) * J_t_m1_scal
            J_z_m1_00 += get_r_shape_linear(r_cell, 0)*get_z_shape_linear(z_cell, 0) * J_z_m1_scal
            J_r_m1_01 += get_r_shape_linear(r_cell, 0)*get_z_shape_linear(z_cell, 1) * J_r_m1_scal
            J_t_m1_01 += get_r_shape_linear(r_cell, 0)*get_z_shape_linear(z_cell, 1) * J_t_m1_scal
            J_z_m1_01 += get_r_shape_linear(r_cell, 0)*get_z_shape_linear(z_cell, 1) * J_z_m1_scal

            # Take into account lower r flips
            if ir_lower == -1:
                J_r_m0_00 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 0) * J_r_m0_scal
                J_t_m0_00 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 0) * J_t_m0_scal
                J_z_m0_00 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 0) * J_z_m0_scal
                J_r_m0_01 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 1) * J_r_m0_scal
                J_t_m0_01 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 1) * J_t_m0_scal
                J_z_m0_01 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 1) * J_z_m0_scal
                J_r_m1_00 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 0) * J_r_m1_scal
                J_t_m1_00 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 0) * J_t_m1_scal
                J_z_m1_00 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 0) * J_z_m1_scal
                J_r_m1_01 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 1) * J_r_m1_scal
                J_t_m1_01 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 1) * J_t_m1_scal
                J_z_m1_01 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 1) * J_z_m1_scal
            else:
                J_r_m0_10 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 0) * J_r_m0_scal
                J_t_m0_10 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 0) * J_t_m0_scal
                J_z_m0_10 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 0) * J_z_m0_scal
                J_r_m0_11 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 1) * J_r_m0_scal
                J_t_m0_11 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 1) * J_t_m0_scal
                J_z_m0_11 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 1) * J_z_m0_scal
                J_r_m1_10 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 0) * J_r_m1_scal
                J_t_m1_10 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 0) * J_t_m1_scal
                J_z_m1_10 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 0) * J_z_m1_scal
                J_r_m1_11 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 1) * J_r_m1_scal
                J_t_m1_11 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 1) * J_t_m1_scal
                J_z_m1_11 += get_r_shape_linear(r_cell, 1)*get_z_shape_linear(z_cell, 1) * J_z_m1_scal

        # Cell shifts for the simulation boundaries
        shift_r = 0
        shift_z = 0
        if (ir_cell+1) > (Nr-1):
            shift_r = -1
        if (iz_cell+1) > Nz-1:
            shift_z -= Nz
        cuda.atomic.add(j_r_m0.real, (iz_cell, ir_cell), J_r_m0_00.real)
        cuda.atomic.add(j_r_m1.real, (iz_cell, ir_cell), J_r_m1_00.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell, ir_cell), J_r_m1_00.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell+1 + shift_z, ir_cell), J_r_m0_01.real)
        cuda.atomic.add(j_r_m1.real, (iz_cell+1 + shift_z, ir_cell), J_r_m1_01.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell+1 + shift_z, ir_cell), J_r_m1_01.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell, ir_cell+1 + shift_r), J_r_m0_10.real)
        cuda.atomic.add(j_r_m1.real, (iz_cell, ir_cell+1 + shift_r), J_r_m1_10.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell, ir_cell+1 + shift_r), J_r_m1_10.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell+1 + shift_z, ir_cell+1 + shift_r), J_r_m0_11.real)
        cuda.atomic.add(j_r_m1.real, (iz_cell+1 + shift_z, ir_cell+1 + shift_r), J_r_m1_11.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell+1 + shift_z, ir_cell+1 + shift_r), J_r_m1_11.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell, ir_cell), J_t_m0_00.real)
        cuda.atomic.add(j_t_m1.real, (iz_cell, ir_cell), J_t_m1_00.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell, ir_cell), J_t_m1_00.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell+1 + shift_z, ir_cell), J_t_m0_01.real)
        cuda.atomic.add(j_t_m1.real, (iz_cell+1 + shift_z, ir_cell), J_t_m1_01.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell+1 + shift_z, ir_cell), J_t_m1_01.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell, ir_cell+1 + shift_r), J_t_m0_10.real)
        cuda.atomic.add(j_t_m1.real, (iz_cell, ir_cell+1 + shift_r), J_t_m1_10.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell, ir_cell+1 + shift_r), J_t_m1_10.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell+1 + shift_z, ir_cell+1 + shift_r), J_t_m0_11.real)
        cuda.atomic.add(j_t_m1.real, (iz_cell+1 + shift_z, ir_cell+1 + shift_r), J_t_m1_11.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell+1 + shift_z, ir_cell+1 + shift_r), J_t_m1_11.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell, ir_cell), J_z_m0_00.real)
        cuda.atomic.add(j_z_m1.real, (iz_cell, ir_cell), J_z_m1_00.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell, ir_cell), J_z_m1_00.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell+1 + shift_z, ir_cell), J_z_m0_01.real)
        cuda.atomic.add(j_z_m1.real, (iz_cell+1 + shift_z, ir_cell), J_z_m1_01.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell+1 + shift_z, ir_cell), J_z_m1_01.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell, ir_cell+1 + shift_r), J_z_m0_10.real)
        cuda.atomic.add(j_z_m1.real, (iz_cell, ir_cell+1 + shift_r), J_z_m1_10.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell, ir_cell+1 + shift_r), J_z_m1_10.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell+1 + shift_z, ir_cell+1 + shift_r), J_z_m0_11.real)
        cuda.atomic.add(j_z_m1.real, (iz_cell+1 + shift_z, ir_cell+1 + shift_r), J_z_m1_11.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell+1 + shift_z, ir_cell+1 + shift_r), J_z_m1_11.imag)
