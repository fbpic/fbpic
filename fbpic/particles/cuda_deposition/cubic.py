# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the deposition methods for rho and J for cubic order shapes
"""
from numba import cuda, int64
import math
from scipy.constants import c
import numpy as np

# Shape Factor helper functions to compute particle shapes.


@cuda.jit(device=True, inline=True)
def z_shape(cell_position, index):
    iz = int64(math.floor(cell_position)) - 1
    if index == 0:
        return (-1./6.)*((cell_position-iz)-2)**3
    if index == 1:
        return (1./6.)*(3*((cell_position-(iz+1))**3)-6*((cell_position-(iz+1))**2)+4)
    if index == 2:
        return (1./6.)*(3*(((iz+2)-cell_position)**3)-6*(((iz+2)-cell_position)**2)+4)
    if index == 3:
        return (-1./6.)*(((iz+3)-cell_position)-2)**3


@cuda.jit(device=True, inline=True)
def r_shape(cell_position, index):
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
# Field deposition utility - rho
# -------------------------------


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], \
                float64, float64, int32, \
                float64, float64, int32, \
                complex128[:,:], complex128[:,:], \
                int32[:], int32[:])')
def deposit_rho_gpu_cubic(x, y, z, w,
                          invdz, zmin, Nz,
                          invdr, rmin, Nr,
                          rho_m0, rho_m1,
                          cell_idx, prefix_sum):
    """
    Deposition of the charge density rho using numba on the GPU.
    Iterates over the cells and over the particles per cell.
    Calculates the weighted amount of rho that is deposited to the
    16 cells surounding the particle based on its shape (cubic).

    The particles are sorted by their cell index (the lower cell
    in r and z that they deposit to) and the deposited field
    is split into 16 variables (one for each surrounding cell) to
    maintain parallelism while avoiding any race conditions.

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
            # Compute values in local copies and consider boundaries
            ir0 = int64(math.floor(r_cell)) - 1

            if (ir0 == -2):
                R_m0_20 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*R_m0_scal
                R_m1_20 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*R_m1_scal
                R_m0_21 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*R_m0_scal
                R_m1_21 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*R_m1_scal
                R_m0_22 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*R_m0_scal
                R_m1_22 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*R_m1_scal
                R_m0_23 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*R_m0_scal
                R_m1_23 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*R_m1_scal

                R_m0_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*R_m0_scal
                R_m1_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*R_m1_scal
                R_m0_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*R_m0_scal
                R_m1_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*R_m1_scal
                R_m0_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*R_m0_scal
                R_m1_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*R_m1_scal
                R_m0_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*R_m0_scal
                R_m1_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*R_m1_scal

                R_m0_10 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*R_m0_scal
                R_m1_10 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*R_m1_scal
                R_m0_11 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*R_m0_scal
                R_m1_11 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*R_m1_scal
                R_m0_12 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*R_m0_scal
                R_m1_12 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*R_m1_scal
                R_m0_13 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*R_m0_scal
                R_m1_13 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*R_m1_scal

                R_m0_20 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*R_m0_scal
                R_m1_20 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*R_m1_scal
                R_m0_21 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*R_m0_scal
                R_m1_21 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*R_m1_scal
                R_m0_22 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*R_m0_scal
                R_m1_22 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*R_m1_scal
                R_m0_23 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*R_m0_scal
                R_m1_23 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*R_m1_scal

            if (ir0 == -1):
                R_m0_10 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*R_m0_scal
                R_m1_10 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*R_m1_scal
                R_m0_11 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*R_m0_scal
                R_m1_11 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*R_m1_scal
                R_m0_12 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*R_m0_scal
                R_m1_12 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*R_m1_scal
                R_m0_13 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*R_m0_scal
                R_m1_13 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*R_m1_scal

                R_m0_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*R_m0_scal
                R_m1_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*R_m1_scal
                R_m0_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*R_m0_scal
                R_m1_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*R_m1_scal
                R_m0_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*R_m0_scal
                R_m1_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*R_m1_scal
                R_m0_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*R_m0_scal
                R_m1_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*R_m1_scal

                R_m0_20 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*R_m0_scal
                R_m1_20 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*R_m1_scal
                R_m0_21 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*R_m0_scal
                R_m1_21 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*R_m1_scal
                R_m0_22 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*R_m0_scal
                R_m1_22 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*R_m1_scal
                R_m0_23 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*R_m0_scal
                R_m1_23 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*R_m1_scal

                R_m0_30 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*R_m0_scal
                R_m1_30 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*R_m1_scal
                R_m0_31 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*R_m0_scal
                R_m1_31 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*R_m1_scal
                R_m0_32 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*R_m0_scal
                R_m1_32 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*R_m1_scal
                R_m0_33 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*R_m0_scal
                R_m1_33 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*R_m1_scal
            if (ir0 >= 0):
                R_m0_00 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*R_m0_scal
                R_m1_00 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*R_m1_scal
                R_m0_01 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*R_m0_scal
                R_m1_01 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*R_m1_scal
                R_m0_02 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*R_m0_scal
                R_m1_02 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*R_m1_scal
                R_m0_03 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*R_m0_scal
                R_m1_03 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*R_m1_scal

                R_m0_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*R_m0_scal
                R_m1_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*R_m1_scal
                R_m0_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*R_m0_scal
                R_m1_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*R_m1_scal
                R_m0_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*R_m0_scal
                R_m1_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*R_m1_scal
                R_m0_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*R_m0_scal
                R_m1_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*R_m1_scal

                R_m0_20 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*R_m0_scal
                R_m1_20 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*R_m1_scal
                R_m0_21 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*R_m0_scal
                R_m1_21 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*R_m1_scal
                R_m0_22 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*R_m0_scal
                R_m1_22 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*R_m1_scal
                R_m0_23 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*R_m0_scal
                R_m1_23 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*R_m1_scal

                R_m0_30 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*R_m0_scal
                R_m1_30 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*R_m1_scal
                R_m0_31 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*R_m0_scal
                R_m1_31 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*R_m1_scal
                R_m0_32 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*R_m0_scal
                R_m1_32 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*R_m1_scal
                R_m0_33 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*R_m0_scal
                R_m1_33 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*R_m1_scal

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



        cuda.atomic.add(rho_m0.real, (iz_cell - 1 + szl, ir_cell - 1 + srl), R_m0_00.real)
        cuda.atomic.add(rho_m1.real, (iz_cell - 1 + szl, ir_cell - 1 + srl), R_m1_00.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell - 1 + szl, ir_cell - 1 + srl), R_m1_00.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell, ir_cell - 1 + srl), R_m0_01.real)
        cuda.atomic.add(rho_m1.real, (iz_cell, ir_cell - 1 + srl), R_m1_01.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell, ir_cell - 1 + srl), R_m1_01.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell + 1 + szu, ir_cell - 1 + srl), R_m0_02.real)
        cuda.atomic.add(rho_m1.real, (iz_cell + 1 + szu, ir_cell - 1 + srl), R_m1_02.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell + 1 + szu, ir_cell - 1 + srl), R_m1_02.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell + 2 + szu2, ir_cell - 1 + srl), R_m0_03.real)
        cuda.atomic.add(rho_m1.real, (iz_cell + 2 + szu2, ir_cell - 1 + srl), R_m1_03.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell + 2 + szu2, ir_cell - 1 + srl), R_m1_03.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell - 1 + szl, ir_cell ), R_m0_10.real)
        cuda.atomic.add(rho_m1.real, (iz_cell - 1 + szl, ir_cell ), R_m1_10.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell - 1 + szl, ir_cell ), R_m1_10.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell, ir_cell), R_m0_11.real)
        cuda.atomic.add(rho_m1.real, (iz_cell, ir_cell), R_m1_11.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell, ir_cell), R_m1_11.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell + 1 + szu, ir_cell), R_m0_12.real)
        cuda.atomic.add(rho_m1.real, (iz_cell + 1 + szu, ir_cell), R_m1_12.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell + 1 + szu, ir_cell), R_m1_12.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell + 2 + szu2, ir_cell), R_m0_13.real)
        cuda.atomic.add(rho_m1.real, (iz_cell + 2 + szu2, ir_cell), R_m1_13.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell + 2 + szu2, ir_cell), R_m1_13.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell - 1 + szl, ir_cell + 1 + sru), R_m0_20.real)
        cuda.atomic.add(rho_m1.real, (iz_cell - 1 + szl, ir_cell + 1 + sru), R_m1_20.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell - 1 + szl, ir_cell + 1 + sru), R_m1_20.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell, ir_cell + 1 + sru), R_m0_21.real)
        cuda.atomic.add(rho_m1.real, (iz_cell, ir_cell + 1 + sru), R_m1_21.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell, ir_cell + 1 + sru), R_m1_21.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell + 1 + szu, ir_cell + 1 + sru), R_m0_22.real)
        cuda.atomic.add(rho_m1.real, (iz_cell + 1 + szu, ir_cell + 1 + sru), R_m1_22.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell + 1 + szu, ir_cell + 1 + sru), R_m1_22.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell + 2 + szu2, ir_cell + 1 + sru), R_m0_23.real)
        cuda.atomic.add(rho_m1.real, (iz_cell + 2 + szu2, ir_cell + 1 + sru), R_m1_23.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell + 2 + szu2, ir_cell + 1 + sru), R_m1_23.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell - 1 + szl, ir_cell + 2 + sru2), R_m0_30.real)
        cuda.atomic.add(rho_m1.real, (iz_cell - 1 + szl, ir_cell + 2 + sru2), R_m1_30.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell - 1 + szl, ir_cell + 2 + sru2), R_m1_30.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell, ir_cell + 2 + sru2), R_m0_31.real)
        cuda.atomic.add(rho_m1.real, (iz_cell, ir_cell + 2 + sru2), R_m1_31.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell, ir_cell + 2 + sru2), R_m1_31.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell + 1 + szu, ir_cell + 2 + sru2), R_m0_32.real)
        cuda.atomic.add(rho_m1.real, (iz_cell + 1 + szu, ir_cell + 2 + sru2), R_m1_32.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell + 1 + szu, ir_cell + 2 + sru2), R_m1_32.imag)

        cuda.atomic.add(rho_m0.real, (iz_cell + 2 + szu2, ir_cell + 2 + sru2), R_m0_33.real)
        cuda.atomic.add(rho_m1.real, (iz_cell + 2 + szu2, ir_cell + 2 + sru2), R_m1_33.real)
        cuda.atomic.add(rho_m1.imag, (iz_cell + 2 + szu2, ir_cell + 2 + sru2), R_m1_33.imag)


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
def deposit_J_gpu_cubic(x, y, z, w,
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
    16 cells surounding the particle based on its shape (cubic).

    The particles are sorted by their cell index (the lower cell
    in r and z that they deposit to) and the deposited field
    is split into 16 variables (one for each cell) to maintain
    parallelism while avoiding any race conditions.

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

            # Compute values in local copies and consider boundaries
            ir0 = int64(math.floor(r_cell)) - 1

            if (ir0 == -2):
                J_r_m0_20 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*J_r_m0_scal
                J_r_m1_20 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*J_r_m1_scal
                J_r_m0_21 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_r_m0_scal
                J_r_m1_21 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_r_m1_scal
                J_r_m0_22 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_r_m0_scal
                J_r_m1_22 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_r_m1_scal
                J_r_m0_23 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_r_m0_scal
                J_r_m1_23 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_r_m1_scal

                J_r_m0_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_r_m0_scal
                J_r_m1_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_r_m1_scal
                J_r_m0_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_r_m0_scal
                J_r_m1_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_r_m1_scal
                J_r_m0_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_r_m0_scal
                J_r_m1_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_r_m1_scal
                J_r_m0_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_r_m0_scal
                J_r_m1_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_r_m1_scal

                J_r_m0_10 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_r_m0_scal
                J_r_m1_10 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_r_m1_scal
                J_r_m0_11 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_r_m0_scal
                J_r_m1_11 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_r_m1_scal
                J_r_m0_12 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_r_m0_scal
                J_r_m1_12 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_r_m1_scal
                J_r_m0_13 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_r_m0_scal
                J_r_m1_13 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_r_m1_scal

                J_r_m0_20 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_r_m0_scal
                J_r_m1_20 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_r_m1_scal
                J_r_m0_21 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_r_m0_scal
                J_r_m1_21 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_r_m1_scal
                J_r_m0_22 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_r_m0_scal
                J_r_m1_22 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_r_m1_scal
                J_r_m0_23 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_r_m0_scal
                J_r_m1_23 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_r_m1_scal
                J_r_m0_20 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*J_r_m0_scal

                J_t_m1_20 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*J_t_m1_scal
                J_t_m0_21 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_t_m0_scal
                J_t_m1_21 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_t_m1_scal
                J_t_m0_22 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_t_m0_scal
                J_t_m1_22 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_t_m1_scal
                J_t_m0_23 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_t_m0_scal
                J_t_m1_23 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_t_m1_scal

                J_t_m0_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_t_m0_scal
                J_t_m1_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_t_m1_scal
                J_t_m0_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_t_m0_scal
                J_t_m1_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_t_m1_scal
                J_t_m0_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_t_m0_scal
                J_t_m1_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_t_m1_scal
                J_t_m0_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_t_m0_scal
                J_t_m1_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_t_m1_scal

                J_t_m0_10 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_t_m0_scal
                J_t_m1_10 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_t_m1_scal
                J_t_m0_11 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_t_m0_scal
                J_t_m1_11 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_t_m1_scal
                J_t_m0_12 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_t_m0_scal
                J_t_m1_12 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_t_m1_scal
                J_t_m0_13 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_t_m0_scal
                J_t_m1_13 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_t_m1_scal

                J_t_m0_20 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_t_m0_scal
                J_t_m1_20 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_t_m1_scal
                J_t_m0_21 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_t_m0_scal
                J_t_m1_21 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_t_m1_scal
                J_t_m0_22 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_t_m0_scal
                J_t_m1_22 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_t_m1_scal
                J_t_m0_23 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_t_m0_scal
                J_t_m1_23 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_t_m1_scal

                J_z_m1_20 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*J_z_m1_scal
                J_z_m0_21 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_z_m0_scal
                J_z_m1_21 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_z_m1_scal
                J_z_m0_22 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_z_m0_scal
                J_z_m1_22 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_z_m1_scal
                J_z_m0_23 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_z_m0_scal
                J_z_m1_23 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_z_m1_scal

                J_z_m0_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_z_m0_scal
                J_z_m1_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_z_m1_scal
                J_z_m0_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_z_m0_scal
                J_z_m1_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_z_m1_scal
                J_z_m0_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_z_m0_scal
                J_z_m1_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_z_m1_scal
                J_z_m0_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_z_m0_scal
                J_z_m1_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_z_m1_scal

                J_z_m0_10 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_z_m0_scal
                J_z_m1_10 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_z_m1_scal
                J_z_m0_11 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_z_m0_scal
                J_z_m1_11 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_z_m1_scal
                J_z_m0_12 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_z_m0_scal
                J_z_m1_12 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_z_m1_scal
                J_z_m0_13 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_z_m0_scal
                J_z_m1_13 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_z_m1_scal

                J_z_m0_20 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_z_m0_scal
                J_z_m1_20 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_z_m1_scal
                J_z_m0_21 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_z_m0_scal
                J_z_m1_21 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_z_m1_scal
                J_z_m0_22 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_z_m0_scal
                J_z_m1_22 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_z_m1_scal
                J_z_m0_23 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_z_m0_scal
                J_z_m1_23 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_z_m1_scal
            if (ir0 == -1):
                J_r_m0_10 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*J_r_m0_scal
                J_r_m1_10 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*J_r_m1_scal
                J_r_m0_11 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_r_m0_scal
                J_r_m1_11 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_r_m1_scal
                J_r_m0_12 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_r_m0_scal
                J_r_m1_12 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_r_m1_scal
                J_r_m0_13 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_r_m0_scal
                J_r_m1_13 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_r_m1_scal

                J_r_m0_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_r_m0_scal
                J_r_m1_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_r_m1_scal
                J_r_m0_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_r_m0_scal
                J_r_m1_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_r_m1_scal
                J_r_m0_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_r_m0_scal
                J_r_m1_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_r_m1_scal
                J_r_m0_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_r_m0_scal
                J_r_m1_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_r_m1_scal

                J_r_m0_20 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_r_m0_scal
                J_r_m1_20 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_r_m1_scal
                J_r_m0_21 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_r_m0_scal
                J_r_m1_21 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_r_m1_scal
                J_r_m0_22 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_r_m0_scal
                J_r_m1_22 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_r_m1_scal
                J_r_m0_23 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_r_m0_scal
                J_r_m1_23 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_r_m1_scal

                J_r_m0_30 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_r_m0_scal
                J_r_m1_30 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_r_m1_scal
                J_r_m0_31 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_r_m0_scal
                J_r_m1_31 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_r_m1_scal
                J_r_m0_32 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_r_m0_scal
                J_r_m1_32 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_r_m1_scal
                J_r_m0_33 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_r_m0_scal
                J_r_m1_33 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_r_m1_scal

                J_t_m0_10 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*J_t_m0_scal
                J_t_m1_10 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*J_t_m1_scal
                J_t_m0_11 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_t_m0_scal
                J_t_m1_11 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_t_m1_scal
                J_t_m0_12 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_t_m0_scal
                J_t_m1_12 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_t_m1_scal
                J_t_m0_13 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_t_m0_scal
                J_t_m1_13 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_t_m1_scal

                J_t_m0_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_t_m0_scal
                J_t_m1_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_t_m1_scal
                J_t_m0_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_t_m0_scal
                J_t_m1_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_t_m1_scal
                J_t_m0_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_t_m0_scal
                J_t_m1_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_t_m1_scal
                J_t_m0_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_t_m0_scal
                J_t_m1_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_t_m1_scal

                J_t_m0_20 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_t_m0_scal
                J_t_m1_20 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_t_m1_scal
                J_t_m0_21 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_t_m0_scal
                J_t_m1_21 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_t_m1_scal
                J_t_m0_22 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_t_m0_scal
                J_t_m1_22 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_t_m1_scal
                J_t_m0_23 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_t_m0_scal
                J_t_m1_23 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_t_m1_scal

                J_t_m0_30 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_t_m0_scal
                J_t_m1_30 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_t_m1_scal
                J_t_m0_31 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_t_m0_scal
                J_t_m1_31 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_t_m1_scal
                J_t_m0_32 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_t_m0_scal
                J_t_m1_32 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_t_m1_scal
                J_t_m0_33 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_t_m0_scal
                J_t_m1_33 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_t_m1_scal

                J_z_m0_10 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*J_z_m0_scal
                J_z_m1_10 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*J_z_m1_scal
                J_z_m0_11 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_z_m0_scal
                J_z_m1_11 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_z_m1_scal
                J_z_m0_12 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_z_m0_scal
                J_z_m1_12 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_z_m1_scal
                J_z_m0_13 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_z_m0_scal
                J_z_m1_13 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_z_m1_scal

                J_z_m0_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_z_m0_scal
                J_z_m1_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_z_m1_scal
                J_z_m0_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_z_m0_scal
                J_z_m1_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_z_m1_scal
                J_z_m0_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_z_m0_scal
                J_z_m1_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_z_m1_scal
                J_z_m0_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_z_m0_scal
                J_z_m1_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_z_m1_scal

                J_z_m0_20 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_z_m0_scal
                J_z_m1_20 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_z_m1_scal
                J_z_m0_21 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_z_m0_scal
                J_z_m1_21 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_z_m1_scal
                J_z_m0_22 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_z_m0_scal
                J_z_m1_22 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_z_m1_scal
                J_z_m0_23 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_z_m0_scal
                J_z_m1_23 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_z_m1_scal

                J_z_m0_30 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_z_m0_scal
                J_z_m1_30 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_z_m1_scal
                J_z_m0_31 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_z_m0_scal
                J_z_m1_31 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_z_m1_scal
                J_z_m0_32 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_z_m0_scal
                J_z_m1_32 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_z_m1_scal
                J_z_m0_33 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_z_m0_scal
                J_z_m1_33 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_z_m1_scal
            if (ir0 >= 0):
                J_r_m0_00 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*J_r_m0_scal
                J_r_m1_00 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*J_r_m1_scal
                J_r_m0_01 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_r_m0_scal
                J_r_m1_01 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_r_m1_scal
                J_r_m0_02 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_r_m0_scal
                J_r_m1_02 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_r_m1_scal
                J_r_m0_03 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_r_m0_scal
                J_r_m1_03 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_r_m1_scal

                J_r_m0_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_r_m0_scal
                J_r_m1_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_r_m1_scal
                J_r_m0_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_r_m0_scal
                J_r_m1_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_r_m1_scal
                J_r_m0_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_r_m0_scal
                J_r_m1_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_r_m1_scal
                J_r_m0_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_r_m0_scal
                J_r_m1_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_r_m1_scal

                J_r_m0_20 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_r_m0_scal
                J_r_m1_20 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_r_m1_scal
                J_r_m0_21 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_r_m0_scal
                J_r_m1_21 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_r_m1_scal
                J_r_m0_22 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_r_m0_scal
                J_r_m1_22 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_r_m1_scal
                J_r_m0_23 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_r_m0_scal
                J_r_m1_23 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_r_m1_scal

                J_r_m0_30 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_r_m0_scal
                J_r_m1_30 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_r_m1_scal
                J_r_m0_31 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_r_m0_scal
                J_r_m1_31 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_r_m1_scal
                J_r_m0_32 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_r_m0_scal
                J_r_m1_32 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_r_m1_scal
                J_r_m0_33 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_r_m0_scal
                J_r_m1_33 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_r_m1_scal

                J_t_m0_00 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*J_t_m0_scal
                J_t_m1_00 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*J_t_m1_scal
                J_t_m0_01 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_t_m0_scal
                J_t_m1_01 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_t_m1_scal
                J_t_m0_02 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_t_m0_scal
                J_t_m1_02 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_t_m1_scal
                J_t_m0_03 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_t_m0_scal
                J_t_m1_03 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_t_m1_scal

                J_t_m0_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_t_m0_scal
                J_t_m1_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_t_m1_scal
                J_t_m0_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_t_m0_scal
                J_t_m1_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_t_m1_scal
                J_t_m0_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_t_m0_scal
                J_t_m1_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_t_m1_scal
                J_t_m0_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_t_m0_scal
                J_t_m1_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_t_m1_scal

                J_t_m0_20 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_t_m0_scal
                J_t_m1_20 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_t_m1_scal
                J_t_m0_21 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_t_m0_scal
                J_t_m1_21 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_t_m1_scal
                J_t_m0_22 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_t_m0_scal
                J_t_m1_22 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_t_m1_scal
                J_t_m0_23 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_t_m0_scal
                J_t_m1_23 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_t_m1_scal

                J_t_m0_30 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_t_m0_scal
                J_t_m1_30 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_t_m1_scal
                J_t_m0_31 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_t_m0_scal
                J_t_m1_31 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_t_m1_scal
                J_t_m0_32 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_t_m0_scal
                J_t_m1_32 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_t_m1_scal
                J_t_m0_33 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_t_m0_scal
                J_t_m1_33 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_t_m1_scal

                J_z_m0_00 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*J_z_m0_scal
                J_z_m1_00 += r_shape(r_cell, 0)*z_shape(z_cell, 0)*J_z_m1_scal
                J_z_m0_01 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_z_m0_scal
                J_z_m1_01 += r_shape(r_cell, 0)*z_shape(z_cell, 1)*J_z_m1_scal
                J_z_m0_02 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_z_m0_scal
                J_z_m1_02 += r_shape(r_cell, 0)*z_shape(z_cell, 2)*J_z_m1_scal
                J_z_m0_03 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_z_m0_scal
                J_z_m1_03 += r_shape(r_cell, 0)*z_shape(z_cell, 3)*J_z_m1_scal

                J_z_m0_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_z_m0_scal
                J_z_m1_10 += r_shape(r_cell, 1)*z_shape(z_cell, 0)*J_z_m1_scal
                J_z_m0_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_z_m0_scal
                J_z_m1_11 += r_shape(r_cell, 1)*z_shape(z_cell, 1)*J_z_m1_scal
                J_z_m0_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_z_m0_scal
                J_z_m1_12 += r_shape(r_cell, 1)*z_shape(z_cell, 2)*J_z_m1_scal
                J_z_m0_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_z_m0_scal
                J_z_m1_13 += r_shape(r_cell, 1)*z_shape(z_cell, 3)*J_z_m1_scal

                J_z_m0_20 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_z_m0_scal
                J_z_m1_20 += r_shape(r_cell, 2)*z_shape(z_cell, 0)*J_z_m1_scal
                J_z_m0_21 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_z_m0_scal
                J_z_m1_21 += r_shape(r_cell, 2)*z_shape(z_cell, 1)*J_z_m1_scal
                J_z_m0_22 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_z_m0_scal
                J_z_m1_22 += r_shape(r_cell, 2)*z_shape(z_cell, 2)*J_z_m1_scal
                J_z_m0_23 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_z_m0_scal
                J_z_m1_23 += r_shape(r_cell, 2)*z_shape(z_cell, 3)*J_z_m1_scal

                J_z_m0_30 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_z_m0_scal
                J_z_m1_30 += r_shape(r_cell, 3)*z_shape(z_cell, 0)*J_z_m1_scal
                J_z_m0_31 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_z_m0_scal
                J_z_m1_31 += r_shape(r_cell, 3)*z_shape(z_cell, 1)*J_z_m1_scal
                J_z_m0_32 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_z_m0_scal
                J_z_m1_32 += r_shape(r_cell, 3)*z_shape(z_cell, 2)*J_z_m1_scal
                J_z_m0_33 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_z_m0_scal
                J_z_m1_33 += r_shape(r_cell, 3)*z_shape(z_cell, 3)*J_z_m1_scal

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

        cuda.atomic.add(j_r_m0.real, (iz_cell - 1 + szl, ir_cell - 1 + srl), J_r_m0_00)
        cuda.atomic.add(j_r_m1.real, (iz_cell - 1 + szl, ir_cell - 1 + srl), J_r_m1_00.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell - 1 + szl, ir_cell - 1 + srl), J_r_m1_00.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell, ir_cell - 1 + srl), J_r_m0_01)
        cuda.atomic.add(j_r_m1.real, (iz_cell, ir_cell - 1 + srl), J_r_m1_01.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell, ir_cell - 1 + srl), J_r_m1_01.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell + 1 + szu, ir_cell - 1 + srl), J_r_m0_02)
        cuda.atomic.add(j_r_m1.real, (iz_cell + 1 + szu, ir_cell - 1 + srl), J_r_m1_02.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell + 1 + szu, ir_cell - 1 + srl), J_r_m1_02.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell + 2 + szu2, ir_cell - 1 + srl), J_r_m0_03)
        cuda.atomic.add(j_r_m1.real, (iz_cell + 2 + szu2, ir_cell - 1 + srl), J_r_m1_03.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell + 2 + szu2, ir_cell - 1 + srl), J_r_m1_03.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell - 1 + szl, ir_cell ), J_r_m0_10)
        cuda.atomic.add(j_r_m1.real, (iz_cell - 1 + szl, ir_cell ), J_r_m1_10.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell - 1 + szl, ir_cell ), J_r_m1_10.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell, ir_cell), J_r_m0_11)
        cuda.atomic.add(j_r_m1.real, (iz_cell, ir_cell), J_r_m1_11.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell, ir_cell), J_r_m1_11.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell + 1 + szu, ir_cell), J_r_m0_12)
        cuda.atomic.add(j_r_m1.real, (iz_cell + 1 + szu, ir_cell), J_r_m1_12.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell + 1 + szu, ir_cell), J_r_m1_12.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell + 2 + szu2, ir_cell), J_r_m0_13)
        cuda.atomic.add(j_r_m1.real, (iz_cell + 2 + szu2, ir_cell), J_r_m1_13.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell + 2 + szu2, ir_cell), J_r_m1_13.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell - 1 + szl, ir_cell + 1 + sru), J_r_m0_20)
        cuda.atomic.add(j_r_m1.real, (iz_cell - 1 + szl, ir_cell + 1 + sru), J_r_m1_20.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell - 1 + szl, ir_cell + 1 + sru), J_r_m1_20.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell, ir_cell + 1 + sru), J_r_m0_21)
        cuda.atomic.add(j_r_m1.real, (iz_cell, ir_cell + 1 + sru), J_r_m1_21.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell, ir_cell + 1 + sru), J_r_m1_21.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell + 1 + szu, ir_cell + 1 + sru), J_r_m0_22)
        cuda.atomic.add(j_r_m1.real, (iz_cell + 1 + szu, ir_cell + 1 + sru), J_r_m1_22.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell + 1 + szu, ir_cell + 1 + sru), J_r_m1_22.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell + 2 + szu2, ir_cell + 1 + sru), J_r_m0_23)
        cuda.atomic.add(j_r_m1.real, (iz_cell + 2 + szu2, ir_cell + 1 + sru), J_r_m1_23.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell + 2 + szu2, ir_cell + 1 + sru), J_r_m1_23.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell - 1 + szl, ir_cell + 2 + sru2), J_r_m0_30)
        cuda.atomic.add(j_r_m1.real, (iz_cell - 1 + szl, ir_cell + 2 + sru2), J_r_m1_30.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell - 1 + szl, ir_cell + 2 + sru2), J_r_m1_30.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell, ir_cell + 2 + sru2), J_r_m0_31)
        cuda.atomic.add(j_r_m1.real, (iz_cell, ir_cell + 2 + sru2), J_r_m1_31.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell, ir_cell + 2 + sru2), J_r_m1_31.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell + 1 + szu, ir_cell + 2 + sru2), J_r_m0_32)
        cuda.atomic.add(j_r_m1.real, (iz_cell + 1 + szu, ir_cell + 2 + sru2), J_r_m1_32.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell + 1 + szu, ir_cell + 2 + sru2), J_r_m1_32.imag)

        cuda.atomic.add(j_r_m0.real, (iz_cell + 2 + szu2, ir_cell + 2 + sru2), J_r_m0_33)
        cuda.atomic.add(j_r_m1.real, (iz_cell + 2 + szu2, ir_cell + 2 + sru2), J_r_m1_33.real)
        cuda.atomic.add(j_r_m1.imag, (iz_cell + 2 + szu2, ir_cell + 2 + sru2), J_r_m1_33.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell - 1 + szl, ir_cell - 1 + srl), J_t_m0_00)
        cuda.atomic.add(j_t_m1.real, (iz_cell - 1 + szl, ir_cell - 1 + srl), J_t_m1_00.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell - 1 + szl, ir_cell - 1 + srl), J_t_m1_00.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell, ir_cell - 1 + srl), J_t_m0_01)
        cuda.atomic.add(j_t_m1.real, (iz_cell, ir_cell - 1 + srl), J_t_m1_01.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell, ir_cell - 1 + srl), J_t_m1_01.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell + 1 + szu, ir_cell - 1 + srl), J_t_m0_02)
        cuda.atomic.add(j_t_m1.real, (iz_cell + 1 + szu, ir_cell - 1 + srl), J_t_m1_02.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell + 1 + szu, ir_cell - 1 + srl), J_t_m1_02.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell + 2 + szu2, ir_cell - 1 + srl), J_t_m0_03)
        cuda.atomic.add(j_t_m1.real, (iz_cell + 2 + szu2, ir_cell - 1 + srl), J_t_m1_03.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell + 2 + szu2, ir_cell - 1 + srl), J_t_m1_03.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell - 1 + szl, ir_cell ), J_t_m0_10)
        cuda.atomic.add(j_t_m1.real, (iz_cell - 1 + szl, ir_cell ), J_t_m1_10.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell - 1 + szl, ir_cell ), J_t_m1_10.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell, ir_cell), J_t_m0_11)
        cuda.atomic.add(j_t_m1.real, (iz_cell, ir_cell), J_t_m1_11.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell, ir_cell), J_t_m1_11.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell + 1 + szu, ir_cell), J_t_m0_12)
        cuda.atomic.add(j_t_m1.real, (iz_cell + 1 + szu, ir_cell), J_t_m1_12.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell + 1 + szu, ir_cell), J_t_m1_12.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell + 2 + szu2, ir_cell), J_t_m0_13)
        cuda.atomic.add(j_t_m1.real, (iz_cell + 2 + szu2, ir_cell), J_t_m1_13.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell + 2 + szu2, ir_cell), J_t_m1_13.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell - 1 + szl, ir_cell + 1 + sru), J_t_m0_20)
        cuda.atomic.add(j_t_m1.real, (iz_cell - 1 + szl, ir_cell + 1 + sru), J_t_m1_20.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell - 1 + szl, ir_cell + 1 + sru), J_t_m1_20.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell, ir_cell + 1 + sru), J_t_m0_21)
        cuda.atomic.add(j_t_m1.real, (iz_cell, ir_cell + 1 + sru), J_t_m1_21.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell, ir_cell + 1 + sru), J_t_m1_21.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell + 1 + szu, ir_cell + 1 + sru), J_t_m0_22)
        cuda.atomic.add(j_t_m1.real, (iz_cell + 1 + szu, ir_cell + 1 + sru), J_t_m1_22.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell + 1 + szu, ir_cell + 1 + sru), J_t_m1_22.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell + 2 + szu2, ir_cell + 1 + sru), J_t_m0_23)
        cuda.atomic.add(j_t_m1.real, (iz_cell + 2 + szu2, ir_cell + 1 + sru), J_t_m1_23.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell + 2 + szu2, ir_cell + 1 + sru), J_t_m1_23.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell - 1 + szl, ir_cell + 2 + sru2), J_t_m0_30)
        cuda.atomic.add(j_t_m1.real, (iz_cell - 1 + szl, ir_cell + 2 + sru2), J_t_m1_30.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell - 1 + szl, ir_cell + 2 + sru2), J_t_m1_30.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell, ir_cell + 2 + sru2), J_t_m0_31)
        cuda.atomic.add(j_t_m1.real, (iz_cell, ir_cell + 2 + sru2), J_t_m1_31.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell, ir_cell + 2 + sru2), J_t_m1_31.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell + 1 + szu, ir_cell + 2 + sru2), J_t_m0_32)
        cuda.atomic.add(j_t_m1.real, (iz_cell + 1 + szu, ir_cell + 2 + sru2), J_t_m1_32.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell + 1 + szu, ir_cell + 2 + sru2), J_t_m1_32.imag)

        cuda.atomic.add(j_t_m0.real, (iz_cell + 2 + szu2, ir_cell + 2 + sru2), J_t_m0_33)
        cuda.atomic.add(j_t_m1.real, (iz_cell + 2 + szu2, ir_cell + 2 + sru2), J_t_m1_33.real)
        cuda.atomic.add(j_t_m1.imag, (iz_cell + 2 + szu2, ir_cell + 2 + sru2), J_t_m1_33.imag)


        cuda.atomic.add(j_z_m0.real, (iz_cell - 1 + szl, ir_cell - 1 + srl), J_z_m0_00)
        cuda.atomic.add(j_z_m1.real, (iz_cell - 1 + szl, ir_cell - 1 + srl), J_z_m1_00.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell - 1 + szl, ir_cell - 1 + srl), J_z_m1_00.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell, ir_cell - 1 + srl), J_z_m0_01)
        cuda.atomic.add(j_z_m1.real, (iz_cell, ir_cell - 1 + srl), J_z_m1_01.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell, ir_cell - 1 + srl), J_z_m1_01.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell + 1 + szu, ir_cell - 1 + srl), J_z_m0_02)
        cuda.atomic.add(j_z_m1.real, (iz_cell + 1 + szu, ir_cell - 1 + srl), J_z_m1_02.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell + 1 + szu, ir_cell - 1 + srl), J_z_m1_02.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell + 2 + szu2, ir_cell - 1 + srl), J_z_m0_03)
        cuda.atomic.add(j_z_m1.real, (iz_cell + 2 + szu2, ir_cell - 1 + srl), J_z_m1_03.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell + 2 + szu2, ir_cell - 1 + srl), J_z_m1_03.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell - 1 + szl, ir_cell ), J_z_m0_10)
        cuda.atomic.add(j_z_m1.real, (iz_cell - 1 + szl, ir_cell ), J_z_m1_10.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell - 1 + szl, ir_cell ), J_z_m1_10.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell, ir_cell), J_z_m0_11)
        cuda.atomic.add(j_z_m1.real, (iz_cell, ir_cell), J_z_m1_11.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell, ir_cell), J_z_m1_11.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell + 1 + szu, ir_cell), J_z_m0_12)
        cuda.atomic.add(j_z_m1.real, (iz_cell + 1 + szu, ir_cell), J_z_m1_12.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell + 1 + szu, ir_cell), J_z_m1_12.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell + 2 + szu2, ir_cell), J_z_m0_13)
        cuda.atomic.add(j_z_m1.real, (iz_cell + 2 + szu2, ir_cell), J_z_m1_13.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell + 2 + szu2, ir_cell), J_z_m1_13.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell - 1 + szl, ir_cell + 1 + sru), J_z_m0_20)
        cuda.atomic.add(j_z_m1.real, (iz_cell - 1 + szl, ir_cell + 1 + sru), J_z_m1_20.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell - 1 + szl, ir_cell + 1 + sru), J_z_m1_20.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell, ir_cell + 1 + sru), J_z_m0_21)
        cuda.atomic.add(j_z_m1.real, (iz_cell, ir_cell + 1 + sru), J_z_m1_21.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell, ir_cell + 1 + sru), J_z_m1_21.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell + 1 + szu, ir_cell + 1 + sru), J_z_m0_22)
        cuda.atomic.add(j_z_m1.real, (iz_cell + 1 + szu, ir_cell + 1 + sru), J_z_m1_22.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell + 1 + szu, ir_cell + 1 + sru), J_z_m1_22.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell + 2 + szu2, ir_cell + 1 + sru), J_z_m0_23)
        cuda.atomic.add(j_z_m1.real, (iz_cell + 2 + szu2, ir_cell + 1 + sru), J_z_m1_23.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell + 2 + szu2, ir_cell + 1 + sru), J_z_m1_23.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell - 1 + szl, ir_cell + 2 + sru2), J_z_m0_30)
        cuda.atomic.add(j_z_m1.real, (iz_cell - 1 + szl, ir_cell + 2 + sru2), J_z_m1_30.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell - 1 + szl, ir_cell + 2 + sru2), J_z_m1_30.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell, ir_cell + 2 + sru2), J_z_m0_31)
        cuda.atomic.add(j_z_m1.real, (iz_cell, ir_cell + 2 + sru2), J_z_m1_31.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell, ir_cell + 2 + sru2), J_z_m1_31.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell + 1 + szu, ir_cell + 2 + sru2), J_z_m0_32)
        cuda.atomic.add(j_z_m1.real, (iz_cell + 1 + szu, ir_cell + 2 + sru2), J_z_m1_32.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell + 1 + szu, ir_cell + 2 + sru2), J_z_m1_32.imag)

        cuda.atomic.add(j_z_m0.real, (iz_cell + 2 + szu2, ir_cell + 2 + sru2), J_z_m0_33)
        cuda.atomic.add(j_z_m1.real, (iz_cell + 2 + szu2, ir_cell + 2 + sru2), J_z_m1_33.real)
        cuda.atomic.add(j_z_m1.imag, (iz_cell + 2 + szu2, ir_cell + 2 + sru2), J_z_m1_33.imag)
