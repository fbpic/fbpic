# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the deposition methods for rho and J for linear order shapes
without using atomic operations.
"""
from numba import cuda
import math
from scipy.constants import c
import numpy as np

# -------------------------------
# Field deposition utility - rho
# -------------------------------


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], \
                float64, float64, int32, \
                float64, float64, int32, \
                complex128[:,:,:], complex128[:,:,:], \
                complex128[:,:,:], complex128[:,:,:],\
                int32[:], int32[:])')
def deposit_rho_gpu(x, y, z, w,
                    invdz, zmin, Nz,
                    invdr, rmin, Nr,
                    rho0, rho1,
                    rho2, rho3,
                    cell_idx, prefix_sum):
    """
    Deposition of the charge density rho using numba on the GPU.
    Iterates over the cells and over the particles per cell.
    Calculates the weighted amount of rho that is deposited to the
    4 cells surounding the particle based on its shape (linear).

    The particles are sorted by their cell index (the lower cell
    in r and z that they deposit to) and the deposited field
    is split into 4 arrays (one for each possible direction,
    e.g. upper in z, lower in r) to maintain parallelism while
    avoiding any race conditions.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    w : 1d array of floats
        The weights of the particles

    rho0, rho1, rho2, rho3 : 3darray of complexs
        2d field arrays, one for each of the deposition directions
        The third dimension contains the two possible modes.
        (is modified by this function)

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
        iz = int(i / Nr)
        ir = int(i - iz * Nr)
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
        # Initialize the local field value for
        # all four possible deposition directions
        # Mode 0, 1 for r, t, z
        # 1 : lower in r, lower in z
        # 2 : lower in r, upper in z
        # 3 : upper in r, lower in z
        # 4 : upper in r, upper in z
        R1_m0 = 0. + 0.j
        R2_m0 = 0. + 0.j
        R3_m0 = 0. + 0.j
        R4_m0 = 0. + 0.j
        # ------------
        R1_m1 = 0. + 0.j
        R2_m1 = 0. + 0.j
        R3_m1 = 0. + 0.j
        R4_m1 = 0. + 0.j
        # Loop over the number of particles per cell
        for j in range(frequency_per_cell):
            # Get the particle index before the sorting
            # --------------------------------------------
            # (Since incl_offset is a cumulative sum of particle number,
            # and since python index starts at 0, one has to add -1)
            ptcl_idx = incl_offset - 1 - j

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
                invr = 1. / rj
                cos = xj * invr  # Cosine
                sin = yj * invr  # Sine
            else:
                cos = 1.
                sin = 0.
            exptheta_m0 = 1.
            exptheta_m1 = cos + 1.j * sin

            # Get linear weights for the deposition
            # --------------------------------------------
            # Positions of the particles, in the cell unit
            r_cell = invdr * (rj - rmin) - 0.5
            z_cell = invdz * (zj - zmin) - 0.5
            # Original index of the uppper and lower cell
            ir_lower = int(math.floor(r_cell))
            ir_upper = ir_lower + 1
            iz_lower = int(math.floor(z_cell))
            iz_upper = iz_lower + 1
            # Linear weight
            Sr_lower = ir_upper - r_cell
            Sr_upper = r_cell - ir_lower
            Sz_lower = iz_upper - z_cell
            Sz_upper = z_cell - iz_lower
            # Set guard weights to zero
            Sr_guard = 0.

            # Treat the boundary conditions
            # --------------------------------------------
            # guard cells in lower r
            if ir_lower < 0:
                Sr_guard = Sr_lower
                Sr_lower = 0.
                ir_lower = 0
            # absorbing in upper r
            if ir_lower > Nr - 1:
                ir_lower = Nr - 1
            if ir_upper > Nr - 1:
                ir_upper = Nr - 1
            # periodic boundaries in z
            # lower z boundaries
            if iz_lower < 0:
                iz_lower += Nz
            if iz_upper < 0:
                iz_upper += Nz
            # upper z boundaries
            if iz_lower > Nz - 1:
                iz_lower -= Nz
            if iz_upper > Nz - 1:
                iz_upper -= Nz

            # Calculate rho
            # --------------------------------------------
            # Mode 0
            R_m0 = wj * exptheta_m0
            # Mode 1
            R_m1 = wj * exptheta_m1

            # Caculate the weighted currents for each
            # of the four possible direction
            # --------------------------------------------
            if ir_lower == ir_upper:
                # In the case that ir_lower and ir_upper are equal,
                # the current is added only to the array corresponding
                # to ir_lower.
                # (This is the case for the boundaries in r)
                R1_m0 += Sz_lower * Sr_lower * R_m0
                R1_m0 += Sz_lower * Sr_upper * R_m0
                R3_m0 += Sz_upper * Sr_lower * R_m0
                R3_m0 += Sz_upper * Sr_upper * R_m0
                # -----------------------------
                R1_m1 += Sz_lower * Sr_lower * R_m1
                R1_m1 += Sz_lower * Sr_upper * R_m1
                R3_m1 += Sz_upper * Sr_lower * R_m1
                R3_m1 += Sz_upper * Sr_upper * R_m1
                # -----------------------------
            if ir_lower != ir_upper:
                # In the case that ir_lower and ir_upper are different,
                # add the current to the four arrays according to
                # the direction.
                R1_m0 += Sz_lower * Sr_lower * R_m0
                R2_m0 += Sz_lower * Sr_upper * R_m0
                R3_m0 += Sz_upper * Sr_lower * R_m0
                R4_m0 += Sz_upper * Sr_upper * R_m0
                # -----------------------------
                R1_m1 += Sz_lower * Sr_lower * R_m1
                R2_m1 += Sz_lower * Sr_upper * R_m1
                R3_m1 += Sz_upper * Sr_lower * R_m1
                R4_m1 += Sz_upper * Sr_upper * R_m1
                # -----------------------------
            if ir_lower == ir_upper == 0:
                # Treat the guard cells.
                # Add the current to the guard cells
                # for particles that had an original
                # cell index < 0.
                R1_m0 += -1. * Sz_lower * Sr_guard * R_m0
                R3_m0 += -1. * Sz_upper * Sr_guard * R_m0
                # ---------------------------------
                R1_m1 += -1. * Sz_lower * Sr_guard * R_m1
                R3_m1 += -1. * Sz_upper * Sr_guard * R_m1
        # Write the calculated field values to
        # the field arrays defined on the interpolation grid
        rho0[iz, ir, 0] = R1_m0
        rho0[iz, ir, 1] = R1_m1
        rho1[iz, ir, 0] = R2_m0
        rho1[iz, ir, 1] = R2_m1
        rho2[iz, ir, 0] = R3_m0
        rho2[iz, ir, 1] = R3_m1
        rho3[iz, ir, 0] = R4_m0
        rho3[iz, ir, 1] = R4_m1


@cuda.jit('void(complex128[:,:], complex128[:,:], \
                complex128[:,:,:], complex128[:,:,:], \
                complex128[:,:,:], complex128[:,:,:])')
def add_rho(rho_m0, rho_m1,
            rho0, rho1,
            rho2, rho3):
    """
    Merges the 4 separate field arrays that contain rho for
    each deposition direction and adds them to the global
    interpolation grid arrays for mode 0 and 1.

    Parameters
    ----------
    rho_m0, rho_m1 : 2darrays of complexs
        The charge density on the interpolation grid for
        mode 0 and 1. (is modified by this function)

    rho0, rho1, rho2, rho3 : 3darrays of complexs
        2d field arrays, one for each of the deposition directions
        The third dimension contains the two possible modes.
    """
    # Get the CUDA Grid in 2D
    i, j = cuda.grid(2)
    # Only for threads within (nz, nr)
    if (i < rho_m0.shape[0] and j < rho_m0.shape[1]):
        # Sum the four field arrays for the different deposition
        # directions and write them to the global field array
        rho_m0[i, j] += rho0[i, j, 0] + \
            rho1[i, j - 1, 0] + \
            rho2[i - 1, j, 0] + \
            rho3[i - 1, j - 1, 0]

        rho_m1[i, j] += rho0[i, j, 1] + \
            rho1[i, j - 1, 1] + \
            rho2[i - 1, j, 1] + \
            rho3[i - 1, j - 1, 1]


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], \
                float64, float64, int32, \
                float64, float64, int32, \
                complex128[:,:,:], complex128[:,:,:], \
                complex128[:,:,:], complex128[:,:,:],\
                int32[:], int32[:])')
def deposit_J_gpu(x, y, z, w,
                  ux, uy, uz, inv_gamma,
                  invdz, zmin, Nz,
                  invdr, rmin, Nr,
                  J0, J1,
                  J2, J3,
                  cell_idx, prefix_sum):
    """
    Deposition of the current J using numba on the GPU.
    Iterates over the cells and over the particles per cell.
    Calculates the weighted amount of J that is deposited to the
    4 cells surounding the particle based on its shape (linear).

    The particles are sorted by their cell index (the lower cell
    in r and z that they deposit to) and the deposited field
    is split into 4 arrays (one for each possible direction,
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

    J0, J1, J2, J3 : 3darray of complexs
        2d field arrays, one for each of the deposition directions
        The third dimension contains the two possible modes and the
        3 directions of J in cylindrical coordinates (r, t, z).
        (is mofidied by this function)

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
        iz = int(i / Nr)
        ir = int(i - iz * Nr)
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
        # Initialize the local field value for
        # all four possible deposition directions
        # Mode 0, 1 for r, t, z
        # 1 : lower in r, lower in z
        # 2 : lower in r, upper in z
        # 3 : upper in r, lower in z
        # 4 : upper in r, upper in z
        Jr1_m0 = 0. + 0.j
        Jr2_m0 = 0. + 0.j
        Jr3_m0 = 0. + 0.j
        Jr4_m0 = 0. + 0.j
        # -------------
        Jr1_m1 = 0. + 0.j
        Jr2_m1 = 0. + 0.j
        Jr3_m1 = 0. + 0.j
        Jr4_m1 = 0. + 0.j
        # -------------
        Jt1_m0 = 0. + 0.j
        Jt2_m0 = 0. + 0.j
        Jt3_m0 = 0. + 0.j
        Jt4_m0 = 0. + 0.j
        # -------------
        Jt1_m1 = 0. + 0.j
        Jt2_m1 = 0. + 0.j
        Jt3_m1 = 0. + 0.j
        Jt4_m1 = 0. + 0.j
        # -------------
        Jz1_m0 = 0. + 0.j
        Jz2_m0 = 0. + 0.j
        Jz3_m0 = 0. + 0.j
        Jz4_m0 = 0. + 0.j
        # -------------
        Jz1_m1 = 0. + 0.j
        Jz2_m1 = 0. + 0.j
        Jz3_m1 = 0. + 0.j
        Jz4_m1 = 0. + 0.j
        # Loop over the number of particles per cell
        for j in range(frequency_per_cell):
            # Get the particle index
            # ----------------------
            # (Since incl_offset is a cumulative sum of particle number,
            # and since python index starts at 0, one has to add -1)
            ptcl_idx = incl_offset - 1 - j

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
                invr = 1. / rj
                cos = xj * invr  # Cosine
                sin = yj * invr  # Sine
            else:
                cos = 1.
                sin = 0.
            exptheta_m0 = 1.
            exptheta_m1 = cos + 1.j * sin

            # Get linear weights for the deposition
            # --------------------------------------------
            # Positions of the particles, in the cell unit
            r_cell = invdr * (rj - rmin) - 0.5
            z_cell = invdz * (zj - zmin) - 0.5
            # Original index of the uppper and lower cell
            # in r and z
            ir_lower = int(math.floor(r_cell))
            ir_upper = ir_lower + 1
            iz_lower = int(math.floor(z_cell))
            iz_upper = iz_lower + 1
            # Linear weight
            Sr_lower = ir_upper - r_cell
            Sr_upper = r_cell - ir_lower
            Sz_lower = iz_upper - z_cell
            Sz_upper = z_cell - iz_lower
            # Set guard weights to zero
            Sr_guard = 0.

            # Treat the boundary conditions
            # --------------------------------------------
            # guard cells in lower r
            if ir_lower < 0:
                Sr_guard = Sr_lower
                Sr_lower = 0.
                ir_lower = 0
            # absorbing in upper r
            if ir_lower > Nr - 1:
                ir_lower = Nr - 1
            if ir_upper > Nr - 1:
                ir_upper = Nr - 1
            # periodic boundaries in z
            # lower z boundaries
            if iz_lower < 0:
                iz_lower += Nz
            if iz_upper < 0:
                iz_upper += Nz
            # upper z boundaries
            if iz_lower > Nz - 1:
                iz_lower -= Nz
            if iz_upper > Nz - 1:
                iz_upper -= Nz

            # Calculate the currents
            # --------------------------------------------
            # Mode 0
            Jr_m0 = wj * c * inv_gammaj * (cos * uxj + sin * uyj) * exptheta_m0
            Jt_m0 = wj * c * inv_gammaj * (cos * uyj - sin * uxj) * exptheta_m0
            Jz_m0 = wj * c * inv_gammaj * uzj * exptheta_m0
            # Mode 1
            Jr_m1 = wj * c * inv_gammaj * (cos * uxj + sin * uyj) * exptheta_m1
            Jt_m1 = wj * c * inv_gammaj * (cos * uyj - sin * uxj) * exptheta_m1
            Jz_m1 = wj * c * inv_gammaj * uzj * exptheta_m1

            # Caculate the weighted currents for each
            # of the four possible direction
            # --------------------------------------------
            if ir_lower == ir_upper:
                # In the case that ir_lower and ir_upper are equal,
                # the current is added only to the array corresponding
                # to ir_lower.
                # (This is the case for the boundaries in r)
                Jr1_m0 += Sz_lower * Sr_lower * Jr_m0
                Jr1_m0 += Sz_lower * Sr_upper * Jr_m0
                Jr3_m0 += Sz_upper * Sr_lower * Jr_m0
                Jr3_m0 += Sz_upper * Sr_upper * Jr_m0
                # -------------------------------
                Jr1_m1 += Sz_lower * Sr_lower * Jr_m1
                Jr1_m1 += Sz_lower * Sr_upper * Jr_m1
                Jr3_m1 += Sz_upper * Sr_lower * Jr_m1
                Jr3_m1 += Sz_upper * Sr_upper * Jr_m1
                # -------------------------------
                Jt1_m0 += Sz_lower * Sr_lower * Jt_m0
                Jt1_m0 += Sz_lower * Sr_upper * Jt_m0
                Jt3_m0 += Sz_upper * Sr_lower * Jt_m0
                Jt3_m0 += Sz_upper * Sr_upper * Jt_m0
                # -------------------------------
                Jt1_m1 += Sz_lower * Sr_lower * Jt_m1
                Jt1_m1 += Sz_lower * Sr_upper * Jt_m1
                Jt3_m1 += Sz_upper * Sr_lower * Jt_m1
                Jt3_m1 += Sz_upper * Sr_upper * Jt_m1
                # -------------------------------
                Jz1_m0 += Sz_lower * Sr_lower * Jz_m0
                Jz1_m0 += Sz_lower * Sr_upper * Jz_m0
                Jz3_m0 += Sz_upper * Sr_lower * Jz_m0
                Jz3_m0 += Sz_upper * Sr_upper * Jz_m0
                # -------------------------------
                Jz1_m1 += Sz_lower * Sr_lower * Jz_m1
                Jz1_m1 += Sz_lower * Sr_upper * Jz_m1
                Jz3_m1 += Sz_upper * Sr_lower * Jz_m1
                Jz3_m1 += Sz_upper * Sr_upper * Jz_m1
                # -------------------------------
            if ir_lower != ir_upper:
                # In the case that ir_lower and ir_upper are different,
                # add the current to the four arrays according to
                # the direction.
                Jr1_m0 += Sz_lower * Sr_lower * Jr_m0
                Jr2_m0 += Sz_lower * Sr_upper * Jr_m0
                Jr3_m0 += Sz_upper * Sr_lower * Jr_m0
                Jr4_m0 += Sz_upper * Sr_upper * Jr_m0
                # -------------------------------
                Jr1_m1 += Sz_lower * Sr_lower * Jr_m1
                Jr2_m1 += Sz_lower * Sr_upper * Jr_m1
                Jr3_m1 += Sz_upper * Sr_lower * Jr_m1
                Jr4_m1 += Sz_upper * Sr_upper * Jr_m1
                # -------------------------------
                Jt1_m0 += Sz_lower * Sr_lower * Jt_m0
                Jt2_m0 += Sz_lower * Sr_upper * Jt_m0
                Jt3_m0 += Sz_upper * Sr_lower * Jt_m0
                Jt4_m0 += Sz_upper * Sr_upper * Jt_m0
                # -------------------------------
                Jt1_m1 += Sz_lower * Sr_lower * Jt_m1
                Jt2_m1 += Sz_lower * Sr_upper * Jt_m1
                Jt3_m1 += Sz_upper * Sr_lower * Jt_m1
                Jt4_m1 += Sz_upper * Sr_upper * Jt_m1
                # -------------------------------
                Jz1_m0 += Sz_lower * Sr_lower * Jz_m0
                Jz2_m0 += Sz_lower * Sr_upper * Jz_m0
                Jz3_m0 += Sz_upper * Sr_lower * Jz_m0
                Jz4_m0 += Sz_upper * Sr_upper * Jz_m0
                # -------------------------------
                Jz1_m1 += Sz_lower * Sr_lower * Jz_m1
                Jz2_m1 += Sz_lower * Sr_upper * Jz_m1
                Jz3_m1 += Sz_upper * Sr_lower * Jz_m1
                Jz4_m1 += Sz_upper * Sr_upper * Jz_m1
                # -------------------------------
            if ir_lower == ir_upper == 0:
                # Treat the guard cells.
                # Add the current to the guard cells
                # for particles that had an original
                # cell index < 0.
                Jr1_m0 += -1. * Sz_lower * Sr_guard * Jr_m0
                Jr3_m0 += -1. * Sz_upper * Sr_guard * Jr_m0
                # -----------------------------------
                Jr1_m1 += -1. * Sz_lower * Sr_guard * Jr_m1
                Jr3_m1 += -1. * Sz_upper * Sr_guard * Jr_m1
                # -----------------------------------
                Jt1_m0 += -1. * Sz_lower * Sr_guard * Jt_m0
                Jt3_m0 += -1. * Sz_upper * Sr_guard * Jt_m0
                # -----------------------------------
                Jt1_m1 += -1. * Sz_lower * Sr_guard * Jt_m1
                Jt3_m1 += -1. * Sz_upper * Sr_guard * Jt_m1
                # -----------------------------------
                Jz1_m0 += -1. * Sz_lower * Sr_guard * Jz_m0
                Jz3_m0 += -1. * Sz_upper * Sr_guard * Jz_m0
                # -----------------------------------
                Jz1_m1 += -1. * Sz_lower * Sr_guard * Jz_m1
                Jz3_m1 += -1. * Sz_upper * Sr_guard * Jz_m1
        # Write the calculated field values to
        # the field arrays defined on the interpolation grid
        J0[iz, ir, 0] = Jr1_m0
        J0[iz, ir, 1] = Jr1_m1
        J0[iz, ir, 2] = Jt1_m0
        J0[iz, ir, 3] = Jt1_m1
        J0[iz, ir, 4] = Jz1_m0
        J0[iz, ir, 5] = Jz1_m1
        # --------------------
        J1[iz, ir, 0] = Jr2_m0
        J1[iz, ir, 1] = Jr2_m1
        J1[iz, ir, 2] = Jt2_m0
        J1[iz, ir, 3] = Jt2_m1
        J1[iz, ir, 4] = Jz2_m0
        J1[iz, ir, 5] = Jz2_m1
        # --------------------
        J2[iz, ir, 0] = Jr3_m0
        J2[iz, ir, 1] = Jr3_m1
        J2[iz, ir, 2] = Jt3_m0
        J2[iz, ir, 3] = Jt3_m1
        J2[iz, ir, 4] = Jz3_m0
        J2[iz, ir, 5] = Jz3_m1
        # --------------------
        J3[iz, ir, 0] = Jr4_m0
        J3[iz, ir, 1] = Jr4_m1
        J3[iz, ir, 2] = Jt4_m0
        J3[iz, ir, 3] = Jt4_m1
        J3[iz, ir, 4] = Jz4_m0
        J3[iz, ir, 5] = Jz4_m1


@cuda.jit('void(complex128[:,:], complex128[:,:], \
                complex128[:,:], complex128[:,:], \
                complex128[:,:], complex128[:,:], \
                complex128[:,:,:], complex128[:,:,:], \
                complex128[:,:,:], complex128[:,:,:])')
def add_J(Jr_m0, Jr_m1,
          Jt_m0, Jt_m1,
          Jz_m0, Jz_m1,
          J0, J1,
          J2, J3):
    """
    Merges the 4 separate field arrays that contain J for
    each deposition direction and adds them to the global
    interpolation grid arrays for mode 0 and 1.

    Parameters
    ----------
    Jr_m0, Jr_m1, Jt_m0, Jt_m1, Jz_m0, Jz_m1,: 2darrays of complexs
        The current component in each direction (r, t, z)
        on the interpolation grid for mode 0 and 1.
        (is modified by this function)

    J0, J1, J2, J3 : 3darrays of complexs
        2d field arrays, one for each of the deposition directions
        The third dimension contains the two possible modes and
        the 3 different components of J (r, t, z).
    """
    # Get the CUDA Grid in 2D
    i, j = cuda.grid(2)
    # Only for threads within (nz, nr)
    if (i < Jr_m0.shape[0] and j < Jr_m0.shape[1]):
        # Sum the four field arrays for the different deposition
        # directions and write them to the global field array
        Jr_m0[i, j] += J0[i, j, 0] + \
            J1[i, j - 1, 0] + \
            J2[i - 1, j, 0] + \
            J3[i - 1, j - 1, 0]

        Jr_m1[i, j] += J0[i, j, 1] + \
            J1[i, j - 1, 1] + \
            J2[i - 1, j, 1] + \
            J3[i - 1, j - 1, 1]

        Jt_m0[i, j] += J0[i, j, 2] + \
            J1[i, j - 1, 2] + \
            J2[i - 1, j, 2] + \
            J3[i - 1, j - 1, 2]

        Jt_m1[i, j] += J0[i, j, 3] + \
            J1[i, j - 1, 3] + \
            J2[i - 1, j, 3] + \
            J3[i - 1, j - 1, 3]

        Jz_m0[i, j] += J0[i, j, 4] + \
            J1[i, j - 1, 4] + \
            J2[i - 1, j, 4] + \
            J3[i - 1, j - 1, 4]

        Jz_m1[i, j] += J0[i, j, 5] + \
            J1[i, j - 1, 5] + \
            J2[i - 1, j, 5] + \
            J3[i - 1, j - 1, 5]
