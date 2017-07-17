# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the deposition methods for rho and J for linear and cubic
order shapes on the CPU with threading.
"""
import numba
from numba import prange, int64
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

@numba.njit(parallel=True)
def deposit_rho_prange_linear(x, y, z, w,
                           invdz, zmin, Nz,
                           invdr, rmin, Nr,
                           rho_m0_global, rho_m1_global,
                           nthreads, tx_chunks, tx_N):
    """
    Deposition of the charge density rho using numba prange on the CPU.
    Iterates over the threads in parallel, while each thread iterates
    over a batch of particles. Intermediate results for each threads are
    stored in copies of the global grid. At the end of the parallel loop,
    the thread-local field arrays are combined (summed) to the global array.

    Calculates the weighted amount of rho that is deposited to the
    4 cells surounding the particle based on its shape (linear).

    The rest of the execution is similar to the CUDA equivalent function.

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
    """
    # Deposit the field per cell in parallel (for threads < number of cells)
    for tx in prange( nthreads ):
        # Create thread_local helper arrays
        # FIXME! ( instead of using zeros_like,
        # it would be nicer to use np.zeros((Nz,Nr)) )
        # Loop over all particles in thread chunk
        for idx in range( tx_chunks[tx] ):
            # Calculate thread local particle index
            ptcl_idx = idx + tx*tx_N
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

            R_m0_00 = r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * R_m0_scal
            R_m0_01 = r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * R_m0_scal
            R_m1_00 = r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * R_m1_scal
            R_m1_01 = r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * R_m1_scal

            if ir_flip == -1:
                R_m0_00 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * R_m0_scal
                R_m0_01 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * R_m0_scal
                R_m1_00 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * R_m1_scal
                R_m1_01 += r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * R_m1_scal
            else:
                R_m0_10 = r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * R_m0_scal
                R_m0_11 = r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * R_m0_scal
                R_m1_10 = r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * R_m1_scal
                R_m1_11 = r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * R_m1_scal

            # Cell shifts for the simulation boundaries
            shift_r = 0
            shift_z = 0
            if ir_cell+1 > (Nr-1):
                shift_r = -1
            if iz_cell+1 > Nz-1:
                shift_z -= Nz
            # Write to thread local arrays
            rho_m0_global[tx, iz_cell, ir_cell] += R_m0_00
            rho_m1_global[tx, iz_cell, ir_cell] += R_m1_00

            rho_m0_global[tx,iz_cell+1 + shift_z, ir_cell] += R_m0_01
            rho_m1_global[tx,iz_cell+1 + shift_z, ir_cell] += R_m1_01

            rho_m0_global[tx,iz_cell, ir_cell+1 + shift_r] += R_m0_10
            rho_m1_global[tx,iz_cell, ir_cell+1 + shift_r] += R_m1_10

            rho_m0_global[tx,iz_cell+1 + shift_z, ir_cell+1 + shift_r] += R_m0_11
            rho_m1_global[tx,iz_cell+1 + shift_z, ir_cell+1 + shift_r] += R_m1_11

    return

# -------------------------------
# Field deposition - linear - J
# -------------------------------

@numba.njit(parallel=True)
def deposit_J_prange_linear(x, y, z, w,
                         ux, uy, uz, inv_gamma,
                         invdz, zmin, Nz,
                         invdr, rmin, Nr,
                         j_r_m0_global, j_r_m1_global,
                         j_t_m0_global, j_t_m1_global,
                         j_z_m0_global, j_z_m1_global,
                         nthreads, tx_chunks, tx_N):
    """
    Deposition of the current density J using numba prange on the CPU.
    Iterates over the threads in parallel, while each thread iterates
    over a batch of particles. Intermediate results for each threads are
    stored in copies of the global grid. At the end of the parallel loop,
    the thread-local field arrays are combined (summed) to the global array.

    Calculates the weighted amount of J that is deposited to the
    4 cells surounding the particle based on its shape (linear).

    The rest of the execution is similar to the CUDA equivalent function.

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
    """
    # Deposit the field per cell in parallel (for threads < number of cells)
    for tx in prange( nthreads ):
        # Create thread_local helper arrays
        # FIXME! ( instead of using zeros_like,
        # it would be nicer to use np.zeros((Nz,Nr)) )
        # Loop over all particles in thread chunk
        for idx in range( tx_chunks[tx] ):
            # Calculate thread local particle index
            ptcl_idx = idx + tx*tx_N
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

            J_r_m0_00 = r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_r_m0_scal
            J_t_m0_00 = r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_t_m0_scal
            J_z_m0_00 = r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_z_m0_scal
            J_r_m0_01 = r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_r_m0_scal
            J_t_m0_01 = r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_t_m0_scal
            J_z_m0_01 = r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_z_m0_scal
            J_r_m1_00 = r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_r_m1_scal
            J_t_m1_00 = r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_t_m1_scal
            J_z_m1_00 = r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 0) * J_z_m1_scal
            J_r_m1_01 = r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_r_m1_scal
            J_t_m1_01 = r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_t_m1_scal
            J_z_m1_01 = r_shape_linear(r_cell, 0)*z_shape_linear(z_cell, 1) * J_z_m1_scal

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
                J_r_m0_10 = r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_r_m0_scal
                J_t_m0_10 = r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_t_m0_scal
                J_z_m0_10 = r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_z_m0_scal
                J_r_m0_11 = r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_r_m0_scal
                J_t_m0_11 = r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_t_m0_scal
                J_z_m0_11 = r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_z_m0_scal
                J_r_m1_10 = r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_r_m1_scal
                J_t_m1_10 = r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_t_m1_scal
                J_z_m1_10 = r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 0) * J_z_m1_scal
                J_r_m1_11 = r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_r_m1_scal
                J_t_m1_11 = r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_t_m1_scal
                J_z_m1_11 = r_shape_linear(r_cell, 1)*z_shape_linear(z_cell, 1) * J_z_m1_scal

            # Cell shifts for the simulation boundaries
            shift_r = 0
            shift_z = 0
            if (ir_cell+1) > (Nr-1):
                shift_r = -1
            if (iz_cell+1) > Nz-1:
                shift_z -= Nz

            j_r_m0_global[tx,iz_cell, ir_cell] += J_r_m0_00
            j_r_m1_global[tx,iz_cell, ir_cell] += J_r_m1_00

            j_r_m0_global[tx,iz_cell+1 + shift_z, ir_cell] += J_r_m0_01
            j_r_m1_global[tx,iz_cell+1 + shift_z, ir_cell] += J_r_m1_01

            j_r_m0_global[tx,iz_cell, ir_cell+1 + shift_r] += J_r_m0_10
            j_r_m1_global[tx,iz_cell, ir_cell+1 + shift_r] += J_r_m1_10

            j_r_m0_global[tx,iz_cell+1 + shift_z, ir_cell+1 + shift_r] += J_r_m0_11
            j_r_m1_global[tx,iz_cell+1 + shift_z, ir_cell+1 + shift_r] += J_r_m1_11

            j_t_m0_global[tx,iz_cell, ir_cell] += J_t_m0_00
            j_t_m1_global[tx,iz_cell, ir_cell] += J_t_m1_00

            j_t_m0_global[tx,iz_cell+1 + shift_z, ir_cell] += J_t_m0_01
            j_t_m1_global[tx,iz_cell+1 + shift_z, ir_cell] += J_t_m1_01

            j_t_m0_global[tx,iz_cell, ir_cell+1 + shift_r] += J_t_m0_10
            j_t_m1_global[tx,iz_cell, ir_cell+1 + shift_r] += J_t_m1_10

            j_t_m0_global[tx,iz_cell+1 + shift_z, ir_cell+1 + shift_r] += J_t_m0_11
            j_t_m1_global[tx,iz_cell+1 + shift_z, ir_cell+1 + shift_r] += J_t_m1_11

            j_z_m0_global[tx,iz_cell, ir_cell] += J_z_m0_00
            j_z_m1_global[tx,iz_cell, ir_cell] += J_z_m1_00

            j_z_m0_global[tx,iz_cell+1 + shift_z, ir_cell] += J_z_m0_01
            j_z_m1_global[tx,iz_cell+1 + shift_z, ir_cell] += J_z_m1_01

            j_z_m0_global[tx,iz_cell, ir_cell+1 + shift_r] += J_z_m0_10
            j_z_m1_global[tx,iz_cell, ir_cell+1 + shift_r] += J_z_m1_10

            j_z_m0_global[tx,iz_cell+1 + shift_z, ir_cell+1 + shift_r] += J_z_m0_11
            j_z_m1_global[tx,iz_cell+1 + shift_z, ir_cell+1 + shift_r] += J_z_m1_11

    return

# -----------------------------------------------------------------------
# Parallel reduction of the global arrays for threads into a single array
# -----------------------------------------------------------------------

@numba.njit( parallel=True )
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
