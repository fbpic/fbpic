# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the field gathering methods linear and cubic order shapes
on the CPU with threading.
"""
from numba import int64
from fbpic.threading_utils import njit_parallel, prange
import math
import numpy as np

# -----------------------
# Field gathering linear
# -----------------------

@njit_parallel
def gather_field_numba_linear(x, y, z,
                    invdz, zmin, Nz,
                    invdr, rmin, Nr,
                    Er_m0, Et_m0, Ez_m0,
                    Er_m1, Et_m1, Ez_m1,
                    Br_m0, Bt_m0, Bz_m0,
                    Br_m1, Bt_m1, Bz_m1,
                    Ex, Ey, Ez,
                    Bx, By, Bz ):
    """
    Gathering of the fields (E and B) using numba with multi-threading.
    Iterates over the particles, calculates the weighted amount
    of fields acting on each particle based on its shape (linear).
    Fields are gathered in cylindrical coordinates and then
    transformed to cartesian coordinates.
    Supports only mode 0 and 1.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box along the
        direction considered

    Nz, Nr : int
        Number of gridpoints along the considered direction

    Er_m0, Et_m0, Ez_m0 : 2darray of complexs
        The electric fields on the interpolation grid for the mode 0

    Er_m1, Et_m1, Ez_m1 : 2darray of complexs
        The electric fields on the interpolation grid for the mode 1

    Br_m0, Bt_m0, Bz_m0 : 2darray of complexs
        The magnetic fields on the interpolation grid for the mode 0

    Br_m1, Bt_m1, Bz_m1 : 2darray of complexs
        The magnetic fields on the interpolation grid for the mode 1

    Ex, Ey, Ez : 1darray of floats
        The electric fields acting on the particles
        (is modified by this function)

    Bx, By, Bz : 1darray of floats
        The magnetic fields acting on the particles
        (is modified by this function)
    """
    # Deposit the field per cell in parallel
    for i in prange(x.shape[0]):
        # Preliminary arrays for the cylindrical conversion
        # --------------------------------------------
        # Position
        xj = x[i]
        yj = y[i]
        zj = z[i]

        # Cylindrical conversion
        rj = math.sqrt( xj**2 + yj**2 )
        if (rj !=0. ) :
            invr = 1./rj
            cos = xj*invr  # Cosine
            sin = yj*invr  # Sine
        else :
            cos = 1.
            sin = 0.
        exptheta_m0 = 1.
        exptheta_m1 = cos - 1.j*sin

        # Get linear weights for the deposition
        # --------------------------------------------
        # Positions of the particles, in the cell unit
        r_cell =  invdr*(rj - rmin) - 0.5
        z_cell =  invdz*(zj - zmin) - 0.5
        # Original index of the uppper and lower cell
        ir_lower = int(math.floor( r_cell ))
        ir_upper = ir_lower + 1
        iz_lower = int(math.floor( z_cell ))
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
        if ir_lower > Nr-1:
            ir_lower = Nr-1
        if ir_upper > Nr-1:
            ir_upper = Nr-1
        # periodic boundaries in z
        # lower z boundaries
        if iz_lower < 0:
            iz_lower += Nz
        if iz_upper < 0:
            iz_upper += Nz
        # upper z boundaries
        if iz_lower > Nz-1:
            iz_lower -= Nz
        if iz_upper > Nz-1:
            iz_upper -= Nz

        #Precalculate Shapes
        S_ll = Sz_lower*Sr_lower
        S_lu = Sz_lower*Sr_upper
        S_ul = Sz_upper*Sr_lower
        S_uu = Sz_upper*Sr_upper
        S_lg = Sz_lower*Sr_guard
        S_ug = Sz_upper*Sr_guard

        # E-Field
        # ----------------------------
        # Define the initial placeholders for the
        # gathered field for each coordinate
        Fr = 0.
        Ft = 0.
        Fz = 0.

        # Mode 0
        # ----------------------------
        # Create temporary variables
        # for the "per mode" gathering
        Fr_m = 0.j
        Ft_m = 0.j
        Fz_m = 0.j
        # Add the fields for mode 0
        # Lower cell in z, Lower cell in r
        Fr_m += S_ll * Er_m0[ iz_lower, ir_lower ]
        Ft_m += S_ll * Et_m0[ iz_lower, ir_lower ]
        Fz_m += S_ll * Ez_m0[ iz_lower, ir_lower ]
        # Lower cell in z, Upper cell in r
        Fr_m += S_lu * Er_m0[ iz_lower, ir_upper ]
        Ft_m += S_lu * Et_m0[ iz_lower, ir_upper ]
        Fz_m += S_lu * Ez_m0[ iz_lower, ir_upper ]
        # Upper cell in z, Lower cell in r
        Fr_m += S_ul * Er_m0[ iz_upper, ir_lower ]
        Ft_m += S_ul * Et_m0[ iz_upper, ir_lower ]
        Fz_m += S_ul * Ez_m0[ iz_upper, ir_lower ]
        # Upper cell in z, Upper cell in r
        Fr_m += S_uu * Er_m0[ iz_upper, ir_upper ]
        Ft_m += S_uu * Et_m0[ iz_upper, ir_upper ]
        Fz_m += S_uu * Ez_m0[ iz_upper, ir_upper ]
        # Add the fields from the guard cells
        if ir_lower == ir_upper == 0:
            # Lower cell in z
            Fr_m += -1. * S_lg * Er_m0[ iz_lower, 0]
            Ft_m += -1. * S_lg * Et_m0[ iz_lower, 0]
            Fz_m +=  1. * S_lg * Ez_m0[ iz_lower, 0]
            # Upper cell in z
            Fr_m += -1. * S_ug * Er_m0[ iz_upper, 0]
            Ft_m += -1. * S_ug * Et_m0[ iz_upper, 0]
            Fz_m +=  1. * S_ug * Ez_m0[ iz_upper, 0]
        # Add the fields from the mode 0
        Fr += (Fr_m*exptheta_m0).real
        Ft += (Ft_m*exptheta_m0).real
        Fz += (Fz_m*exptheta_m0).real

        # Mode 1
        # ----------------------------
        # Clear the temporary variables
        # for the "per mode" gathering
        Fr_m = 0.j
        Ft_m = 0.j
        Fz_m = 0.j
        # Add the fields for mode 1
        # Lower cell in z, Lower cell in r
        Fr_m += S_ll * Er_m1[ iz_lower, ir_lower ]
        Ft_m += S_ll * Et_m1[ iz_lower, ir_lower ]
        Fz_m += S_ll * Ez_m1[ iz_lower, ir_lower ]
        # Lower cell in z, Upper cell in r
        Fr_m += S_lu * Er_m1[ iz_lower, ir_upper ]
        Ft_m += S_lu * Et_m1[ iz_lower, ir_upper ]
        Fz_m += S_lu * Ez_m1[ iz_lower, ir_upper ]
        # Upper cell in z, Lower cell in r
        Fr_m += S_ul * Er_m1[ iz_upper, ir_lower ]
        Ft_m += S_ul * Et_m1[ iz_upper, ir_lower ]
        Fz_m += S_ul * Ez_m1[ iz_upper, ir_lower ]
        # Upper cell in z, Upper cell in r
        Fr_m += S_uu * Er_m1[ iz_upper, ir_upper ]
        Ft_m += S_uu * Et_m1[ iz_upper, ir_upper ]
        Fz_m += S_uu * Ez_m1[ iz_upper, ir_upper ]
        # Add the fields from the guard cells
        if ir_lower == ir_upper == 0:
            # Lower cell in z
            Fr_m +=  1. * S_lg * Er_m1[ iz_lower, 0]
            Ft_m +=  1. * S_lg * Et_m1[ iz_lower, 0]
            Fz_m += -1. * S_lg * Ez_m1[ iz_lower, 0]
            # Upper cell in z
            Fr_m +=  1. * S_ug * Er_m1[ iz_upper, 0]
            Ft_m +=  1. * S_ug * Et_m1[ iz_upper, 0]
            Fz_m += -1. * S_ug * Ez_m1[ iz_upper, 0]
        # Add the fields from the mode 1
        Fr += 2*(Fr_m*exptheta_m1).real
        Ft += 2*(Ft_m*exptheta_m1).real
        Fz += 2*(Fz_m*exptheta_m1).real

        # Convert to Cartesian coordinates
        # and write to particle field arrays
        Ex[i] = cos*Fr - sin*Ft
        Ey[i] = sin*Fr + cos*Ft
        Ez[i] = Fz

        # B-Field
        # ----------------------------
        # Clear the placeholders for the
        # gathered field for each coordinate
        Fr = 0.
        Ft = 0.
        Fz = 0.

        # Mode 0
        # ----------------------------
        # Create temporary variables
        # for the "per mode" gathering
        Fr_m = 0.j
        Ft_m = 0.j
        Fz_m = 0.j
        # Add the fields for mode 0
        # Lower cell in z, Lower cell in r
        Fr_m += S_ll * Br_m0[ iz_lower, ir_lower ]
        Ft_m += S_ll * Bt_m0[ iz_lower, ir_lower ]
        Fz_m += S_ll * Bz_m0[ iz_lower, ir_lower ]
        # Lower cell in z, Upper cell in r
        Fr_m += S_lu * Br_m0[ iz_lower, ir_upper ]
        Ft_m += S_lu * Bt_m0[ iz_lower, ir_upper ]
        Fz_m += S_lu * Bz_m0[ iz_lower, ir_upper ]
        # Upper cell in z, Lower cell in r
        Fr_m += S_ul * Br_m0[ iz_upper, ir_lower ]
        Ft_m += S_ul * Bt_m0[ iz_upper, ir_lower ]
        Fz_m += S_ul * Bz_m0[ iz_upper, ir_lower ]
        # Upper cell in z, Upper cell in r
        Fr_m += S_uu * Br_m0[ iz_upper, ir_upper ]
        Ft_m += S_uu * Bt_m0[ iz_upper, ir_upper ]
        Fz_m += S_uu * Bz_m0[ iz_upper, ir_upper ]
        # Add the fields from the guard cells
        if ir_lower == ir_upper == 0:
            # Lower cell in z
            Fr_m += -1. * S_lg * Br_m0[ iz_lower, 0]
            Ft_m += -1. * S_lg * Bt_m0[ iz_lower, 0]
            Fz_m +=  1. * S_lg * Bz_m0[ iz_lower, 0]
            # Upper cell in z
            Fr_m += -1. * S_ug * Br_m0[ iz_upper, 0]
            Ft_m += -1. * S_ug * Bt_m0[ iz_upper, 0]
            Fz_m +=  1. * S_ug * Bz_m0[ iz_upper, 0]
        # Add the fields from the mode 0
        Fr += (Fr_m*exptheta_m0).real
        Ft += (Ft_m*exptheta_m0).real
        Fz += (Fz_m*exptheta_m0).real

        # Mode 1
        # ----------------------------
        # Clear the temporary variables
        # for the "per mode" gathering
        Fr_m = 0.j
        Ft_m = 0.j
        Fz_m = 0.j
        # Add the fields for mode 1
        # Lower cell in z, Lower cell in r
        Fr_m += S_ll * Br_m1[ iz_lower, ir_lower ]
        Ft_m += S_ll * Bt_m1[ iz_lower, ir_lower ]
        Fz_m += S_ll * Bz_m1[ iz_lower, ir_lower ]
        # Lower cell in z, Upper cell in r
        Fr_m += S_lu * Br_m1[ iz_lower, ir_upper ]
        Ft_m += S_lu * Bt_m1[ iz_lower, ir_upper ]
        Fz_m += S_lu * Bz_m1[ iz_lower, ir_upper ]
        # Upper cell in z, Lower cell in r
        Fr_m += S_ul * Br_m1[ iz_upper, ir_lower ]
        Ft_m += S_ul * Bt_m1[ iz_upper, ir_lower ]
        Fz_m += S_ul * Bz_m1[ iz_upper, ir_lower ]
        # Upper cell in z, Upper cell in r
        Fr_m += S_uu * Br_m1[ iz_upper, ir_upper ]
        Ft_m += S_uu * Bt_m1[ iz_upper, ir_upper ]
        Fz_m += S_uu * Bz_m1[ iz_upper, ir_upper ]

        # Add the fields from the guard cells
        if ir_lower == ir_upper == 0:
            # Lower cell in z
            Fr_m +=  1. * S_lg * Br_m1[ iz_lower, 0]
            Ft_m +=  1. * S_lg * Bt_m1[ iz_lower, 0]
            Fz_m += -1. * S_lg * Bz_m1[ iz_lower, 0]
            # Upper cell in z
            Fr_m +=  1. * S_ug * Br_m1[ iz_upper, 0]
            Ft_m +=  1. * S_ug * Bt_m1[ iz_upper, 0]
            Fz_m += -1. * S_ug * Bz_m1[ iz_upper, 0]
        # Add the fields from the mode 1
        Fr += 2*(Fr_m*exptheta_m1).real
        Ft += 2*(Ft_m*exptheta_m1).real
        Fz += 2*(Fz_m*exptheta_m1).real

        # Convert to Cartesian coordinates
        # and write to particle field arrays
        Bx[i] = cos*Fr - sin*Ft
        By[i] = sin*Fr + cos*Ft
        Bz[i] = Fz

    return Ex, Ey, Ez, Bx, By, Bz

# -----------------------
# Field gathering cubic
# -----------------------

@njit_parallel
def gather_field_numba_cubic(x, y, z,
                    invdz, zmin, Nz,
                    invdr, rmin, Nr,
                    Er_m0, Et_m0, Ez_m0,
                    Er_m1, Et_m1, Ez_m1,
                    Br_m0, Bt_m0, Bz_m0,
                    Br_m1, Bt_m1, Bz_m1,
                    Ex, Ey, Ez,
                    Bx, By, Bz,
                    nthreads, ptcl_chunk_indices):
    """
    Gathering of the fields (E and B) using numba with multi-threading.
    Iterates over the particles, calculates the weighted amount
    of fields acting on each particle based on its shape (cubic).
    Fields are gathered in cylindrical coordinates and then
    transformed to cartesian coordinates.
    Supports only mode 0 and 1.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box along the
        direction considered

    Nz, Nr : int
        Number of gridpoints along the considered direction

    Er_m0, Et_m0, Ez_m0 : 2darray of complexs
        The electric fields on the interpolation grid for the mode 0

    Er_m1, Et_m1, Ez_m1 : 2darray of complexs
        The electric fields on the interpolation grid for the mode 1

    Br_m0, Bt_m0, Bz_m0 : 2darray of complexs
        The magnetic fields on the interpolation grid for the mode 0

    Br_m1, Bt_m1, Bz_m1 : 2darray of complexs
        The magnetic fields on the interpolation grid for the mode 1

    Ex, Ey, Ez : 1darray of floats
        The electric fields acting on the particles
        (is modified by this function)

    Bx, By, Bz : 1darray of floats
        The magnetic fields acting on the particles
        (is modified by this function)

    nthreads : int
        Number of CPU threads used with numba prange

    ptcl_chunk_indices : array of int, of size nthreads+1
        The indices (of the particle array) between which each thread
        should loop. (i.e. divisions of particle array between threads)
    """
    # Gather the field per cell in parallel
    for nt in prange( nthreads ):

        # Create private arrays for each thread
        # to store the particle index and shape
        ir = np.empty( 4, dtype=int64)
        Sr = np.empty( 4 )
        iz = np.empty( 4, dtype=int64)
        Sz = np.empty( 4 )

        # Loop over all particles in thread chunk
        for i in range( ptcl_chunk_indices[nt],
                            ptcl_chunk_indices[nt+1] ):

            # Preliminary arrays for the cylindrical conversion
            # --------------------------------------------
            # Position
            xj = x[i]
            yj = y[i]
            zj = z[i]

            # Cylindrical conversion
            rj = math.sqrt(xj**2 + yj**2)
            if (rj != 0.):
                invr = 1./rj
                cos = xj*invr  # Cosine
                sin = yj*invr  # Sine
            else:
                cos = 1.
                sin = 0.
            exptheta_m0 = 1.
            exptheta_m1 = cos - 1.j*sin

            # Get weights for the deposition
            # --------------------------------------------
            # Positions of the particle, in the cell unit
            r_cell = invdr*(rj - rmin) - 0.5
            z_cell = invdz*(zj - zmin) - 0.5

            # Calculate the shape factors
            ir[0] = int64(math.floor(r_cell)) - 1
            ir[1] = ir[0] + 1
            ir[2] = ir[1] + 1
            ir[3] = ir[2] + 1
            Sr[0] = -1./6. * ((r_cell-ir[0])-2)**3
            Sr[1] = 1./6. * (3*((r_cell-ir[1])**3)-6*((r_cell-ir[1])**2)+4)
            Sr[2] = 1./6. * (3*((ir[2]-r_cell)**3)-6*((ir[2]-r_cell)**2)+4)
            Sr[3] = -1./6. * ((ir[3]-r_cell)-2)**3
            iz[0] = int64(math.floor(z_cell)) - 1
            iz[1] = iz[0] + 1
            iz[2] = iz[1] + 1
            iz[3] = iz[2] + 1
            Sz[0] = -1./6. * ((z_cell-iz[0])-2)**3
            Sz[1] = 1./6. * (3*((z_cell-iz[1])**3)-6*((z_cell-iz[1])**2)+4)
            Sz[2] = 1./6. * (3*((iz[2]-z_cell)**3)-6*((iz[2]-z_cell)**2)+4)
            Sz[3] = -1./6. * ((iz[3]-z_cell)-2)**3
            # Lower and upper periodic boundary for z
            index_z = 0
            while index_z < 4:
                if iz[index_z] < 0:
                    iz[index_z] += Nz
                if iz[index_z] > Nz - 1:
                    iz[index_z] -= Nz
                index_z += 1
            # Lower and upper boundary for r
            index_r = 0
            while index_r < 4:
                if ir[index_r] < 0:
                    ir[index_r] = abs(ir[index_r])-1
                    Sr[index_r] = (-1.)*Sr[index_r]
                if ir[index_r] > Nr - 1:
                    ir[index_r] = Nr - 1
                index_r += 1

            # E-Field
            # ----------------------------
            # Define the initial placeholders for the
            # gathered field for each coordinate
            Fr = 0.
            Ft = 0.
            Fz = 0.

            # Mode 0
            # ----------------------------
            # Create temporary variables
            # for the "per mode" gathering
            Fr_m = 0.j
            Ft_m = 0.j
            Fz_m = 0.j
            # Add the fields for mode 0
            index_r = 0
            while index_r < 4:
                index_z = 0
                while index_z < 4:
                    Fr_m += Sz[index_z]*Sr[index_r] * \
                        Er_m0[iz[index_z], ir[index_r]]
                    Ft_m += Sz[index_z]*Sr[index_r] * \
                        Et_m0[iz[index_z], ir[index_r]]
                    if Sz[index_z]*Sr[index_r] < 0:
                        Fz_m += (-1.)*Sz[index_z]*Sr[index_r]* \
                            Ez_m0[iz[index_z], ir[index_r]]
                    else:
                        Fz_m += Sz[index_z]*Sr[index_r]* \
                            Ez_m0[iz[index_z], ir[index_r]]
                    index_z += 1
                index_r += 1

            Fr += (Fr_m*exptheta_m0).real
            Ft += (Ft_m*exptheta_m0).real
            Fz += (Fz_m*exptheta_m0).real

            # Mode 1
            # ----------------------------
            # Clear the temporary variables
            # for the "per mode" gathering
            Fr_m = 0.j
            Ft_m = 0.j
            Fz_m = 0.j
            # Add the fields for mode 1
            index_r = 0
            while index_r < 4:
                index_z = 0
                while index_z < 4:
                    if Sz[index_z]*Sr[index_r] < 0:
                        Fr_m += (-1.)*Sz[index_z]*Sr[index_r]* \
                                    Er_m1[iz[index_z], ir[index_r]]
                        Ft_m += (-1.)*Sz[index_z]*Sr[index_r]* \
                                    Et_m1[iz[index_z], ir[index_r]]
                    else:
                        Fr_m += Sz[index_z]*Sr[index_r]* \
                                    Er_m1[iz[index_z], ir[index_r]]
                        Ft_m += Sz[index_z]*Sr[index_r]* \
                                    Et_m1[iz[index_z], ir[index_r]]
                    Fz_m += Sz[index_z]*Sr[index_r] * \
                        Ez_m1[iz[index_z], ir[index_r]]
                    index_z += 1
                index_r += 1

            # Add the fields from the mode 1
            Fr += 2*(Fr_m*exptheta_m1).real
            Ft += 2*(Ft_m*exptheta_m1).real
            Fz += 2*(Fz_m*exptheta_m1).real

            # Convert to Cartesian coordinates
            # and write to particle field arrays
            Ex[i] = (cos*Fr - sin*Ft)
            Ey[i] = (sin*Fr + cos*Ft)
            Ez[i] = Fz

            # B-Field
            # ----------------------------
            # Clear the placeholders for the
            # gathered field for each coordinate
            Fr = 0.
            Ft = 0.
            Fz = 0.

            # Mode 0
            # ----------------------------
            # Create temporary variables
            # for the "per mode" gathering
            Fr_m = 0.j
            Ft_m = 0.j
            Fz_m = 0.j
            # Add the fields for mode 0
            index_r = 0
            while index_r < 4:
                index_z = 0
                while index_z < 4:
                    Fr_m += Sz[index_z]*Sr[index_r]* \
                        Br_m0[iz[index_z], ir[index_r]]
                    Ft_m += Sz[index_z]*Sr[index_r]* \
                        Bt_m0[iz[index_z], ir[index_r]]
                    if Sz[index_z]*Sr[index_r] < 0:
                        Fz_m += (-1.)*Sz[index_z]*Sr[index_r]* \
                            Bz_m0[iz[index_z], ir[index_r]]
                    else:
                        Fz_m += Sz[index_z]*Sr[index_r]* \
                            Bz_m0[iz[index_z], ir[index_r]]
                    index_z += 1
                index_r += 1

            # Add the fields from the mode 0
            Fr += (Fr_m*exptheta_m0).real
            Ft += (Ft_m*exptheta_m0).real
            Fz += (Fz_m*exptheta_m0).real

            # Mode 1
            # ----------------------------
            # Clear the temporary variables
            # for the "per mode" gathering
            Fr_m = 0.j
            Ft_m = 0.j
            Fz_m = 0.j

            # Add the fields for mode 1
            index_r = 0
            while index_r < 4:
                index_z = 0
                while index_z < 4:
                    if Sz[index_z]*Sr[index_r] < 0:
                        Fr_m += (-1.)*Sz[index_z]*Sr[index_r]* \
                            Br_m1[iz[index_z], ir[index_r]]
                        Ft_m += (-1.)*Sz[index_z]*Sr[index_r]* \
                            Bt_m1[iz[index_z], ir[index_r]]
                    else:
                        Fr_m += Sz[index_z]*Sr[index_r]* \
                            Br_m1[iz[index_z], ir[index_r]]
                        Ft_m += Sz[index_z]*Sr[index_r]* \
                            Bt_m1[iz[index_z], ir[index_r]]
                    Fz_m += Sz[index_z]*Sr[index_r]*Bz_m1[iz[index_z], ir[index_r]]
                    index_z += 1
                index_r += 1

            # Add the fields from the mode 1
            Fr += 2*(Fr_m*exptheta_m1).real
            Ft += 2*(Ft_m*exptheta_m1).real
            Fz += 2*(Fz_m*exptheta_m1).real

            # Convert to Cartesian coordinates
            # and write to particle field arrays
            Bx[i] = cos*Fr - sin*Ft
            By[i] = sin*Fr + cos*Ft
            Bz[i] = Fz

    return Ex, Ey, Ez, Bx, By, Bz
