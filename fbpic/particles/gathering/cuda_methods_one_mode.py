# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the field gathering methods linear and cubic order shapes
on the GPU using CUDA, for one azimuthal mode at a time
"""
from numba import cuda, float64, int64
from fbpic.utils.cuda import compile_cupy
import math
# Import inline functions
from .inline_functions import \
    add_linear_gather_for_mode, add_cubic_gather_for_mode
# Compile the inline functions for GPU
add_linear_gather_for_mode = cuda.jit( add_linear_gather_for_mode,
                                        device=True, inline=True )
add_cubic_gather_for_mode = cuda.jit( add_cubic_gather_for_mode,
                                        device=True, inline=True )

@compile_cupy
def erase_eb_cuda( Ex, Ey, Ez, Bx, By, Bz, Ntot ):
    """
    Reset the arrays of fields (i.e. set them to 0)

    Parameters
    ----------
    Ex, Ey, Ez, Bx, By, Bz: 1d arrays of floats
        (One element per macroparticle)
        Represents the fields on the macroparticles
    """
    i = cuda.grid(1)
    if i < Ntot:
        Ex[i] = 0
        Ey[i] = 0
        Ez[i] = 0
        Bx[i] = 0
        By[i] = 0
        Bz[i] = 0

# -----------------------
# Field gathering linear
# -----------------------

@compile_cupy
def gather_field_gpu_linear_one_mode(x, y, z,
                    rmax_gather,
                    invdz, zmin, Nz,
                    invdr, rmin, Nr,
                    Er_m, Et_m, Ez_m,
                    Br_m, Bt_m, Bz_m, m,
                    Ex, Ey, Ez,
                    Bx, By, Bz):
    """
    Gathering of the fields (E and B) using numba on the GPU.
    Iterates over the particles, calculates the weighted amount
    of fields acting on each particle based on its shape (linear).
    Fields are gathered in cylindrical coordinates and then
    transformed to cartesian coordinates.
    Supports only mode 0 and 1.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    rmax_gather: float (in meters)
        The radius above which particle do not gather anymore

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box along the
        direction considered

    Nz, Nr : int
        Number of gridpoints along the considered direction

    Er_m, Et_m, Ez_m : 2darray of complexs
        The electric fields on the interpolation grid for the mode m

    Br_m, Bt_m, Bz_m : 2darray of complexs
        The magnetic fields on the interpolation grid for the mode m

    m: int
        Index of the azimuthal mode

    Ex, Ey, Ez : 1darray of floats
        The electric fields acting on the particles
        (is modified by this function)

    Bx, By, Bz : 1darray of floats
        The magnetic fields acting on the particles
        (is modified by this function)
    """
    # Get the 1D CUDA grid
    i = cuda.grid(1)
    # Deposit the field per cell in parallel
    # (for threads < number of particles)
    if i < x.shape[0]:
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
        # Calculate azimuthal complex factor
        exptheta_m = 1.
        for _ in range(m):
            exptheta_m *= (cos - 1.j*sin)

        # Get linear weights for the deposition
        # --------------------------------------------
        # Positions of the particles, in the cell unit
        r_cell =  invdr*(rj - rmin) - 0.5
        z_cell =  invdz*(zj - zmin) - 0.5

        # Only perform gathering for particles that are below rmax_gather
        if rj < rmax_gather:

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

            # Precalculate Shapes
            S_ll = Sz_lower*Sr_lower
            S_lu = Sz_lower*Sr_upper
            S_ul = Sz_upper*Sr_lower
            S_uu = Sz_upper*Sr_upper
            S_lg = Sz_lower*Sr_guard
            S_ug = Sz_upper*Sr_guard

            # E-Field
            # -------
            Fr = 0.
            Ft = 0.
            Fz = 0.
            # Add contribution from mode m
            Fr, Ft, Fz = add_linear_gather_for_mode( m,
                Fr, Ft, Fz, exptheta_m, Er_m, Et_m, Ez_m,
                iz_lower, iz_upper, ir_lower, ir_upper,
                S_ll, S_lu, S_lg, S_ul, S_uu, S_ug )
            # Convert to Cartesian coordinates
            # and write to particle field arrays
            Ex[i] += cos*Fr - sin*Ft
            Ey[i] += sin*Fr + cos*Ft
            Ez[i] += Fz

            # B-Field
            # -------
            # Clear the placeholders for the
            # gathered field for each coordinate
            Fr = 0.
            Ft = 0.
            Fz = 0.
            # Add contribution from mode m
            Fr, Ft, Fz = add_linear_gather_for_mode( m,
                Fr, Ft, Fz, exptheta_m, Br_m, Bt_m, Bz_m,
                iz_lower, iz_upper, ir_lower, ir_upper,
                S_ll, S_lu, S_lg, S_ul, S_uu, S_ug )
            # Convert to Cartesian coordinates
            # and write to particle field arrays
            Bx[i] += cos*Fr - sin*Ft
            By[i] += sin*Fr + cos*Ft
            Bz[i] += Fz

# -----------------------
# Field gathering cubic
# -----------------------

@compile_cupy
def gather_field_gpu_cubic_one_mode(x, y, z,
                    rmax_gather,
                    invdz, zmin, Nz,
                    invdr, rmin, Nr,
                    Er_m, Et_m, Ez_m,
                    Br_m, Bt_m, Bz_m, m,
                    Ex, Ey, Ez,
                    Bx, By, Bz):
    """
    Gathering of the fields (E and B) using numba on the GPU.
    Iterates over the particles, calculates the weighted amount
    of fields acting on each particle based on its shape (cubic).
    Fields are gathered in cylindrical coordinates and then
    transformed to cartesian coordinates.
    Supports only mode 0 and 1.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    rmax_gather: float (in meters)
        The radius above which particle do not gather anymore

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box along the
        direction considered

    Nz, Nr : int
        Number of gridpoints along the considered direction

    Er_m, Et_m, Ez_m : 2darray of complexs
        The electric fields on the interpolation grid for the mode m

    Br_m, Bt_m, Bz_m : 2darray of complexs
        The magnetic fields on the interpolation grid for the mode m

    m: int
        Index of the azimuthal mode

    Ex, Ey, Ez : 1darray of floats
        The electric fields acting on the particles
        (is modified by this function)

    Bx, By, Bz : 1darray of floats
        The magnetic fields acting on the particles
        (is modified by this function)
    """

    # Get the 1D CUDA grid
    i = cuda.grid(1)
    # Deposit the field per cell in parallel
    # (for threads < number of particles)
    if i < x.shape[0]:
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
        # Calculate azimuthal complex factor
        exptheta_m = 1.
        for _ in range(m):
            exptheta_m *= (cos - 1.j*sin)

        # Get weights for the deposition
        # --------------------------------------------
        # Positions of the particle, in the cell unit
        r_cell = invdr*(rj - rmin) - 0.5
        z_cell = invdz*(zj - zmin) - 0.5

        # Only perform gathering for particles that are below rmax_gather
        if rj < rmax_gather:

            # Calculate the shape factors
            Sr = cuda.local.array((4,), dtype=float64)
            ir_lowest = int64(math.floor(r_cell)) - 1
            r_local = r_cell-ir_lowest
            Sr[0] = -1./6. * (r_local-2.)**3
            Sr[1] = 1./6. * (3.*(r_local-1.)**3 - 6.*(r_local-1.)**2 + 4.)
            Sr[2] = 1./6. * (3.*(2.-r_local)**3 - 6.*(2.-r_local)**2 + 4.)
            Sr[3] = -1./6. * (1.-r_local)**3
            Sz = cuda.local.array((4,), dtype=float64)
            iz_lowest = int64(math.floor(z_cell)) - 1
            z_local = z_cell-iz_lowest
            Sz[0] = -1./6. * (z_local-2.)**3
            Sz[1] = 1./6. * (3.*(z_local-1.)**3 - 6.*(z_local-1.)**2 + 4.)
            Sz[2] = 1./6. * (3.*(2.-z_local)**3 - 6.*(2.-z_local)**2 + 4.)
            Sz[3] = -1./6. * (1.-z_local)**3

            # E-Field
            # -------
            Fr = 0.
            Ft = 0.
            Fz = 0.
            # Add contribution from mode m
            Fr, Ft, Fz = add_cubic_gather_for_mode( m,
                Fr, Ft, Fz, exptheta_m, Er_m, Et_m, Ez_m,
                ir_lowest, iz_lowest, Sr, Sz, Nr, Nz )
            # Convert to Cartesian coordinates
            # and write to particle field arrays
            Ex[i] += cos*Fr - sin*Ft
            Ey[i] += sin*Fr + cos*Ft
            Ez[i] += Fz

            # B-Field
            # -------
            # Clear the placeholders for the
            # gathered field for each coordinate
            Fr = 0.
            Ft = 0.
            Fz = 0.
            # Add contribution from mode m
            Fr, Ft, Fz =  add_cubic_gather_for_mode( m,
                Fr, Ft, Fz, exptheta_m, Br_m, Bt_m, Bz_m,
                ir_lowest, iz_lowest, Sr, Sz, Nr, Nz )
            # Convert to Cartesian coordinates
            # and write to particle field arrays
            Bx[i] += cos*Fr - sin*Ft
            By[i] += sin*Fr + cos*Ft
            Bz[i] += Fz
