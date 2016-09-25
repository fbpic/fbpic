# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the optimized particles methods that use cuda on a GPU
"""
from numba import cuda
from accelerate.cuda import sorting
import math
from scipy.constants import c
import numpy as np

# -----------------------
# Particle pusher utility
# -----------------------

@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], \
            float64[:], float64[:], float64[:], \
            float64[:], float64[:], float64[:], \
            float64, float64, int32, float64)')
def push_p_gpu( ux, uy, uz, inv_gamma, 
                Ex, Ey, Ez, Bx, By, Bz,
                q, m, Ntot, dt ) :
    """
    Advance the particles' momenta, using numba on the GPU

    Parameters
    ----------
    ux, uy, uz : 1darray of floats
        The velocity of the particles
        (is modified by this function)

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    Ex, Ey, Ez : 1darray of floats
        The electric fields acting on the particles

    Bx, By, Bz : 1darray of floats
        The magnetic fields acting on the particles

    q : float
        The charge of the particle species

    m : float
        The mass of the particle species

    Ntot : int
        The total number of particles

    dt : float
        The time by which the momenta is advanced
    """
    # Set a few constants
    econst = q*dt/(m*c)
    bconst = 0.5*q*dt/m
    
    #Cuda 1D grid
    ip = cuda.grid(1)

    # Loop over the particles
    if ip < Ntot:

        # Shortcut for initial 1./gamma
        inv_gamma_i = inv_gamma[ip]
            
        # Get the magnetic rotation vector
        taux = bconst*Bx[ip]
        tauy = bconst*By[ip]
        tauz = bconst*Bz[ip]
        tau2 = taux**2 + tauy**2 + tauz**2
            
        # Get the momenta at the half timestep
        uxp = ux[ip] + econst*Ex[ip] \
        + inv_gamma_i*( uy[ip]*tauz - uz[ip]*tauy )
        uyp = uy[ip] + econst*Ey[ip] \
        + inv_gamma_i*( uz[ip]*taux - ux[ip]*tauz )
        uzp = uz[ip] + econst*Ez[ip] \
        + inv_gamma_i*( ux[ip]*tauy - uy[ip]*taux )
        sigma = 1 + uxp**2 + uyp**2 + uzp**2 - tau2
        utau = uxp*taux + uyp*tauy + uzp*tauz

        # Get the new 1./gamma
        inv_gamma_f = math.sqrt(
            2./( sigma + math.sqrt( sigma**2 + 4*(tau2 + utau**2 ) ) )
        )
        inv_gamma[ip] = inv_gamma_f

        # Reuse the tau and utau arrays to save memory
        tx = inv_gamma_f*taux
        ty = inv_gamma_f*tauy
        tz = inv_gamma_f*tauz
        ut = inv_gamma_f*utau
        s = 1./( 1 + tau2*inv_gamma_f**2 )

        # Get the new u
        ux[ip] = s*( uxp + tx*ut + uyp*tz - uzp*ty )
        uy[ip] = s*( uyp + ty*ut + uzp*tx - uxp*tz )
        uz[ip] = s*( uzp + tz*ut + uxp*ty - uyp*tx )

@cuda.jit('void(float64[:], float64[:], float64[:], \
            float64[:], float64[:], float64[:], \
            float64[:], float64)')
def push_x_gpu( x, y, z, ux, uy, uz, inv_gamma, dt ) :
    """
    Advance the particles' positions over one half-timestep
    
    This assumes that the positions (x, y, z) are initially either
    one half-timestep *behind* the momenta (ux, uy, uz), or at the
    same timestep as the momenta.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)

    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    dt : float (seconds)
        The time by which the position is advanced
    """
    # Half timestep, multiplied by c
    chdt = c*0.5*dt

    i = cuda.grid(1)
    if i < x.shape[0]:
        # Particle push
        inv_g = inv_gamma[i]
        x[i] += chdt*inv_g*ux[i]
        y[i] += chdt*inv_g*uy[i]
        z[i] += chdt*inv_g*uz[i]

# -----------------------
# Field gathering utility
# -----------------------

@cuda.jit('void(float64[:], float64[:], float64[:], \
            float64, float64, int32, \
            float64, float64, int32, \
            complex128[:,:], complex128[:,:], complex128[:,:], \
            complex128[:,:], complex128[:,:], complex128[:,:], \
            complex128[:,:], complex128[:,:], complex128[:,:], \
            complex128[:,:], complex128[:,:], complex128[:,:], \
            float64[:], float64[:], float64[:], \
            float64[:], float64[:], float64[:])')
def gather_field_gpu(x, y, z,
                    invdz, zmin, Nz,
                    invdr, rmin, Nr,
                    Er_m0, Et_m0, Ez_m0,
                    Er_m1, Et_m1, Ez_m1,
                    Br_m0, Bt_m0, Bz_m0,
                    Br_m1, Bt_m1, Bz_m1,
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
        (is mofidied by this function)

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
        iz = int(i/Nr)
        ir = int(i - iz * Nr)
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
        # Initialize the local field value for 
        # all four possible deposition directions
        # Mode 0, 1 for r, t, z
        # 1 : lower in r, lower in z
        # 2 : lower in r, upper in z
        # 3 : upper in r, lower in z
        # 4 : upper in r, upper in z
        R1_m0 = 0.+0.j
        R2_m0 = 0.+0.j
        R3_m0 = 0.+0.j
        R4_m0 = 0.+0.j
        # ------------
        R1_m1 = 0.+0.j
        R2_m1 = 0.+0.j
        R3_m1 = 0.+0.j
        R4_m1 = 0.+0.j
        # Loop over the number of particles per cell
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
            rj = math.sqrt( xj**2 + yj**2 )
            # Avoid division by 0.
            if (rj != 0.) :
                invr = 1./rj
                cos = xj*invr  # Cosine
                sin = yj*invr  # Sine
            else :
                cos = 1.
                sin = 0.
            exptheta_m0 = 1.
            exptheta_m1 = cos + 1.j*sin

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
                R1_m0 += Sz_lower*Sr_lower*R_m0
                R1_m0 += Sz_lower*Sr_upper*R_m0
                R3_m0 += Sz_upper*Sr_lower*R_m0
                R3_m0 += Sz_upper*Sr_upper*R_m0
                # -----------------------------
                R1_m1 += Sz_lower*Sr_lower*R_m1
                R1_m1 += Sz_lower*Sr_upper*R_m1
                R3_m1 += Sz_upper*Sr_lower*R_m1
                R3_m1 += Sz_upper*Sr_upper*R_m1
                # -----------------------------
            if ir_lower != ir_upper:
                # In the case that ir_lower and ir_upper are different,
                # add the current to the four arrays according to
                # the direction.
                R1_m0 += Sz_lower*Sr_lower*R_m0
                R2_m0 += Sz_lower*Sr_upper*R_m0
                R3_m0 += Sz_upper*Sr_lower*R_m0
                R4_m0 += Sz_upper*Sr_upper*R_m0
                # -----------------------------
                R1_m1 += Sz_lower*Sr_lower*R_m1
                R2_m1 += Sz_lower*Sr_upper*R_m1
                R3_m1 += Sz_upper*Sr_lower*R_m1
                R4_m1 += Sz_upper*Sr_upper*R_m1
                # -----------------------------
            if ir_lower == ir_upper == 0:
                # Treat the guard cells.
                # Add the current to the guard cells
                # for particles that had an original 
                # cell index < 0.
                R1_m0 += -1.*Sz_lower*Sr_guard*R_m0
                R3_m0 += -1.*Sz_upper*Sr_guard*R_m0
                # ---------------------------------
                R1_m1 += -1.*Sz_lower*Sr_guard*R_m1
                R3_m1 += -1.*Sz_upper*Sr_guard*R_m1
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
                        rho1[i, j-1, 0] + \
                        rho2[i-1, j, 0] + \
                        rho3[i-1, j-1, 0]

        rho_m1[i, j] += rho0[i, j, 1] + \
                        rho1[i, j-1, 1] + \
                        rho2[i-1, j, 1] + \
                        rho3[i-1, j-1, 1]

# -------------------------------
# Field deposition utility - J
# -------------------------------

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
        iz = int(i/Nr)
        ir = int(i - iz * Nr)
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
        # Initialize the local field value for 
        # all four possible deposition directions
        # Mode 0, 1 for r, t, z
        # 1 : lower in r, lower in z
        # 2 : lower in r, upper in z
        # 3 : upper in r, lower in z
        # 4 : upper in r, upper in z
        Jr1_m0 = 0.+0.j
        Jr2_m0 = 0.+0.j
        Jr3_m0 = 0.+0.j
        Jr4_m0 = 0.+0.j
        # -------------
        Jr1_m1 = 0.+0.j
        Jr2_m1 = 0.+0.j
        Jr3_m1 = 0.+0.j
        Jr4_m1 = 0.+0.j
        # -------------
        Jt1_m0 = 0.+0.j
        Jt2_m0 = 0.+0.j
        Jt3_m0 = 0.+0.j
        Jt4_m0 = 0.+0.j
        # -------------
        Jt1_m1 = 0.+0.j
        Jt2_m1 = 0.+0.j
        Jt3_m1 = 0.+0.j
        Jt4_m1 = 0.+0.j
        # -------------
        Jz1_m0 = 0.+0.j
        Jz2_m0 = 0.+0.j
        Jz3_m0 = 0.+0.j
        Jz4_m0 = 0.+0.j
        # -------------
        Jz1_m1 = 0.+0.j
        Jz2_m1 = 0.+0.j
        Jz3_m1 = 0.+0.j
        Jz4_m1 = 0.+0.j
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
            rj = math.sqrt( xj**2 + yj**2 )
            # Avoid division by 0.
            if (rj!=0.) :
                invr = 1./rj
                cos = xj*invr  # Cosine
                sin = yj*invr  # Sine
            else :
                cos = 1.
                sin = 0.
            exptheta_m0 = 1.
            exptheta_m1 = cos + 1.j*sin

            # Get linear weights for the deposition
            # --------------------------------------------
            # Positions of the particles, in the cell unit
            r_cell =  invdr*(rj - rmin) - 0.5
            z_cell =  invdz*(zj - zmin) - 0.5
            # Original index of the uppper and lower cell
            # in r and z
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
            
            # Calculate the currents
            # --------------------------------------------
            # Mode 0
            Jr_m0 = wj * c * inv_gammaj*( cos*uxj + sin*uyj ) * exptheta_m0
            Jt_m0 = wj * c * inv_gammaj*( cos*uyj - sin*uxj ) * exptheta_m0
            Jz_m0 = wj * c * inv_gammaj*uzj * exptheta_m0
            # Mode 1
            Jr_m1 = wj * c * inv_gammaj*( cos*uxj + sin*uyj ) * exptheta_m1
            Jt_m1 = wj * c * inv_gammaj*( cos*uyj - sin*uxj ) * exptheta_m1
            Jz_m1 = wj * c * inv_gammaj*uzj * exptheta_m1

            # Caculate the weighted currents for each
            # of the four possible direction
            # --------------------------------------------
            if ir_lower == ir_upper:
                # In the case that ir_lower and ir_upper are equal,
                # the current is added only to the array corresponding
                # to ir_lower. 
                # (This is the case for the boundaries in r)
                Jr1_m0 += Sz_lower*Sr_lower*Jr_m0
                Jr1_m0 += Sz_lower*Sr_upper*Jr_m0
                Jr3_m0 += Sz_upper*Sr_lower*Jr_m0
                Jr3_m0 += Sz_upper*Sr_upper*Jr_m0
                # -------------------------------
                Jr1_m1 += Sz_lower*Sr_lower*Jr_m1
                Jr1_m1 += Sz_lower*Sr_upper*Jr_m1
                Jr3_m1 += Sz_upper*Sr_lower*Jr_m1
                Jr3_m1 += Sz_upper*Sr_upper*Jr_m1
                # -------------------------------
                Jt1_m0 += Sz_lower*Sr_lower*Jt_m0
                Jt1_m0 += Sz_lower*Sr_upper*Jt_m0
                Jt3_m0 += Sz_upper*Sr_lower*Jt_m0
                Jt3_m0 += Sz_upper*Sr_upper*Jt_m0
                # -------------------------------
                Jt1_m1 += Sz_lower*Sr_lower*Jt_m1
                Jt1_m1 += Sz_lower*Sr_upper*Jt_m1
                Jt3_m1 += Sz_upper*Sr_lower*Jt_m1
                Jt3_m1 += Sz_upper*Sr_upper*Jt_m1
                # -------------------------------
                Jz1_m0 += Sz_lower*Sr_lower*Jz_m0
                Jz1_m0 += Sz_lower*Sr_upper*Jz_m0
                Jz3_m0 += Sz_upper*Sr_lower*Jz_m0
                Jz3_m0 += Sz_upper*Sr_upper*Jz_m0
                # -------------------------------
                Jz1_m1 += Sz_lower*Sr_lower*Jz_m1
                Jz1_m1 += Sz_lower*Sr_upper*Jz_m1
                Jz3_m1 += Sz_upper*Sr_lower*Jz_m1
                Jz3_m1 += Sz_upper*Sr_upper*Jz_m1
                # -------------------------------
            if ir_lower != ir_upper:
                # In the case that ir_lower and ir_upper are different,
                # add the current to the four arrays according to
                # the direction.
                Jr1_m0 += Sz_lower*Sr_lower*Jr_m0
                Jr2_m0 += Sz_lower*Sr_upper*Jr_m0
                Jr3_m0 += Sz_upper*Sr_lower*Jr_m0
                Jr4_m0 += Sz_upper*Sr_upper*Jr_m0
                # -------------------------------
                Jr1_m1 += Sz_lower*Sr_lower*Jr_m1
                Jr2_m1 += Sz_lower*Sr_upper*Jr_m1
                Jr3_m1 += Sz_upper*Sr_lower*Jr_m1
                Jr4_m1 += Sz_upper*Sr_upper*Jr_m1
                # -------------------------------
                Jt1_m0 += Sz_lower*Sr_lower*Jt_m0
                Jt2_m0 += Sz_lower*Sr_upper*Jt_m0
                Jt3_m0 += Sz_upper*Sr_lower*Jt_m0
                Jt4_m0 += Sz_upper*Sr_upper*Jt_m0
                # -------------------------------
                Jt1_m1 += Sz_lower*Sr_lower*Jt_m1
                Jt2_m1 += Sz_lower*Sr_upper*Jt_m1
                Jt3_m1 += Sz_upper*Sr_lower*Jt_m1
                Jt4_m1 += Sz_upper*Sr_upper*Jt_m1
                # -------------------------------
                Jz1_m0 += Sz_lower*Sr_lower*Jz_m0
                Jz2_m0 += Sz_lower*Sr_upper*Jz_m0
                Jz3_m0 += Sz_upper*Sr_lower*Jz_m0
                Jz4_m0 += Sz_upper*Sr_upper*Jz_m0
                # -------------------------------
                Jz1_m1 += Sz_lower*Sr_lower*Jz_m1
                Jz2_m1 += Sz_lower*Sr_upper*Jz_m1
                Jz3_m1 += Sz_upper*Sr_lower*Jz_m1
                Jz4_m1 += Sz_upper*Sr_upper*Jz_m1
                # -------------------------------
            if ir_lower == ir_upper == 0:
                # Treat the guard cells.
                # Add the current to the guard cells
                # for particles that had an original 
                # cell index < 0.
                Jr1_m0 += -1.*Sz_lower*Sr_guard*Jr_m0
                Jr3_m0 += -1.*Sz_upper*Sr_guard*Jr_m0
                # -----------------------------------
                Jr1_m1 += -1.*Sz_lower*Sr_guard*Jr_m1
                Jr3_m1 += -1.*Sz_upper*Sr_guard*Jr_m1
                # -----------------------------------
                Jt1_m0 += -1.*Sz_lower*Sr_guard*Jt_m0
                Jt3_m0 += -1.*Sz_upper*Sr_guard*Jt_m0
                # -----------------------------------
                Jt1_m1 += -1.*Sz_lower*Sr_guard*Jt_m1
                Jt3_m1 += -1.*Sz_upper*Sr_guard*Jt_m1
                # -----------------------------------
                Jz1_m0 += -1.*Sz_lower*Sr_guard*Jz_m0
                Jz3_m0 += -1.*Sz_upper*Sr_guard*Jz_m0
                # -----------------------------------
                Jz1_m1 += -1.*Sz_lower*Sr_guard*Jz_m1
                Jz3_m1 += -1.*Sz_upper*Sr_guard*Jz_m1
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
            Jr_m0[i, j] +=  J0[i, j, 0] + \
                            J1[i, j-1, 0] + \
                            J2[i-1, j, 0] + \
                            J3[i-1, j-1, 0]

            Jr_m1[i, j] +=  J0[i, j, 1] + \
                            J1[i, j-1, 1] + \
                            J2[i-1, j, 1] + \
                            J3[i-1, j-1, 1]

            Jt_m0[i, j] +=  J0[i, j, 2] + \
                            J1[i, j-1, 2] + \
                            J2[i-1, j, 2] + \
                            J3[i-1, j-1, 2]

            Jt_m1[i, j] +=  J0[i, j, 3] + \
                            J1[i, j-1, 3] + \
                            J2[i-1, j, 3] + \
                            J3[i-1, j-1, 3]

            Jz_m0[i, j] +=  J0[i, j, 4] + \
                            J1[i, j-1, 4] + \
                            J2[i-1, j, 4] + \
                            J3[i-1, j-1, 4]

            Jz_m1[i, j] +=  J0[i, j, 5] + \
                            J1[i, j-1, 5] + \
                            J2[i-1, j, 5] + \
                            J3[i-1, j-1, 5]

# -----------------------------------------------------
# Sorting utilities - get_cell_idx / sort / prefix_sum
# -----------------------------------------------------

@cuda.jit('void(int32[:], uint32[:], \
                float64[:], float64[:], float64[:], \
                float64, float64, int32, \
                float64, float64, int32)')
def get_cell_idx_per_particle(cell_idx, sorted_idx,
                              x, y, z,
                              invdz, zmin, Nz, 
                              invdr, rmin, Nr):
    """
    Get the cell index of each particle.
    The cell index is 1d and calculated by:
    cell index in z + cell index in r * number of cells in z.
    The cell_idx of a particle is defined by 
    the lower cell in r and z, that it deposits its field to.

    Parameters
    ----------
    cell_idx : 1darray of integers
        The cell index of the particle

    sorted_idx : 1darray of integers
        The sorted index array needs to be reset
        before doing the sort

    x, y, z : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)
    
    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box, in each direction

    Nz, Nr : int
        Number of gridpoints along the considered direction
    """
    i = cuda.grid(1)
    if i < cell_idx.shape[0]:
            # Preliminary arrays for the cylindrical conversion
            xj = x[i]
            yj = y[i]
            zj = z[i]
            rj = math.sqrt( xj**2 + yj**2 )
    
            # Positions of the particles, in the cell unit
            r_cell =  invdr*(rj - rmin) - 0.5
            z_cell =  invdz*(zj - zmin) - 0.5

            # Original index of the uppper and lower cell
            ir_lower = int(math.floor( r_cell ))
            iz_lower = int(math.floor( z_cell ))

            # Treat the boundary conditions
            # guard cells in lower r
            if ir_lower < 0:
                ir_lower = 0          
            # absorbing in upper r
            if ir_lower > Nr-1:
                ir_lower = Nr-1
            # periodic boundaries in z
            if iz_lower < 0:
                iz_lower += Nz
            if iz_lower > Nz-1:
                iz_lower -= Nz
                
            # Reset sorted_idx array
            sorted_idx[i] = i
            # Calculate the 1D cell_idx by cell_idx_ir + cell_idx_iz * Nr
            cell_idx[i] = ir_lower + iz_lower * Nr

def sort_particles_per_cell(cell_idx, sorted_idx):
    """
    Sort the cell index of the particles and 
    modify the sorted index array accordingly.

    Parameters
    ----------
    cell_idx : 1darray of integers
        The cell index of the particle
    
    sorted_idx : 1darray of integers
        Represents the original index of the 
        particle before the sorting.
    """
    Ntot = cell_idx.shape[0]
    if Ntot > 0:
        sorter = sorting.RadixSort(Ntot, dtype = np.int32)
        sorter.sort(cell_idx, vals = sorted_idx)

@cuda.jit('void(int32[:], int32[:])')
def incl_prefix_sum(cell_idx, prefix_sum):
    """
    Perform an inclusive parallel prefix sum on the sorted 
    cell index array. The prefix sum array represents the
    cumulative sum of the number of particles per cell
    for each cell index.

    Parameters
    ----------
    cell_idx : 1darray of integers
        The cell index of the particle
    
    prefix_sum : 1darray of integers
        Represents the cumulative sum of 
        the particles per cell
    """
    # i is the index of the macroparticle
    i = cuda.grid(1)
    if i < cell_idx.shape[0]-1:
        # ci: index of the cell of the present macroparticle
        ci = cell_idx[i]
        # ci_next: index of the cell of the next macroparticle
        ci_next = cell_idx[i+1]
        # Fill all the cells between ci and ci_next with the
        # inclusive cumulative sum of the number particles until ci
        while ci < ci_next:
            # The cumulative sum of the number of particle per cell
            # until ci is i+1 (since i obeys python index, starting at 0)
            prefix_sum[ci] = i+1
            ci += 1
    # The last "macroparticle" of the cell_idx array fills up the 
    # rest of the prefix sum array
    if i == cell_idx.shape[0]-1:
        # Get the cell_index of the last macroparticle
        ci = cell_idx[i]
        # Fill all the remaining entries of the prefix sum array
        for empty_index in range(ci, prefix_sum.shape[0]):
            prefix_sum[empty_index] = i+1

@cuda.jit('void(int32[:])')
def reset_prefix_sum(prefix_sum):
    """
    Resets the prefix sum. Sets all the values
    to zero.

    Parameters
    ----------    
    prefix_sum : 1darray of integers
        Represents the cumulative sum of 
        the particles per cell
    """
    i = cuda.grid(1)
    if i < prefix_sum.shape[0]:
        prefix_sum[i] = 0

@cuda.jit('void(uint32[:], float64[:], float64[:])')
def write_sorting_buffer(sorted_idx, val, buf):
    """
    Writes the values of a particle array to a buffer,
    while rearranging them to match the sorted cell index array.

    Parameters
    ----------    
    sorted_idx : 1darray of integers
        Represents the original index of the 
        particle before the sorting

    val : 1d array of floats
        A particle data array

    buf : 1d array of floats
        A buffer array to temporarily store the
        sorted particle data array
    """
    i = cuda.grid(1)
    if i < val.shape[0]:
        buf[i] = val[sorted_idx[i]]

# -----------------------------------------------------
# Device array creation utility (will be removed later)
# -----------------------------------------------------

def cuda_deposition_arrays(Nz = None, Nr = None, fieldtype = None):
    """
    Create empty arrays on the GPU for the charge and 
    current deposition in each of the 4 possible direction.

    ###########################################
    # Needs to be moved to the fields package!
    ###########################################

    Parameters
    ----------    
    Nz : int
        Number of cells in z.
    Nr : int
        Number of cells in r.

    fieldtype : string 
        Either 'rho' or 'J'.
    """
    # Create empty arrays to store the four different possible
    # cell directions a particle can deposit to.
    if fieldtype == 'rho':
        # Rho - third dimension represents 2 modes
        rho0 = cuda.device_array(shape = (Nz, Nr, 2), dtype = np.complex128)
        rho1 = cuda.device_array(shape = (Nz, Nr, 2), dtype = np.complex128)
        rho2 = cuda.device_array(shape = (Nz, Nr, 2), dtype = np.complex128)
        rho3 = cuda.device_array(shape = (Nz, Nr, 2), dtype = np.complex128)
        return rho0, rho1, rho2, rho3

    if fieldtype == 'J':
        # J - third dimension represents 2 modes 
        # times 3 dimensions (r, t, z)
        J0 = cuda.device_array(shape = (Nz, Nr, 6), dtype = np.complex128)
        J1 = cuda.device_array(shape = (Nz, Nr, 6), dtype = np.complex128)
        J2 = cuda.device_array(shape = (Nz, Nr, 6), dtype = np.complex128)
        J3 = cuda.device_array(shape = (Nz, Nr, 6), dtype = np.complex128)
        return J0, J1, J2, J3
        
