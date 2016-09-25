# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of generic functions that operate on a GPU.
"""
from numba import cuda

@cuda.jit('void( complex128[:,:,:], complex128[:,:,:], \
                 complex128[:,:], complex128[:,:], complex128[:,:], \
                 complex128[:,:], complex128[:,:], complex128[:,:], \
                 complex128[:,:], complex128[:,:], complex128[:,:], \
                 complex128[:,:], complex128[:,:], complex128[:,:], \
                 int32, int32, int32 )')
def copy_EB_to_gpu_buffers( EB_left, EB_right,
                            Er0, Et0, Ez0, Br0, Bt0, Bz0,
                            Er1, Et1, Ez1, Br1, Bt1, Bz1,
                            copy_left, copy_right, ng ):
    """
    Copy the ng inner domain cells of Er0, ..., Bz1
    to the GPU buffer EB_left and EB_right

    Parameters
    ----------
    EB_left, EB_right: ndarrays of complexs (device arrays)
        Arrays of shape (12, ng, Nr), which serve as buffer for transmission
        to CPU, and then sending via MPI. They are to hold the values of
        E&B in the ng inner cells of the domain, to the left and right.

    Er0, Et0, Ez0, Er1, Et1, Ez1: ndarrays of complexs (device arrays)
        Arrays of shape (Nz, Nr), which contain the different component
        of the E field, in the modes 0 and 1.

    Br0, Bt0, Bz0, Br1, Bt1, Bz1: ndarrays of complexs (device arrays)
        Arrays of shape (Nz, Nr), which contain the different component
        of the B field, in the modes 0 and 1.

    copy_left, copy_right: bool
        Whether to copy the buffers to the left and right of the local domain
        (Buffers are not copied at the left end and right end of the
        global simulation box.)

    ng: int
        Number of guard cells in the longitudinal direction
    """
    # Dimension of the arrays
    Nz, Nr = Er0.shape

    # Obtain Cuda grid
    iz, ir = cuda.grid(2)

    # Offset between modes, for the EB buffer
    offset = 6
    
    # Copy the inner regions of the domain to the buffer
    if ir < Nr:
        if iz < ng:
            # At the left end
            if copy_left:
                iz_left = ng + iz
                EB_left[0, iz, ir] = Er0[ iz_left, ir ]
                EB_left[1, iz, ir] = Et0[ iz_left, ir ]
                EB_left[2, iz, ir] = Ez0[ iz_left, ir ]
                EB_left[3, iz, ir] = Br0[ iz_left, ir ]
                EB_left[4, iz, ir] = Bt0[ iz_left, ir ]
                EB_left[5, iz, ir] = Bz0[ iz_left, ir ]
                EB_left[0+offset, iz, ir] = Er1[ iz_left, ir ]
                EB_left[1+offset, iz, ir] = Et1[ iz_left, ir ]
                EB_left[2+offset, iz, ir] = Ez1[ iz_left, ir ]
                EB_left[3+offset, iz, ir] = Br1[ iz_left, ir ]
                EB_left[4+offset, iz, ir] = Bt1[ iz_left, ir ]
                EB_left[5+offset, iz, ir] = Bz1[ iz_left, ir ]
            # At the right end
            if copy_right:
                iz_right = Nz - 2*ng + iz
                EB_right[0, iz, ir] = Er0[ iz_right, ir ]
                EB_right[1, iz, ir] = Et0[ iz_right, ir ]
                EB_right[2, iz, ir] = Ez0[ iz_right, ir ]
                EB_right[3, iz, ir] = Br0[ iz_right, ir ]
                EB_right[4, iz, ir] = Bt0[ iz_right, ir ]
                EB_right[5, iz, ir] = Bz0[ iz_right, ir ]
                EB_right[0+offset, iz, ir] = Er1[ iz_right, ir ]
                EB_right[1+offset, iz, ir] = Et1[ iz_right, ir ]
                EB_right[2+offset, iz, ir] = Ez1[ iz_right, ir ]
                EB_right[3+offset, iz, ir] = Br1[ iz_right, ir ]
                EB_right[4+offset, iz, ir] = Bt1[ iz_right, ir ]
                EB_right[5+offset, iz, ir] = Bz1[ iz_right, ir ]

        
@cuda.jit('void( complex128[:,:,:], complex128[:,:,:], \
                 complex128[:,:], complex128[:,:], complex128[:,:], \
                 complex128[:,:], complex128[:,:], complex128[:,:], \
                 complex128[:,:], complex128[:,:], complex128[:,:], \
                 complex128[:,:], complex128[:,:], complex128[:,:], \
                 int32, int32, int32 )')
def copy_EB_from_gpu_buffers( EB_left, EB_right,
                            Er0, Et0, Ez0, Br0, Bt0, Bz0,
                            Er1, Et1, Ez1, Br1, Bt1, Bz1,
                            copy_left, copy_right, ng ):
    """
    Copy the GPU buffer EB_left and EB_right to the ng guards
    cells of Er0, ..., Bz1

    Parameters
    ----------
    EB_left, EB_right: ndarrays of complexs (device arrays)
        Arrays of shape (12, ng, Nr), which serve as buffer for transmission
        from CPU, and after receiving via MPI. They hold the values of
        E&B in the ng inner cells of the domain, to the left and right.

    Er0, Et0, Ez0, Er1, Et1, Ez1: ndarrays of complexs (device arrays)
        Arrays of shape (Nz, Nr), which contain the different components
        of the E field, in the modes 0 and 1.

    Br0, Bt0, Bz0, Br1, Bt1, Bz1: ndarrays of complexs (device arrays)
        Arrays of shape (Nz, Nr), which contain the different components
        of the B field, in the modes 0 and 1.

    copy_left, copy_right: bool
        Whether to copy the buffers to the left and right of the local domain
        (Buffers are not copied at the left end and right end of the
        global simulation box.)

    ng: int
        Number of guard cells in the longitudinal direction
    """
    # Dimension of the arrays
    Nz, Nr = Er0.shape

    # Obtain Cuda grid
    iz, ir = cuda.grid(2)

    # Offset between modes, for the EB buffer
    offset = 6
    
    # Copy the GPU buffers to the guard cells of the domain
    if ir < Nr:
        if iz < ng:
            # At the left end
            if copy_left:
                iz_left = iz
                Er0[ iz_left, ir ] = EB_left[0, iz, ir]
                Et0[ iz_left, ir ] = EB_left[1, iz, ir]
                Ez0[ iz_left, ir ] = EB_left[2, iz, ir]
                Br0[ iz_left, ir ] = EB_left[3, iz, ir]
                Bt0[ iz_left, ir ] = EB_left[4, iz, ir]
                Bz0[ iz_left, ir ] = EB_left[5, iz, ir]
                Er1[ iz_left, ir ] = EB_left[0+offset, iz, ir]
                Et1[ iz_left, ir ] = EB_left[1+offset, iz, ir]
                Ez1[ iz_left, ir ] = EB_left[2+offset, iz, ir]
                Br1[ iz_left, ir ] = EB_left[3+offset, iz, ir]
                Bt1[ iz_left, ir ] = EB_left[4+offset, iz, ir]
                Bz1[ iz_left, ir ] = EB_left[5+offset, iz, ir]
            # At the right end
            if copy_right:
                iz_right = Nz - ng + iz
                Er0[ iz_right, ir ] = EB_right[0, iz, ir]
                Et0[ iz_right, ir ] = EB_right[1, iz, ir]
                Ez0[ iz_right, ir ] = EB_right[2, iz, ir]
                Br0[ iz_right, ir ] = EB_right[3, iz, ir]
                Bt0[ iz_right, ir ] = EB_right[4, iz, ir]
                Bz0[ iz_right, ir ] = EB_right[5, iz, ir]
                Er1[ iz_right, ir ] = EB_right[0+offset, iz, ir]
                Et1[ iz_right, ir ] = EB_right[1+offset, iz, ir]
                Ez1[ iz_right, ir ] = EB_right[2+offset, iz, ir]
                Br1[ iz_right, ir ] = EB_right[3+offset, iz, ir]
                Bt1[ iz_right, ir ] = EB_right[4+offset, iz, ir]
                Bz1[ iz_right, ir ] = EB_right[5+offset, iz, ir]


@cuda.jit('void( complex128[:,:,:], complex128[:,:,:], \
                 complex128[:,:], complex128[:,:], complex128[:,:], \
                 complex128[:,:], complex128[:,:], complex128[:,:], \
                 int32, int32, int32 )')
def copy_J_to_gpu_buffers( J_left, J_right,
                            Jr0, Jt0, Jz0, Jr1, Jt1, Jz1, 
                            copy_left, copy_right, ng ):
    """
    Copy the 2*ng outermost cells of Jr0, ..., Jz1 to the GPU buffers
    J_left and J_right

    Parameters
    ----------
    J_left, J_right: ndarrays of complexs (device arrays)
        Arrays of shape (6, 2*ng, Nr), which serve as buffer for transmission
        to CPU, and then sending via MPI. They hold the values of
        J in the ng inner cells of the domain, to the left and right.

    Jr0, Jt0, Jz0, Jr1, Jt1, Jz1: ndarrays of complexs (device arrays)
        Arrays of shape (Nz, Nr), which contain the different components
        of the J field, in the modes 0 and 1.

    copy_left, copy_right: bool
        Whether to copy the buffers from the left and right of the local domain
        (Buffers are not copied at the left end and right end of the
        global simulation box.)

    ng: int
        Number of guard cells in the longitudinal direction
    """
    # Dimension of the arrays
    Nz, Nr = Jr0.shape

    # Obtain Cuda grid
    iz, ir = cuda.grid(2)

    # Offset between modes, for the J buffer
    offset = 3
    
    # Copy the inner regions of the domain to the buffer
    if ir < Nr:
        if iz < 2*ng:
            # At the left end
            if copy_left:
                iz_left = iz
                J_left[0, iz, ir] = Jr0[ iz_left, ir ]
                J_left[1, iz, ir] = Jt0[ iz_left, ir ]
                J_left[2, iz, ir] = Jz0[ iz_left, ir ]
                J_left[0+offset, iz, ir] = Jr1[ iz_left, ir ]
                J_left[1+offset, iz, ir] = Jt1[ iz_left, ir ]
                J_left[2+offset, iz, ir] = Jz1[ iz_left, ir ]

            # At the right end
            if copy_right:
                iz_right = Nz - 2*ng + iz
                J_right[0, iz, ir] = Jr0[ iz_right, ir ]
                J_right[1, iz, ir] = Jt0[ iz_right, ir ]
                J_right[2, iz, ir] = Jz0[ iz_right, ir ]
                J_right[0+offset, iz, ir] = Jr1[ iz_right, ir ]
                J_right[1+offset, iz, ir] = Jt1[ iz_right, ir ]
                J_right[2+offset, iz, ir] = Jz1[ iz_right, ir ]


@cuda.jit('void( complex128[:,:,:], complex128[:,:,:], \
                 complex128[:,:], complex128[:,:], complex128[:,:], \
                 complex128[:,:], complex128[:,:], complex128[:,:], \
                 int32, int32, int32 )')
def add_J_from_gpu_buffers( J_left, J_right,
                            Jr0, Jt0, Jz0, Jr1, Jt1, Jz1, 
                            copy_left, copy_right, ng ):
    """
    Add the GPU buffer J_left and J_right to the 2*ng outermost 
    cells of Jr0, ..., Jz1

    Parameters
    ----------
    J_left, J_right: ndarrays of complexs (device arrays)
        Arrays of shape (6, 2*ng, Nr), which serve as buffer for transmission
        from CPU, and after receiving via MPI. They hold the values of
        J in the ng inner cells of the domain, to the left and right.

    Jr0, Jt0, Jz0, Jr1, Jt1, Jz1: ndarrays of complexs (device arrays)
        Arrays of shape (Nz, Nr), which contain the different components
        of the J field, in the modes 0 and 1.

    copy_left, copy_right: bool
        Whether to add the buffers to the left and right of the local domain
        (Buffers are not added at the left end and right end of the
        global simulation box.)

    ng: int
        Number of guard cells in the longitudinal direction
    """
    # Dimension of the arrays
    Nz, Nr = Jr0.shape

    # Obtain Cuda grid
    iz, ir = cuda.grid(2)

    # Offset between modes, for the J buffer
    offset = 3
    
    # Add the GPU buffer to the guard cells of the domain
    if ir < Nr:
        if iz < 2*ng:
            # At the left end
            if copy_left:
                iz_left = iz
                Jr0[ iz_left, ir ] += J_left[0, iz, ir]
                Jt0[ iz_left, ir ] += J_left[1, iz, ir]
                Jz0[ iz_left, ir ] += J_left[2, iz, ir]
                Jr1[ iz_left, ir ] += J_left[0+offset, iz, ir]
                Jt1[ iz_left, ir ] += J_left[1+offset, iz, ir]
                Jz1[ iz_left, ir ] += J_left[2+offset, iz, ir]
            # At the right end
            if copy_right:
                iz_right = Nz - 2*ng + iz
                Jr0[ iz_right, ir ] += J_right[0, iz, ir]
                Jt0[ iz_right, ir ] += J_right[1, iz, ir]
                Jz0[ iz_right, ir ] += J_right[2, iz, ir]
                Jr1[ iz_right, ir ] += J_right[0+offset, iz, ir]
                Jt1[ iz_right, ir ] += J_right[1+offset, iz, ir]
                Jz1[ iz_right, ir ] += J_right[2+offset, iz, ir]


@cuda.jit('void( complex128[:,:,:], complex128[:,:,:], \
                 complex128[:,:], complex128[:,:], \
                 int32, int32, int32 )')
def copy_rho_to_gpu_buffers( rho_left, rho_right, rho0, rho1,
                            copy_left, copy_right, ng ):
    """
    Copy the 2*ng outermost cells of rho0, rho1 to the GPU buffers
    rho_left and rho_right

    Parameters
    ----------
    rho_left, rho_right: ndarrays of complexs (device arrays)
        Arrays of shape (2, 2*ng, Nr), which serve as buffer for transmission
        to CPU, and then sending via MPI. They hold the values of
        rho in the ng inner cells of the domain, to the left and right.

    rho0, rho1: ndarrays of complexs (device arrays)
        Arrays of shape (Nz, Nr), which contain the rho field,
        in the modes 0 and 1.

    copy_left, copy_right: bool
        Whether to copy the buffers from the left and right of the local domain
        (Buffers are not copied at the left end and right end of the
        global simulation box.)

    ng: int
        Number of guard cells in the longitudinal direction
    """
    # Dimension of the arrays
    Nz, Nr = rho0.shape

    # Obtain Cuda grid
    iz, ir = cuda.grid(2)

    # Offset between modes, for the rho buffer
    offset = 1
    
    # Copy the inner regions of the domain to the buffer
    if ir < Nr:
        if iz < 2*ng:
            # At the left end
            if copy_left:
                iz_left = iz
                rho_left[0, iz, ir] = rho0[ iz_left, ir ]
                rho_left[0+offset, iz, ir] = rho1[ iz_left, ir ]

            # At the right end
            if copy_right:
                iz_right = Nz - 2*ng + iz
                rho_right[0, iz, ir] = rho0[ iz_right, ir ]
                rho_right[0+offset, iz, ir] = rho1[ iz_right, ir ]


@cuda.jit('void( complex128[:,:,:], complex128[:,:,:], \
                 complex128[:,:], complex128[:,:], \
                 int32, int32, int32 )')
def add_rho_from_gpu_buffers( rho_left, rho_right, rho0, rho1,
                            copy_left, copy_right, ng ):
    """
    Add the GPU buffers rho_left and rho_right to the 2*ng outermost
    cells of rho0, rho1

    Parameters
    ----------
    rho_left, rho_right: ndarrays of complexs (device arrays)
        Arrays of shape (2, 2*ng, Nr), which serve as buffer for transmission
        from CPU, after receiving via MPI. They hold the values of
        rho in the ng inner cells of the domain, to the left and right.

    rho0, rho1: ndarrays of complexs (device arrays)
        Arrays of shape (Nz, Nr), which contain the rho field,
        in the modes 0 and 1.

    copy_left, copy_right: bool
        Whether to add the buffers from the left and right to the local domain
        (Buffers are not copied at the left end and right end of the
        global simulation box.)

    ng: int
        Number of guard cells in the longitudinal direction
    """
    # Dimension of the arrays
    Nz, Nr = rho0.shape

    # Obtain Cuda grid
    iz, ir = cuda.grid(2)

    # Offset between modes, for the rho buffer
    offset = 1
    
    # Add the GPU buffer to the guard cells of the domain
    if ir < Nr:
        if iz < 2*ng:
            # At the left end
            if copy_left:
                iz_left = iz
                rho0[ iz_left, ir ] += rho_left[0, iz, ir]
                rho1[ iz_left, ir ] += rho_left[0+offset, iz, ir]
            # At the right end
            if copy_right:
                iz_right = Nz - 2*ng + iz
                rho0[ iz_right, ir ] += rho_right[0, iz, ir]
                rho1[ iz_right, ir ] += rho_right[0+offset, iz, ir]
