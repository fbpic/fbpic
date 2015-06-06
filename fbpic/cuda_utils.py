"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of generic functions that operate on a GPU.
"""
from numba import cuda

@cuda.jit('void(complex128[:,:], complex128[:,:], int32, int32)')
def cuda_copy( array_in, array_out, Nz, Nr ) :
    """
    Copy array_in to array_out, on the GPU
    
    This is typically done in cases where one of the two
    arrays is in C-order and the other one is in Fortran order.
    This conversion from C order to Fortran order is needed
    before using cuBlas functions.
        
    Parameters :
    ------------
    array_in, array_out : 2darray of complexs
        Arrays of shape (Nz, Nr)
    
    Nz, Nr : ints
        Dimensions of the arrays
    """

    # Set up cuda grid
    iz, ir = cuda.grid(2)
    
    # Copy from array_in to buffer_in
    if (iz < Nz) and (ir < Nr) :
        array_out[iz, ir] = array_in[iz, ir]

# -----------------------------------------------------
# CUDA grid utilities
# -----------------------------------------------------

def cuda_tpb_bpg_1d(x, TPB = 256):
    """
    # To-do: Add documentation.
    """
    # Calculates the needed blocks per grid
    BPG = int(x/TPB + 1)
    return BPG, TPB

def cuda_tpb_bpg_2d(x, y, TPBx = 8, TPBy = 8):
    """
    # To-do: Add documentation.
    """
    # Calculates the needed blocks per grid
    BPGx = int(x/TPBx + 1)
    BPGy = int(y/TPBy + 1)
    return (BPGx, BPGy), (TPBx, TPBy)