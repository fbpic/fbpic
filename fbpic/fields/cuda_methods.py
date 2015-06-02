"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the optimized fields methods that use cuda on a GPU
"""

from numbapro import cuda

@cuda.jit('void(complex128[:,:], complex128[:,:]), int64, int64')
def cuda_erase( mode0, mode1, Nz, Nr ) :
    """
    Sets the two input arrays to 0

    These arrays are typically interpolation grid arrays, and they
    are set to zero before depositing the currents

    Parameters :
    ------------
    mode0, mode1 : 2darrays of complexs
       Arrays that represent the fields on the grid
       (The first axis corresponds to z and the second axis to r)

    Nz, Nr : ints
       The dimensions of the array
    """
    
    # Cuda 2D grid
    iz, ir = cuda.grid(2)

    # Set the elements of the array to 0
    if (iz < Nz) and (ir < Nr) :
        mode0[iz, ir] = 0
        mode1[iz, ir] = 0

    
