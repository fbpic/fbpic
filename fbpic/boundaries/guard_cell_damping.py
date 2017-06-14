# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure that damps the fields in the guard cells.
"""
import numpy as np
# Check if CUDA is available, then import CUDA functions
from fbpic.cuda_utils import cuda_installed
if cuda_installed:
    from fbpic.cuda_utils import cuda, cuda_tpb_bpg_2d

class GuardCellDamper(object):
    """
    Class that handles the damping of the E and B fields in the guard cells,
    either on CPU or GPU.

    The damping is done in order to avoid:
    - sudden cutting of the stencil between MPI domains
    - wrapping around of the fields at open boundaries

    To mirror those two different goals, the damping is done differently
    in the guard cells that correspond to an open boundary and in the guard
    cells that correspond to a boundary between processors.
    """

    def __init__( self, n_guard, left_proc, right_proc,
                    exchange_period, n_order ):
        """
        Initialize a damping object.

        Parameters
        ----------
        n_guard: int
            The number of guard cells along z

        left_proc, right_proc: int or None
            Indicates whether the boundary is open (proc is None) or
            is a boundary between processors (proc is an integer)

        n_order: int
            The order of the stencil. If the stencil fits into the
            guard cells, no damping is performed, between two processors.
            (Damping is still performed in the guard cells that correspond
            to open boundaries)
        """
        # Register the number of guard cells
        self.n_guard = n_guard

        # Create the damping arrays
        self.left_damp = generate_damp_array(
            n_guard, n_order, left_proc, exchange_period )
        self.right_damp = generate_damp_array(
            n_guard, n_order, right_proc, exchange_period )

        # Transfer the damping array to the GPU
        if cuda_installed:
            self.d_left_damp = cuda.to_device( self.left_damp )
            self.d_right_damp = cuda.to_device( self.right_damp )


    def damp_guard_EB( self, interp ):
        """
        Damp the fields E and B in the guard cells.

        Parameters
        ----------
        interp: list of InterpolationGrid objects (one per azimuthal mode)
            Objects that contain the fields to be damped.
        """
        # Damp the fields on the CPU or the GPU
        if interp[0].use_cuda:
            # Damp the fields on the GPU
            dim_grid, dim_block = cuda_tpb_bpg_2d( self.n_guard, interp[0].Nr )

            cuda_damp_EB[dim_grid, dim_block](
                interp[0].Er, interp[0].Et, interp[0].Ez,
                interp[0].Br, interp[0].Bt, interp[0].Bz,
                interp[1].Er, interp[1].Et, interp[1].Ez,
                interp[1].Br, interp[1].Bt, interp[1].Bz,
                self.d_left_damp, self.d_right_damp, self.n_guard )

        else:
            # Damp the fields on the CPU
            n_guard = self.n_guard
            for m in range(len(interp)):
                # Damp the fields in left guard cells
                interp[m].Er[:n_guard,:] *= self.left_damp[:,np.newaxis]
                interp[m].Et[:n_guard,:] *= self.left_damp[:,np.newaxis]
                interp[m].Ez[:n_guard,:] *= self.left_damp[:,np.newaxis]
                interp[m].Br[:n_guard,:] *= self.left_damp[:,np.newaxis]
                interp[m].Bt[:n_guard,:] *= self.left_damp[:,np.newaxis]
                interp[m].Bz[:n_guard,:] *= self.left_damp[:,np.newaxis]
                # Damp the fields in right guard cells
                interp[m].Er[-n_guard:,:] *= self.right_damp[::-1,np.newaxis]
                interp[m].Et[-n_guard:,:] *= self.right_damp[::-1,np.newaxis]
                interp[m].Ez[-n_guard:,:] *= self.right_damp[::-1,np.newaxis]
                interp[m].Br[-n_guard:,:] *= self.right_damp[::-1,np.newaxis]
                interp[m].Bt[-n_guard:,:] *= self.right_damp[::-1,np.newaxis]
                interp[m].Bz[-n_guard:,:] *= self.right_damp[::-1,np.newaxis]

def generate_damp_array( n_guard, n_order, neighbor_proc, exchange_period ):
    """
    Create a 1d damping array of length n_guard.

    The expression of the damping array depends on whether the guard cells
    correspond to an open boundary or a boundary with another processor.

    Parameters
    ----------
    n_guard: int
        Number of guard cells along z

    n_order: int
        The order of the stencil, use -1 for infinite order.
        If the stencil fits into the guard cells, no damping is performed,
        between two processors. (Damping is still performed in the guard
        cells that correspond to open boundaries)

    neighbor_proc: int or None
        Indicate wether the present guard cells correspond to an open
        boundary (neighbor_proc = None) or a boundary with another
        processor (neighbor_proc is an integer)

    exchange_period: int
        The number of timestep before the moving window is moved.
        The larger this number, the lower the damping in the open boundaries.

    Returns
    -------
    A 1darray of doubles, of length n_guard, which represents the damping.
    """
    # Array of cell indices
    i_cell = np.arange( n_guard )

    # Boundary with a neighboring proc
    if neighbor_proc is not None:
        # If the stencil fits in the guard cells, do not perform any damping
        if (n_order!=-1) and (n_order/2 <= n_guard):
            damping_array = np.ones( n_guard )
        # If the stencil does not fit in guard cells, perform wide damping
        else:
            damping_array = np.where( i_cell < n_guard/2,
                    np.sin( i_cell * np.pi/n_guard )**2, 1. )

    # Open boundary
    elif neighbor_proc is None:
        # Perform narrow damping, with one quarter of the cells at 0
        damping_array = np.where( i_cell < n_guard/2,
                np.sin(2*(i_cell - n_guard/4)*np.pi/n_guard)**2, 1. )
        damping_array = np.where( i_cell < n_guard/4, 0., damping_array )

    return( damping_array )


if cuda_installed :
    @cuda.jit('void(complex128[:,:], complex128[:,:], complex128[:,:], \
                    complex128[:,:], complex128[:,:], complex128[:,:], \
                    complex128[:,:], complex128[:,:], complex128[:,:], \
                    complex128[:,:], complex128[:,:], complex128[:,:], \
                    float64[:], float64[:], int32)')
    def cuda_damp_EB( Er0, Et0, Ez0, Br0, Bt0, Bz0,
                      Er1, Et1, Ez1, Br1, Bt1, Bz1,
                      damp_array_left, damp_array_right, n_guard ) :
        """
        Multiply the E and B fields in the left and right guard cells
        by the arrays damp_array_left and damp_array_right.

        Parameters :
        ------------
        Er0, Et0, Ez0, Br0, Bt0, Bz0,
        Er1, Et1, Ez1, Br1, Bt1, Bz1 : 2darrays of complexs
            Contain the fields to be damped
            The first axis corresponds to z and the second to r

        damp_array_left, damp_array_right : 1darray of floats
            An array of length n_guards, which contains the damping factors

        n_guard: int
            Number of guard cells
        """
        # Obtain Cuda grid
        iz, ir = cuda.grid(2)

        # Obtain the size of the array along z and r
        Nz, Nr = Er0.shape

        # Modify the fields
        if ir < Nr :
            # Apply the damping arrays
            if iz < n_guard:
                damp_factor_left = damp_array_left[iz]
                damp_factor_right = damp_array_right[iz]

                # At the left end
                Er0[iz, ir] *= damp_factor_left
                Et0[iz, ir] *= damp_factor_left
                Ez0[iz, ir] *= damp_factor_left
                Br0[iz, ir] *= damp_factor_left
                Bt0[iz, ir] *= damp_factor_left
                Bz0[iz, ir] *= damp_factor_left
                Er1[iz, ir] *= damp_factor_left
                Et1[iz, ir] *= damp_factor_left
                Ez1[iz, ir] *= damp_factor_left
                Br1[iz, ir] *= damp_factor_left
                Bt1[iz, ir] *= damp_factor_left
                Bz1[iz, ir] *= damp_factor_left

                # At the right end
                iz_right = Nz - iz - 1
                Er0[iz_right, ir] *= damp_factor_right
                Et0[iz_right, ir] *= damp_factor_right
                Ez0[iz_right, ir] *= damp_factor_right
                Br0[iz_right, ir] *= damp_factor_right
                Bt0[iz_right, ir] *= damp_factor_right
                Bz0[iz_right, ir] *= damp_factor_right
                Er1[iz_right, ir] *= damp_factor_right
                Et1[iz_right, ir] *= damp_factor_right
                Ez1[iz_right, ir] *= damp_factor_right
                Br1[iz_right, ir] *= damp_factor_right
                Bt1[iz_right, ir] *= damp_factor_right
                Bz1[iz_right, ir] *= damp_factor_right
