# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to handle mpi buffers for the fields
"""
import numpy as np
# Check if CUDA is available, then import CUDA functions
from fbpic.cuda_utils import cuda_installed
if cuda_installed:
    from fbpic.cuda_utils import cuda, cuda_tpb_bpg_2d
    from .cuda_methods import \
        copy_EB_to_gpu_buffers, copy_EB_from_gpu_buffers, \
        copy_J_to_gpu_buffers, add_J_from_gpu_buffers, \
        copy_rho_to_gpu_buffers, add_rho_from_gpu_buffers

class BufferHandler(object):
    """
    Class that handles the buffers when exchanging the fields
    between MPI domains.
    """

    def __init__( self, n_guard, Nr, Nm, left_proc, right_proc ):
        """
        Initialize the guard cell buffers for the fields.
        These buffers are used in order to group the MPI exchanges.

        Parameters
        ----------
        n_guard: int
           Number of guard cells

        Nr, Nm: int
           Number of points in the radial direction and
           number of azimuthal modes

        left_proc, right_proc: int or None
           Rank of the proc to the right and to the left
           (None for open boundary)
        """
        # Register parameters
        self.Nm = Nm
        self.n_guard = n_guard
        self.left_proc = left_proc
        self.right_proc = right_proc

        # For the E and B fields: Only the guard cells are exchanged
        # For J and rho: 2 * the guard cells are exchanged
        # - Sending buffer on the CPU at right and left of the box
        self.EB_send_r = np.empty((6*Nm, n_guard, Nr), dtype = np.complex128)
        self.EB_send_l = np.empty((6*Nm, n_guard, Nr), dtype = np.complex128)
        self.J_send_r = np.empty((3*Nm, 2*n_guard, Nr), dtype = np.complex128)
        self.J_send_l = np.empty((3*Nm, 2*n_guard, Nr), dtype = np.complex128)
        self.rho_send_r = np.empty((Nm, 2*n_guard, Nr), dtype = np.complex128)
        self.rho_send_l = np.empty((Nm, 2*n_guard, Nr), dtype = np.complex128)
        # - Receiving buffer on the CPU at right and left of the box
        self.EB_recv_r = np.empty((6*Nm, n_guard, Nr), dtype = np.complex128)
        self.EB_recv_l = np.empty((6*Nm, n_guard, Nr), dtype = np.complex128)
        self.J_recv_r = np.empty((3*Nm, 2*n_guard, Nr), dtype = np.complex128)
        self.J_recv_l = np.empty((3*Nm, 2*n_guard, Nr), dtype = np.complex128)
        self.rho_recv_r = np.empty((Nm, 2*n_guard, Nr), dtype = np.complex128)
        self.rho_recv_l = np.empty((Nm, 2*n_guard, Nr), dtype = np.complex128)
        # - Buffers on the GPU at right and left of the box
        if cuda_installed:
            self.d_EB_r = cuda.to_device( self.EB_send_r )
            self.d_EB_l = cuda.to_device( self.EB_send_l )
            self.d_J_r = cuda.to_device( self.J_send_r )
            self.d_J_l = cuda.to_device( self.J_send_l )
            self.d_rho_r = cuda.to_device( self.rho_send_r )
            self.d_rho_l = cuda.to_device( self.rho_send_l )

    def copy_EB_buffers( self, interp,
                         before_sending=False, after_receiving=False):
        """
        Either copy the inner part of the domain to the sending buffer
        for E & B, or copy the receving buffer for E & B to the guard
        cells of the domain.

        Depending on whether the field data is initially on the CPU
        or on the GPU, this function will do the appropriate exchange
        with the device.

        Parameters
        ----------
        interp: a list of InterpolationGrid objects
            (one element per azimuthal mode)

        before_sending: bool
            Whether to copy the inner part of the domain to the sending buffer

        after_receiving: bool
            Whether to copy the receiving buffer to the guard cells
        """
        # Shortcut for the guard cells
        ng = self.n_guard
        copy_left = (self.left_proc is not None)
        copy_right = (self.right_proc is not None)

        # When using the GPU
        if interp[0].use_cuda:

            # Calculate the number of blocks and threads per block
            dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d( ng, interp[0].Nr )

            if before_sending:
                # Copy the inner regions of the domain to the GPU buffers
                copy_EB_to_gpu_buffers[ dim_grid_2d, dim_block_2d ](
                    self.d_EB_l, self.d_EB_r,
                    interp[0].Er, interp[0].Et, interp[0].Ez,
                    interp[0].Br, interp[0].Bt, interp[0].Bz,
                    interp[1].Er, interp[1].Et, interp[1].Ez,
                    interp[1].Br, interp[1].Bt, interp[1].Bz,
                    copy_left, copy_right, ng )
                # Copy the GPU buffers to the sending CPU buffers
                if copy_left:
                    self.d_EB_l.copy_to_host( self.EB_send_l )
                if copy_right:
                    self.d_EB_r.copy_to_host( self.EB_send_r )

            elif after_receiving:
                # Copy the CPU receiving buffers to the GPU buffers
                if copy_left:
                    self.d_EB_l.copy_to_device( self.EB_recv_l )
                if copy_right:
                    self.d_EB_r.copy_to_device( self.EB_recv_r )
                # Copy the GPU buffers to the guard cells of the domain
                copy_EB_from_gpu_buffers[ dim_grid_2d, dim_block_2d ](
                    self.d_EB_l, self.d_EB_r,
                    interp[0].Er, interp[0].Et, interp[0].Ez,
                    interp[0].Br, interp[0].Bt, interp[0].Bz,
                    interp[1].Er, interp[1].Et, interp[1].Ez,
                    interp[1].Br, interp[1].Bt, interp[1].Bz,
                    copy_left, copy_right, ng )

        # Without GPU
        else:
            for m in range(self.Nm):
                offset = 6*m

                if before_sending:
                    # Copy the inner regions of the domain to the buffer
                    if copy_left:
                        self.EB_send_l[0+offset,:,:] = interp[m].Er[ng:2*ng,:]
                        self.EB_send_l[1+offset,:,:] = interp[m].Et[ng:2*ng,:]
                        self.EB_send_l[2+offset,:,:] = interp[m].Ez[ng:2*ng,:]
                        self.EB_send_l[3+offset,:,:] = interp[m].Br[ng:2*ng,:]
                        self.EB_send_l[4+offset,:,:] = interp[m].Bt[ng:2*ng,:]
                        self.EB_send_l[5+offset,:,:] = interp[m].Bz[ng:2*ng,:]
                    if copy_right:
                        self.EB_send_r[0+offset,:,:]= interp[m].Er[-2*ng:-ng,:]
                        self.EB_send_r[1+offset,:,:]= interp[m].Et[-2*ng:-ng,:]
                        self.EB_send_r[2+offset,:,:]= interp[m].Ez[-2*ng:-ng,:]
                        self.EB_send_r[3+offset,:,:]= interp[m].Br[-2*ng:-ng,:]
                        self.EB_send_r[4+offset,:,:]= interp[m].Bt[-2*ng:-ng,:]
                        self.EB_send_r[5+offset,:,:]= interp[m].Bz[-2*ng:-ng,:]

                elif after_receiving:
                    # Copy the buffer to the guard cells of the domain
                    if copy_left:
                        interp[m].Er[:ng,:] = self.EB_recv_l[0+offset,:,:]
                        interp[m].Et[:ng,:] = self.EB_recv_l[1+offset,:,:]
                        interp[m].Ez[:ng,:] = self.EB_recv_l[2+offset,:,:]
                        interp[m].Br[:ng,:] = self.EB_recv_l[3+offset,:,:]
                        interp[m].Bt[:ng,:] = self.EB_recv_l[4+offset,:,:]
                        interp[m].Bz[:ng,:] = self.EB_recv_l[5+offset,:,:]
                    if copy_right:
                        interp[m].Er[-ng:,:] = self.EB_recv_r[0+offset,:,:]
                        interp[m].Et[-ng:,:] = self.EB_recv_r[1+offset,:,:]
                        interp[m].Ez[-ng:,:] = self.EB_recv_r[2+offset,:,:]
                        interp[m].Br[-ng:,:] = self.EB_recv_r[3+offset,:,:]
                        interp[m].Bt[-ng:,:] = self.EB_recv_r[4+offset,:,:]
                        interp[m].Bz[-ng:,:] = self.EB_recv_r[5+offset,:,:]

    def copy_J_buffers( self, interp,
                        before_sending=False, after_receiving=False):
        """
        Either copy the inner part of the domain to the sending buffer for J,
        or add the receving buffer for J to the guard cells of the domain.

        Depending on whether the field data is initially on the CPU
        or on the GPU, this function will do the appropriate exchange
        with the device.

        Parameters
        ----------
        interp: a list of InterpolationGrid objects
            (one element per azimuthal mode)

        before_sending: bool
            Whether to copy the inner part of the domain to the sending buffer

        after_receiving: bool
            Whether to add the receiving buffer to the guard cells
        """
        # Shortcut for the guard cells
        ng = self.n_guard
        copy_left = (self.left_proc is not None)
        copy_right = (self.right_proc is not None)

        # When using the GPU
        if interp[0].use_cuda:

            # Calculate the number of blocks and threads per block
            dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d( 2*ng, interp[0].Nr )

            if before_sending:
                # Copy the inner regions of the domain to the GPU buffers
                copy_J_to_gpu_buffers[ dim_grid_2d, dim_block_2d ](
                    self.d_J_l, self.d_J_r,
                    interp[0].Jr, interp[0].Jt, interp[0].Jz,
                    interp[1].Jr, interp[1].Jt, interp[1].Jz,
                    copy_left, copy_right, ng )
                # Copy the GPU buffers to the CPU sending buffers
                if copy_left:
                    self.d_J_l.copy_to_host( self.J_send_l )
                if copy_right:
                    self.d_J_r.copy_to_host( self.J_send_r )

            elif after_receiving:
                # Copy the CPU receiving buffers to the GPU buffers
                if copy_left:
                    self.d_J_l.copy_to_device( self.J_recv_l )
                if copy_right:
                    self.d_J_r.copy_to_device( self.J_recv_r )
                # Add the GPU buffers to the guard cells of the domain
                add_J_from_gpu_buffers[ dim_grid_2d, dim_block_2d ](
                    self.d_J_l, self.d_J_r,
                    interp[0].Jr, interp[0].Jt, interp[0].Jz,
                    interp[1].Jr, interp[1].Jt, interp[1].Jz,
                    copy_left, copy_right, ng )

        # Without GPU
        else:
            for m in range(self.Nm):
                offset = 3*m

                if before_sending:
                    # Copy the inner region of the domain to the buffer
                    if copy_left:
                        self.J_send_l[0+offset,:,:] = interp[m].Jr[:2*ng,:]
                        self.J_send_l[1+offset,:,:] = interp[m].Jt[:2*ng,:]
                        self.J_send_l[2+offset,:,:] = interp[m].Jz[:2*ng,:]
                    if copy_right:
                        self.J_send_r[0+offset,:,:] = interp[m].Jr[-2*ng:,:]
                        self.J_send_r[1+offset,:,:] = interp[m].Jt[-2*ng:,:]
                        self.J_send_r[2+offset,:,:] = interp[m].Jz[-2*ng:,:]

                elif after_receiving:
                    # Add the buffer to the guard cells of the domain
                    if copy_left:
                        interp[m].Jr[:2*ng,:] += self.J_recv_l[0+offset,:,:]
                        interp[m].Jt[:2*ng,:] += self.J_recv_l[1+offset,:,:]
                        interp[m].Jz[:2*ng,:] += self.J_recv_l[2+offset,:,:]
                    if copy_right:
                        interp[m].Jr[-2*ng:,:] += self.J_recv_r[0+offset,:,:]
                        interp[m].Jt[-2*ng:,:] += self.J_recv_r[1+offset,:,:]
                        interp[m].Jz[-2*ng:,:] += self.J_recv_r[2+offset,:,:]

    def copy_rho_buffers( self, interp,
                          before_sending=False, after_receiving=False):
        """
        Either copy the inner part of the domain to the sending buffer for rho,
        or add the receving buffer for rho to the guard cells of the domain.

        Depending on whether the field data is initially on the CPU
        or on the GPU, this function will do the appropriate exchange
        with the device.

        Parameters
        ----------
        interp: a list of InterpolationGrid objects
            (one element per azimuthal mode)

        before_sending: bool
            Whether to copy the inner part of the domain to the sending buffer

        after_receiving: bool
            Whether to add the receiving buffer to the guard cells
        """
        # Shortcut for the guard cells
        ng = self.n_guard
        copy_left = (self.left_proc is not None)
        copy_right = (self.right_proc is not None)

        # When using the GPU
        if interp[0].use_cuda:

            # Calculate the number of blocks and threads per block
            dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d( 2*ng, interp[0].Nr )

            if before_sending:
                # Copy the inner regions of the domain to the GPU buffers
                copy_rho_to_gpu_buffers[ dim_grid_2d, dim_block_2d ](
                    self.d_rho_l, self.d_rho_r,
                    interp[0].rho, interp[1].rho, copy_left, copy_right, ng )
                # Copy the GPU buffers to the sending CPU buffers
                if copy_left:
                    self.d_rho_l.copy_to_host( self.rho_send_l )
                if copy_right:
                    self.d_rho_r.copy_to_host( self.rho_send_r )

            elif after_receiving:
                # Copy the receiving CPU buffers to the GPU buffers
                if copy_left:
                    self.d_rho_l.copy_to_device( self.rho_recv_l )
                if copy_right:
                    self.d_rho_r.copy_to_device( self.rho_recv_r )
                # Add the GPU buffers to the guard cells of the domain
                add_rho_from_gpu_buffers[ dim_grid_2d, dim_block_2d ](
                    self.d_rho_l, self.d_rho_r,
                    interp[0].rho, interp[1].rho, copy_left, copy_right, ng )

        # Without GPU
        else:
            for m in range(self.Nm):
                offset = 1*m

                if before_sending:
                    # Copy the inner regions of the domain to the buffer
                    if copy_left:
                        self.rho_send_l[0+offset,:,:] = interp[m].rho[:2*ng,:]
                    if copy_right:
                        self.rho_send_r[0+offset,:,:] = interp[m].rho[-2*ng:,:]

                elif after_receiving:
                    # Add the buffer to the guard cells of the domain
                    if copy_left:
                        interp[m].rho[:2*ng,:] += self.rho_recv_l[0+offset,:,:]
                    if copy_right:
                        interp[m].rho[-2*ng:,:]+= self.rho_recv_r[0+offset,:,:]
