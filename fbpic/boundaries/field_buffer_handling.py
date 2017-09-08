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
        copy_vec_to_gpu_buffer, \
        replace_vec_from_gpu_buffer, \
        add_vec_from_gpu_buffer, \
        copy_scal_to_gpu_buffer, \
        replace_scal_from_gpu_buffer, \
        add_scal_from_gpu_buffer

class BufferHandler(object):
    """
    Class that handles the buffers when exchanging the fields
    between MPI domains.
    """

    def __init__( self, n_guard, Nr, left_proc, right_proc ):
        """
        Initialize the guard cell buffers for the fields.
        These buffers are used in order to group the MPI exchanges.

        Parameters
        ----------
        n_guard: int
           Number of guard cells

        Nr: int
           Number of points in the radial direction

        left_proc, right_proc: int or None
           Rank of the proc to the right and to the left
           (None for open boundary)
        """
        # Register parameters
        self.Nr = Nr
        self.n_guard = n_guard
        self.left_proc = left_proc
        self.right_proc = right_proc
        # Shortcut
        ng = self.n_guard
        # Allocate buffer arrays that are send via MPI to exchange
        # the fields between domains (either replacing or adding fields)
        # Buffers are allocated for the left and right side of the domain
        if not cuda_installed:
            # Allocate buffers on the CPU
            # - Replacing vector field buffers
            self.vec_rep_send_l = np.empty((6, ng, Nr), dtype=np.complex128)
            self.vec_rep_send_r = np.empty((6, ng, Nr), dtype=np.complex128)
            self.vec_rep_recv_l = np.empty((6, ng, Nr), dtype=np.complex128)
            self.vec_rep_recv_r = np.empty((6, ng, Nr), dtype=np.complex128)
            # - Adding vector field buffers
            self.vec_add_send_l = np.empty((6, 2*ng, Nr), dtype=np.complex128)
            self.vec_add_send_r = np.empty((6, 2*ng, Nr), dtype=np.complex128)
            self.vec_add_recv_l = np.empty((6, 2*ng, Nr), dtype=np.complex128)
            self.vec_add_recv_r = np.empty((6, 2*ng, Nr), dtype=np.complex128)
            # - Replacing scalar field buffers
            self.scal_rep_send_l = np.empty((2, ng, Nr), dtype=np.complex128)
            self.scal_rep_send_r = np.empty((2, ng, Nr), dtype=np.complex128)
            self.scal_rep_recv_l = np.empty((2, ng, Nr), dtype=np.complex128)
            self.scal_rep_recv_r = np.empty((2, ng, Nr), dtype=np.complex128)
            # - Adding scalar field buffers
            self.scal_add_send_l = np.empty((2, 2*ng, Nr), dtype=np.complex128)
            self.scal_add_send_r = np.empty((2, 2*ng, Nr), dtype=np.complex128)
            self.scal_add_recv_l = np.empty((2, 2*ng, Nr), dtype=np.complex128)
            self.scal_add_recv_r = np.empty((2, 2*ng, Nr), dtype=np.complex128)
        else:
            # Allocate buffers on the CPU and GPU
            # Use cuda.pinned_array so that CPU array is pagelocked.
            # (cannot be swapped out to disk and GPU can access it via DMA)
            pin_ary = cuda.pinned_array
            # - Replacing vector field buffers
            self.vec_rep_send_l = pin_ary((6, ng, Nr), dtype=np.complex128)
            self.vec_rep_send_r = pin_ary((6, ng, Nr), dtype=np.complex128)
            self.vec_rep_recv_l = pin_ary((6, ng, Nr), dtype=np.complex128)
            self.vec_rep_recv_r = pin_ary((6, ng, Nr), dtype=np.complex128)
            self.d_vec_rep_buffer_l = cuda.to_device( self.vec_rep_send_l )
            self.d_vec_rep_buffer_r = cuda.to_device( self.vec_rep_send_r )
            # - Adding vector field buffers
            self.vec_add_send_l = pin_ary((6, 2*ng, Nr), dtype=np.complex128)
            self.vec_add_send_r = pin_ary((6, 2*ng, Nr), dtype=np.complex128)
            self.vec_add_recv_l = pin_ary((6, 2*ng, Nr), dtype=np.complex128)
            self.vec_add_recv_r = pin_ary((6, 2*ng, Nr), dtype=np.complex128)
            self.d_vec_add_buffer_l = cuda.to_device( self.vec_add_send_l )
            self.d_vec_add_buffer_r = cuda.to_device( self.vec_add_send_r )
            # - Replacing scalar field buffers
            self.scal_rep_send_l = pin_ary((2, ng, Nr), dtype=np.complex128)
            self.scal_rep_send_r = pin_ary((2, ng, Nr), dtype=np.complex128)
            self.scal_rep_recv_l = pin_ary((2, ng, Nr), dtype=np.complex128)
            self.scal_rep_recv_r = pin_ary((2, ng, Nr), dtype=np.complex128)
            self.d_scal_rep_buffer_l = cuda.to_device( self.scal_rep_send_l )
            self.d_scal_rep_buffer_r = cuda.to_device( self.scal_rep_send_r )
            # - Adding scalar field buffers
            self.scal_add_send_l = pin_ary((2, 2*ng, Nr), dtype=np.complex128)
            self.scal_add_send_r = pin_ary((2, 2*ng, Nr), dtype=np.complex128)
            self.scal_add_recv_l = pin_ary((2, 2*ng, Nr), dtype=np.complex128)
            self.scal_add_recv_r = pin_ary((2, 2*ng, Nr), dtype=np.complex128)
            self.d_scal_add_buffer_l = cuda.to_device( self.scal_add_send_l )
            self.d_scal_add_buffer_r = cuda.to_device( self.scal_add_send_r )

    def handle_vec_buffer( self,
                           grid_0_r, grid_0_t, grid_0_z,
                           grid_1_r, grid_1_t, grid_1_z,
                           method, use_cuda,
                           before_sending=False, after_receiving=False):
        """
        Vector field buffer handling

        1) Copies data from the field grids to the MPI sending buffers
        -- or --
        2) Replaces or adds MPI sending buffers to the field grids

        For method 'replace':

        Either copy the inner part of the domain to the sending buffer
        for a vector field, or replace the receving buffer for a vector field
        to the guard cells of the domain.

        For method 'add':

        Either copy the inner part and the guard region of the domain to the
        sending buffer for a vector field, or add the receving buffer for the
        vector field to the guard cells and the inner region of the domain.

        Depending on whether the field data is initially on the CPU
        or on the GPU, this function will do the appropriate exchange
        with the device.

        Parameters
        ----------
        grid_m_x: InterpolationGrid objects
            6 Interpolation grid objects. One for each of the two modes (0, 1)
            and for each coordinate (r, t, z). (m = mode, x = coordinate)

        method: str
            Can either be 'replace' or 'add' depending on the type
            of field exchange that is needed

        use_cuda: bool
            Whether the simulation runs on GPUs. If True,
            the buffers are copied to the GPU arrays after the MPI exchange.

        before_sending: bool
            Whether to copy the inner part of the domain to the sending buffer

        after_receiving: bool
            Whether to copy the receiving buffer to the guard cells
        """
        # Define region that is copied to or from the buffer
        # depending on the method used.
        if method == 'replace':
            nz_start = self.n_guard
            nz_end = 2*self.n_guard
        if method == 'add':
            nz_start = 0
            nz_end = 2*self.n_guard
        # Whether or not to send to the left or right neighbor
        copy_left = (self.left_proc is not None)
        copy_right = (self.right_proc is not None)

        # When using the GPU
        if use_cuda:
            # Calculate the number of blocks and threads per block
            dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d(
                nz_end - nz_start, self.Nr )

            if before_sending:
                if method == 'replace':
                    # Copy the inner regions of the domain to the buffers
                    copy_vec_to_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                        self.d_vec_rep_buffer_l, self.d_vec_rep_buffer_r,
                        grid_0_r, grid_0_t, grid_0_z,
                        grid_1_r, grid_1_t, grid_1_z,
                        copy_left, copy_right, nz_start, nz_end )
                    # Copy the GPU buffers to the sending CPU buffers
                    if copy_left:
                        self.d_vec_rep_buffer_l.copy_to_host(
                            self.vec_rep_send_l )
                    if copy_right:
                        self.d_vec_rep_buffer_r.copy_to_host(
                            self.vec_rep_send_r )

                if method == 'add':
                    # Copy the inner+guard regions of the domain to the buffers
                    copy_vec_to_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                        self.d_vec_add_buffer_l, self.d_vec_add_buffer_r,
                        grid_0_r, grid_0_t, grid_0_z,
                        grid_1_r, grid_1_t, grid_1_z,
                        copy_left, copy_right, nz_start, nz_end )
                    # Copy the GPU buffers to the sending CPU buffers
                    if copy_left:
                        self.d_vec_add_buffer_l.copy_to_host(
                            self.vec_add_send_l )
                    if copy_right:
                        self.d_vec_add_buffer_r.copy_to_host(
                            self.vec_add_send_r )

            elif after_receiving:
                if method == 'replace':
                    # Copy the CPU receiving buffers to the GPU buffers
                    if copy_left:
                        self.d_vec_rep_buffer_l.copy_to_device(
                            self.vec_rep_recv_l )
                    if copy_right:
                        self.d_vec_rep_buffer_r.copy_to_device(
                            self.vec_rep_recv_r )
                    # Replace the guard cells of the domain with the buffers
                    replace_vec_from_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                        self.d_vec_rep_buffer_l, self.d_vec_rep_buffer_r,
                        grid_0_r, grid_0_t, grid_0_z,
                        grid_1_r, grid_1_t, grid_1_z,
                        copy_left, copy_right, nz_start, nz_end )

                if method == 'add':
                    # Copy the CPU receiving buffers to the GPU buffers
                    if copy_left:
                        self.d_vec_add_buffer_l.copy_to_device(
                            self.vec_add_recv_l )
                    if copy_right:
                        self.d_vec_add_buffer_r.copy_to_device(
                            self.vec_add_recv_r )
                    # Add the buffers to the domain
                    add_vec_from_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                        self.d_vec_add_buffer_l, self.d_vec_add_buffer_r,
                        grid_0_r, grid_0_t, grid_0_z,
                        grid_1_r, grid_1_t, grid_1_z,
                        copy_left, copy_right, nz_start, nz_end )
        # Without GPU
        else:
            if before_sending:
                if method == 'replace':
                    # Copy the inner regions of the domain to the buffers
                    if copy_left:
                        self.vec_rep_send_l[0,:,:]=grid_0_r[nz_start:nz_end,:]
                        self.vec_rep_send_l[1,:,:]=grid_0_t[nz_start:nz_end,:]
                        self.vec_rep_send_l[2,:,:]=grid_0_z[nz_start:nz_end,:]
                        self.vec_rep_send_l[3,:,:]=grid_1_r[nz_start:nz_end,:]
                        self.vec_rep_send_l[4,:,:]=grid_1_t[nz_start:nz_end,:]
                        self.vec_rep_send_l[5,:,:]=grid_1_z[nz_start:nz_end,:]
                    if copy_right:
                        self.vec_rep_send_r[0,:,:]=grid_0_r[::-1][nz_start:nz_end,:][::-1]
                        self.vec_rep_send_r[1,:,:]=grid_0_t[::-1][nz_start:nz_end,:][::-1]
                        self.vec_rep_send_r[2,:,:]=grid_0_z[::-1][nz_start:nz_end,:][::-1]
                        self.vec_rep_send_r[3,:,:]=grid_1_r[::-1][nz_start:nz_end,:][::-1]
                        self.vec_rep_send_r[4,:,:]=grid_1_t[::-1][nz_start:nz_end,:][::-1]
                        self.vec_rep_send_r[5,:,:]=grid_1_z[::-1][nz_start:nz_end,:][::-1]

                if method == 'add':
                    # Copy the inner+guard regions of the domain to the buffers
                    if copy_left:
                        self.vec_add_send_l[0,:,:]=grid_0_r[nz_start:nz_end,:]
                        self.vec_add_send_l[1,:,:]=grid_0_t[nz_start:nz_end,:]
                        self.vec_add_send_l[2,:,:]=grid_0_z[nz_start:nz_end,:]
                        self.vec_add_send_l[3,:,:]=grid_1_r[nz_start:nz_end,:]
                        self.vec_add_send_l[4,:,:]=grid_1_t[nz_start:nz_end,:]
                        self.vec_add_send_l[5,:,:]=grid_1_z[nz_start:nz_end,:]
                    if copy_right:
                        self.vec_add_send_r[0,:,:]=grid_0_r[::-1][nz_start:nz_end,:][::-1]
                        self.vec_add_send_r[1,:,:]=grid_0_t[::-1][nz_start:nz_end,:][::-1]
                        self.vec_add_send_r[2,:,:]=grid_0_z[::-1][nz_start:nz_end,:][::-1]
                        self.vec_add_send_r[3,:,:]=grid_1_r[::-1][nz_start:nz_end,:][::-1]
                        self.vec_add_send_r[4,:,:]=grid_1_t[::-1][nz_start:nz_end,:][::-1]
                        self.vec_add_send_r[5,:,:]=grid_1_z[::-1][nz_start:nz_end,:][::-1]

            elif after_receiving:
                if method == 'replace':
                    # Replace the guard cells of the domain with the buffers
                    if copy_left:
                        grid_0_r[:nz_end-nz_start,:]=self.vec_rep_recv_l[0,:,:]
                        grid_0_t[:nz_end-nz_start,:]=self.vec_rep_recv_l[1,:,:]
                        grid_0_z[:nz_end-nz_start,:]=self.vec_rep_recv_l[2,:,:]
                        grid_1_r[:nz_end-nz_start,:]=self.vec_rep_recv_l[3,:,:]
                        grid_1_t[:nz_end-nz_start,:]=self.vec_rep_recv_l[4,:,:]
                        grid_1_z[:nz_end-nz_start,:]=self.vec_rep_recv_l[5,:,:]
                    if copy_right:
                        grid_0_r[-(nz_end-nz_start):,:]=self.vec_rep_recv_r[0,:,:]
                        grid_0_t[-(nz_end-nz_start):,:]=self.vec_rep_recv_r[1,:,:]
                        grid_0_z[-(nz_end-nz_start):,:]=self.vec_rep_recv_r[2,:,:]
                        grid_1_r[-(nz_end-nz_start):,:]=self.vec_rep_recv_r[3,:,:]
                        grid_1_t[-(nz_end-nz_start):,:]=self.vec_rep_recv_r[4,:,:]
                        grid_1_z[-(nz_end-nz_start):,:]=self.vec_rep_recv_r[5,:,:]

                if method == 'add':
                    # Add buffers to the domain
                    if copy_left:
                        grid_0_r[:nz_end-nz_start,:]+=self.vec_add_recv_l[0,:,:]
                        grid_0_t[:nz_end-nz_start,:]+=self.vec_add_recv_l[1,:,:]
                        grid_0_z[:nz_end-nz_start,:]+=self.vec_add_recv_l[2,:,:]
                        grid_1_r[:nz_end-nz_start,:]+=self.vec_add_recv_l[3,:,:]
                        grid_1_t[:nz_end-nz_start,:]+=self.vec_add_recv_l[4,:,:]
                        grid_1_z[:nz_end-nz_start,:]+=self.vec_add_recv_l[5,:,:]
                    if copy_right:
                        grid_0_r[-(nz_end-nz_start):,:]+=self.vec_add_recv_r[0,:,:]
                        grid_0_t[-(nz_end-nz_start):,:]+=self.vec_add_recv_r[1,:,:]
                        grid_0_z[-(nz_end-nz_start):,:]+=self.vec_add_recv_r[2,:,:]
                        grid_1_r[-(nz_end-nz_start):,:]+=self.vec_add_recv_r[3,:,:]
                        grid_1_t[-(nz_end-nz_start):,:]+=self.vec_add_recv_r[4,:,:]
                        grid_1_z[-(nz_end-nz_start):,:]+=self.vec_add_recv_r[5,:,:]


    def handle_scal_buffer( self,
                            grid_0, grid_1,
                            method, use_cuda,
                            before_sending=False, after_receiving=False):
        """
        Scalar field buffer handling

        1) Copies data from the field grid to the MPI sending buffers
        -- or --
        2) Replaces or adds MPI sending buffers to the field grid

        For method 'replace':

        Either copy the inner part of the domain to the sending buffer
        for a scalar field, or replace the receving buffer for a scalar field
        to the guard cells of the domain.

        For method 'add':

        Either copy the inner part and the guard region of the domain to the
        sending buffer for a scalar field, or add the receving buffer for the
        scalar field to the guard cells and the inner region of the domain.

        Depending on whether the field data is initially on the CPU
        or on the GPU, this function will do the appropriate exchange
        with the device.

        Parameters
        ----------
        grid_m: InterpolationGrid objects
            2 Interpolation grid objects. One for each of the two modes (0, 1)
            (m = mode, x = coordinate)

        method: str
            Can either be 'replace' or 'add' depending on the type
            of field exchange that is needed

        use_cuda: bool
            Whether the simulation runs on GPUs. If True,
            the buffers are copied to the GPU arrays after the MPI exchange.

        before_sending: bool
            Whether to copy the inner part of the domain to the sending buffer

        after_receiving: bool
            Whether to copy the receiving buffer to the guard cells
        """
        if method == 'replace':
            nz_start = self.n_guard
            nz_end = 2*self.n_guard
        if method == 'add':
            nz_start = 0
            nz_end = 2*self.n_guard

        copy_left = (self.left_proc is not None)
        copy_right = (self.right_proc is not None)

        # When using the GPU
        if use_cuda:
            # Calculate the number of blocks and threads per block
            dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d(
                nz_end - nz_start, self.Nr )

            if before_sending:
                if method == 'replace':
                    # Copy the inner regions of the domain to the GPU buffers
                    copy_scal_to_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                        self.d_scal_rep_buffer_l, self.d_scal_rep_buffer_r,
                        grid_0, grid_1,
                        copy_left, copy_right, nz_start, nz_end )
                    # Copy the GPU buffers to the sending CPU buffers
                    if copy_left:
                        self.d_scal_rep_buffer_l.copy_to_host(
                            self.scal_rep_send_l )
                    if copy_right:
                        self.d_scal_rep_buffer_r.copy_to_host(
                            self.scal_rep_send_r )

                if method == 'add':
                    # Copy the inner+guard regions of the domain to the buffers
                    copy_scal_to_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                        self.d_scal_add_buffer_l, self.d_scal_add_buffer_r,
                        grid_0, grid_1,
                        copy_left, copy_right, nz_start, nz_end )
                    # Copy the GPU buffers to the sending CPU buffers
                    if copy_left:
                        self.d_scal_add_buffer_l.copy_to_host(
                            self.scal_add_send_l )
                    if copy_right:
                        self.d_scal_add_buffer_r.copy_to_host(
                            self.scal_add_send_r )

            elif after_receiving:
                if method == 'replace':
                    # Copy the CPU receiving buffers to the GPU buffers
                    if copy_left:
                        self.d_scal_rep_buffer_l.copy_to_device(
                            self.scal_rep_recv_l )
                    if copy_right:
                        self.d_scal_rep_buffer_r.copy_to_device(
                            self.scal_rep_recv_r )
                    # Replace the guard cells of the domain with the buffers
                    replace_scal_from_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                        self.d_scal_rep_buffer_l, self.d_scal_rep_buffer_r,
                        grid_0, grid_1,
                        copy_left, copy_right, nz_start, nz_end )

                if method == 'add':
                    # Copy the CPU receiving buffers to the GPU buffers
                    if copy_left:
                        self.d_scal_add_buffer_l.copy_to_device(
                            self.scal_add_recv_l )
                    if copy_right:
                        self.d_scal_add_buffer_r.copy_to_device(
                            self.scal_add_recv_r )
                    # Add the GPU buffers to the domain
                    add_scal_from_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                        self.d_scal_add_buffer_l, self.d_scal_add_buffer_r,
                        grid_0, grid_1,
                        copy_left, copy_right, nz_start, nz_end )
        # Without GPU
        else:
            if before_sending:
                if method == 'replace':
                    # Copy the inner regions of the domain to the buffer
                    if copy_left:
                        self.scal_rep_send_l[0,:,:]=grid_0[nz_start:nz_end,:]
                        self.scal_rep_send_l[1,:,:]=grid_1[nz_start:nz_end,:]
                    if copy_right:
                        self.scal_rep_send_r[0,:,:]=grid_0[::-1][nz_start:nz_end,:][::-1]
                        self.scal_rep_send_r[1,:,:]=grid_1[::-1][nz_start:nz_end,:][::-1]

                if method == 'add':
                    # Copy the inner+guard regions of the domain to the buffer
                    if copy_left:
                        self.scal_add_send_l[0,:,:]=grid_0[nz_start:nz_end,:]
                        self.scal_add_send_l[1,:,:]=grid_1[nz_start:nz_end,:]
                    if copy_right:
                        self.scal_add_send_r[0,:,:]=grid_0[::-1][nz_start:nz_end,:][::-1]
                        self.scal_add_send_r[1,:,:]=grid_1[::-1][nz_start:nz_end,:][::-1]

            elif after_receiving:
                if method == 'replace':
                    # Replace the guard cells of the domain with the buffers
                    if copy_left:
                        grid_0[:nz_end-nz_start,:]=self.scal_rep_recv_l[0,:,:]
                        grid_1[:nz_end-nz_start,:]=self.scal_rep_recv_l[1,:,:]
                    if copy_right:
                        grid_0[-(nz_end-nz_start):,:]=self.scal_rep_recv_r[0,:,:]
                        grid_1[-(nz_end-nz_start):,:]=self.scal_rep_recv_r[1,:,:]

                if method == 'add':
                    # Add buffers to the domain
                    if copy_left:
                        grid_0[:nz_end-nz_start,:]+=self.scal_add_recv_l[0,:,:]
                        grid_1[:nz_end-nz_start,:]+=self.scal_add_recv_l[1,:,:]
                    if copy_right:
                        grid_0[-(nz_end-nz_start):,:]+=self.scal_add_recv_r[0,:,:]
                        grid_1[-(nz_end-nz_start):,:]+=self.scal_add_recv_r[1,:,:]
