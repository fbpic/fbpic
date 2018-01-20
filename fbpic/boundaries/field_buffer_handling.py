# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to handle mpi buffers for the fields
"""
import numpy as np
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    from fbpic.utils.cuda import cuda, cuda_tpb_bpg_2d
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

    def __init__( self, n_guard, Nr, Nm, left_proc, right_proc ):
        """
        Initialize the guard cell buffers for the fields.
        These buffers are used in order to group the MPI exchanges.

        Parameters
        ----------
        n_guard: int
           Number of guard cells

        Nr: int
           Number of points in the radial direction

        Nm: int
           Number of azimuthal modes

        left_proc, right_proc: int or None
           Rank of the proc to the right and to the left
           (None for open boundary)
        """
        # Register parameters
        self.Nr = Nr
        self.Nm = Nm
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
            self.vec_rep_send_l = np.empty((3*Nm,ng,Nr), dtype=np.complex128)
            self.vec_rep_send_r = np.empty((3*Nm,ng,Nr), dtype=np.complex128)
            self.vec_rep_recv_l = np.empty((3*Nm,ng,Nr), dtype=np.complex128)
            self.vec_rep_recv_r = np.empty((3*Nm,ng,Nr), dtype=np.complex128)
            # - Adding vector field buffers
            self.vec_add_send_l = np.empty((3*Nm,2*ng,Nr), dtype=np.complex128)
            self.vec_add_send_r = np.empty((3*Nm,2*ng,Nr), dtype=np.complex128)
            self.vec_add_recv_l = np.empty((3*Nm,2*ng,Nr), dtype=np.complex128)
            self.vec_add_recv_r = np.empty((3*Nm,2*ng,Nr), dtype=np.complex128)
            # - Replacing scalar field buffers
            self.scal_rep_send_l = np.empty((Nm,ng,Nr), dtype=np.complex128)
            self.scal_rep_send_r = np.empty((Nm,ng,Nr), dtype=np.complex128)
            self.scal_rep_recv_l = np.empty((Nm,ng,Nr), dtype=np.complex128)
            self.scal_rep_recv_r = np.empty((Nm,ng,Nr), dtype=np.complex128)
            # - Adding scalar field buffers
            self.scal_add_send_l = np.empty((Nm,2*ng,Nr), dtype=np.complex128)
            self.scal_add_send_r = np.empty((Nm,2*ng,Nr), dtype=np.complex128)
            self.scal_add_recv_l = np.empty((Nm,2*ng,Nr), dtype=np.complex128)
            self.scal_add_recv_r = np.empty((Nm,2*ng,Nr), dtype=np.complex128)
        else:
            # Allocate buffers on the CPU and GPU
            # Use cuda.pinned_array so that CPU array is pagelocked.
            # (cannot be swapped out to disk and GPU can access it via DMA)
            pin_ary = cuda.pinned_array
            # - Replacing vector field buffers
            self.vec_rep_send_l = pin_ary((3*Nm,ng,Nr), dtype=np.complex128)
            self.vec_rep_send_r = pin_ary((3*Nm,ng,Nr), dtype=np.complex128)
            self.vec_rep_recv_l = pin_ary((3*Nm,ng,Nr), dtype=np.complex128)
            self.vec_rep_recv_r = pin_ary((3*Nm,ng,Nr), dtype=np.complex128)
            self.d_vec_rep_buffer_l = cuda.to_device( self.vec_rep_send_l )
            self.d_vec_rep_buffer_r = cuda.to_device( self.vec_rep_send_r )
            # - Adding vector field buffers
            self.vec_add_send_l = pin_ary((3*Nm,2*ng,Nr), dtype=np.complex128)
            self.vec_add_send_r = pin_ary((3*Nm,2*ng,Nr), dtype=np.complex128)
            self.vec_add_recv_l = pin_ary((3*Nm,2*ng,Nr), dtype=np.complex128)
            self.vec_add_recv_r = pin_ary((3*Nm,2*ng,Nr), dtype=np.complex128)
            self.d_vec_add_buffer_l = cuda.to_device( self.vec_add_send_l )
            self.d_vec_add_buffer_r = cuda.to_device( self.vec_add_send_r )
            # - Replacing scalar field buffers
            self.scal_rep_send_l = pin_ary((Nm,ng,Nr), dtype=np.complex128)
            self.scal_rep_send_r = pin_ary((Nm,ng,Nr), dtype=np.complex128)
            self.scal_rep_recv_l = pin_ary((Nm,ng,Nr), dtype=np.complex128)
            self.scal_rep_recv_r = pin_ary((Nm,ng,Nr), dtype=np.complex128)
            self.d_scal_rep_buffer_l = cuda.to_device( self.scal_rep_send_l )
            self.d_scal_rep_buffer_r = cuda.to_device( self.scal_rep_send_r )
            # - Adding scalar field buffers
            self.scal_add_send_l = pin_ary((Nm,2*ng,Nr), dtype=np.complex128)
            self.scal_add_send_r = pin_ary((Nm,2*ng,Nr), dtype=np.complex128)
            self.scal_add_recv_l = pin_ary((Nm,2*ng,Nr), dtype=np.complex128)
            self.scal_add_recv_r = pin_ary((Nm,2*ng,Nr), dtype=np.complex128)
            self.d_scal_add_buffer_l = cuda.to_device( self.scal_add_send_l )
            self.d_scal_add_buffer_r = cuda.to_device( self.scal_add_send_r )

    def handle_vec_buffer( self, grid_r, grid_t, grid_z, method, use_cuda,
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
        grid_r, grid_t, grid_z: lists of 2darrays
            (One element per azimuthal mode)
            The 2d arrays represent the fields on the interpolation grid

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
        Nz = grid_r[0].shape[0]

        # When using the GPU
        if use_cuda:
            # Calculate the number of blocks and threads per block
            dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d(
                nz_end - nz_start, self.Nr )

            if before_sending:
                if method == 'replace':
                    # Copy the inner regions of the domain to the buffers
                    for m in range(self.Nm):
                        copy_vec_to_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                            self.d_vec_rep_buffer_l, self.d_vec_rep_buffer_r,
                            grid_r[m], grid_t[m], grid_z[m], m,
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
                    for m in range(self.Nm):
                        copy_vec_to_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                        self.d_vec_add_buffer_l, self.d_vec_add_buffer_r,
                            grid_r[m], grid_t[m], grid_z[m], m,
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
                    for m in range(self.Nm):
                        replace_vec_from_gpu_buffer[dim_grid_2d, dim_block_2d](
                            self.d_vec_rep_buffer_l, self.d_vec_rep_buffer_r,
                            grid_r[m], grid_t[m], grid_z[m], m,
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
                    for m in range(self.Nm):
                        add_vec_from_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                            self.d_vec_add_buffer_l, self.d_vec_add_buffer_r,
                            grid_r[m], grid_t[m], grid_z[m], m,
                            copy_left, copy_right, nz_start, nz_end )
        # Without GPU
        else:
            if before_sending:
                if method == 'replace':
                    # Copy the inner regions of the domain to the buffers
                    if copy_left:
                        for m in range(self.Nm):
                            self.vec_rep_send_l[3*m+0,:,:]=grid_r[m][nz_start:nz_end,:]
                            self.vec_rep_send_l[3*m+1,:,:]=grid_t[m][nz_start:nz_end,:]
                            self.vec_rep_send_l[3*m+2,:,:]=grid_z[m][nz_start:nz_end,:]
                    if copy_right:
                        for m in range(self.Nm):
                            self.vec_rep_send_r[3*m+0,:,:]=grid_r[m][Nz-nz_end:Nz-nz_start,:]
                            self.vec_rep_send_r[3*m+1,:,:]=grid_t[m][Nz-nz_end:Nz-nz_start,:]
                            self.vec_rep_send_r[3*m+2,:,:]=grid_z[m][Nz-nz_end:Nz-nz_start,:]

                if method == 'add':
                    # Copy the inner+guard regions of the domain to the buffers
                    if copy_left:
                        for m in range(self.Nm):
                            self.vec_add_send_l[3*m+0,:,:]=grid_r[m][nz_start:nz_end,:]
                            self.vec_add_send_l[3*m+1,:,:]=grid_t[m][nz_start:nz_end,:]
                            self.vec_add_send_l[3*m+2,:,:]=grid_z[m][nz_start:nz_end,:]
                    if copy_right:
                        for m in range(self.Nm):
                            self.vec_add_send_r[3*m+0,:,:]=grid_r[m][Nz-nz_end:Nz-nz_start,:]
                            self.vec_add_send_r[3*m+1,:,:]=grid_t[m][Nz-nz_end:Nz-nz_start,:]
                            self.vec_add_send_r[3*m+2,:,:]=grid_z[m][Nz-nz_end:Nz-nz_start,:]

            elif after_receiving:
                if method == 'replace':
                    # Replace the guard cells of the domain with the buffers
                    if copy_left:
                        for m in range(self.Nm):
                            grid_r[m][:nz_end-nz_start,:]=self.vec_rep_recv_l[3*m+0,:,:]
                            grid_t[m][:nz_end-nz_start,:]=self.vec_rep_recv_l[3*m+1,:,:]
                            grid_z[m][:nz_end-nz_start,:]=self.vec_rep_recv_l[3*m+2,:,:]
                    if copy_right:
                        for m in range(self.Nm):
                            grid_r[m][-(nz_end-nz_start):,:]=self.vec_rep_recv_r[3*m+0,:,:]
                            grid_t[m][-(nz_end-nz_start):,:]=self.vec_rep_recv_r[3*m+1,:,:]
                            grid_z[m][-(nz_end-nz_start):,:]=self.vec_rep_recv_r[3*m+2,:,:]

                if method == 'add':
                    # Add buffers to the domain
                    if copy_left:
                        for m in range(self.Nm):
                            grid_r[m][:nz_end-nz_start,:]+=self.vec_add_recv_l[3*m+0,:,:]
                            grid_t[m][:nz_end-nz_start,:]+=self.vec_add_recv_l[3*m+1,:,:]
                            grid_z[m][:nz_end-nz_start,:]+=self.vec_add_recv_l[3*m+2,:,:]
                    if copy_right:
                        for m in range(self.Nm):
                            grid_r[m][-(nz_end-nz_start):,:]+=self.vec_add_recv_r[3*m+0,:,:]
                            grid_t[m][-(nz_end-nz_start):,:]+=self.vec_add_recv_r[3*m+1,:,:]
                            grid_z[m][-(nz_end-nz_start):,:]+=self.vec_add_recv_r[3*m+2,:,:]


    def handle_scal_buffer( self, grid, method, use_cuda,
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
        grid: list of 2darrays
            (One element per azimuthal mode)
            The 2d arrays represent the fields on the interpolation grid

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
        Nz = grid[0].shape[0]

        # When using the GPU
        if use_cuda:
            # Calculate the number of blocks and threads per block
            dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d(
                nz_end - nz_start, self.Nr )

            if before_sending:
                if method == 'replace':
                    # Copy the inner regions of the domain to the GPU buffers
                    for m in range(self.Nm):
                        copy_scal_to_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                            self.d_scal_rep_buffer_l, self.d_scal_rep_buffer_r,
                            grid[m], m, copy_left, copy_right, nz_start, nz_end)
                    # Copy the GPU buffers to the sending CPU buffers
                    if copy_left:
                        self.d_scal_rep_buffer_l.copy_to_host(
                            self.scal_rep_send_l )
                    if copy_right:
                        self.d_scal_rep_buffer_r.copy_to_host(
                            self.scal_rep_send_r )

                if method == 'add':
                    # Copy the inner+guard regions of the domain to the buffers
                    for m in range(self.Nm):
                        copy_scal_to_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                            self.d_scal_add_buffer_l, self.d_scal_add_buffer_r,
                            grid[m], m, copy_left, copy_right, nz_start, nz_end)
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
                    for m in range(self.Nm):
                        replace_scal_from_gpu_buffer[dim_grid_2d, dim_block_2d](
                            self.d_scal_rep_buffer_l, self.d_scal_rep_buffer_r,
                            grid[m], m, copy_left, copy_right, nz_start, nz_end)

                if method == 'add':
                    # Copy the CPU receiving buffers to the GPU buffers
                    if copy_left:
                        self.d_scal_add_buffer_l.copy_to_device(
                            self.scal_add_recv_l )
                    if copy_right:
                        self.d_scal_add_buffer_r.copy_to_device(
                            self.scal_add_recv_r )
                    # Add the GPU buffers to the domain
                    for m in range(self.Nm):
                        add_scal_from_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                            self.d_scal_add_buffer_l, self.d_scal_add_buffer_r,
                            grid[m], m, copy_left, copy_right, nz_start, nz_end)
        # Without GPU
        else:
            if before_sending:
                if method == 'replace':
                    # Copy the inner regions of the domain to the buffer
                    if copy_left:
                        for m in range(self.Nm):
                            self.scal_rep_send_l[m,:,:]=grid[m][nz_start:nz_end,:]
                    if copy_right:
                        for m in range(self.Nm):
                            self.scal_rep_send_r[m,:,:]=grid[m][Nz-nz_end:Nz-nz_start,:]

                if method == 'add':
                    # Copy the inner+guard regions of the domain to the buffer
                    if copy_left:
                        for m in range(self.Nm):
                            self.scal_add_send_l[m,:,:]=grid[m][nz_start:nz_end,:]
                    if copy_right:
                        for m in range(self.Nm):
                            self.scal_add_send_r[m,:,:]=grid[m][Nz-nz_end:Nz-nz_start,:]

            elif after_receiving:
                if method == 'replace':
                    # Replace the guard cells of the domain with the buffers
                    if copy_left:
                        for m in range(self.Nm):
                            grid[m][:nz_end-nz_start,:]=self.scal_rep_recv_l[m,:,:]
                    if copy_right:
                        for m in range(self.Nm):
                            grid[m][-(nz_end-nz_start):,:]=self.scal_rep_recv_r[m,:,:]

                if method == 'add':
                    # Add buffers to the domain
                    if copy_left:
                        for m in range(self.Nm):
                            grid[m][:nz_end-nz_start,:]+=self.scal_add_recv_l[m,:,:]
                    if copy_right:
                        for m in range(self.Nm):
                            grid[m][-(nz_end-nz_start):,:]+=self.scal_add_recv_r[m,:,:]
