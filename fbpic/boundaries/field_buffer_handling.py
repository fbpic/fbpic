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
    import cupy
    from fbpic.utils.cuda import cuda, cuda_tpb_bpg_2d
    from .cuda_methods import \
        copy_vec_to_gpu_buffer, \
        replace_vec_from_gpu_buffer, \
        add_vec_from_gpu_buffer, \
        copy_scal_to_gpu_buffer, \
        replace_scal_from_gpu_buffer, \
        add_scal_from_gpu_buffer, \
        copy_pml_to_gpu_buffer, \
        replace_pml_from_gpu_buffer

class BufferHandler(object):
    """
    Class that handles the buffers when exchanging the fields
    between MPI domains.
    """

    def __init__( self, n_guard, Nr, Nm, left_proc, right_proc, use_pml ):
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

        use_pml: bool
           Whether to use PML fields
        """
        # Register parameters
        self.Nr = Nr
        self.Nm = Nm
        self.n_guard = n_guard
        self.left_proc = left_proc
        self.right_proc = right_proc
        # Shortcut
        ng = self.n_guard

        # Get number of field components for E and B
        if use_pml:
            n_fld = 5 # e.g. Er, Et, Ez, Er_pml, Et_pml
        else:
            n_fld = 3 # e.g. Er, Et, Ez

        # Allocate buffer arrays that are send via MPI to exchange
        # the fields between domains (either replacing or adding fields)
        # Buffers are allocated for the left and right side of the domain

        # Allocate buffers on the CPU
        if cuda_installed:
            # Use cuda.pinned_array so that CPU array is pagelocked.
            # (cannot be swapped out to disk and GPU can access it via DMA)
            alloc_cpu = cuda.pinned_array
        else:
            # Use regular numpy arrays
            alloc_cpu = np.empty
        # Allocate buffers of different size, for the different exchange types
        self.send_l = {
            'E:replace': alloc_cpu( (n_fld*Nm,   ng, Nr), dtype=np.complex128),
            'B:replace': alloc_cpu( (n_fld*Nm,   ng, Nr), dtype=np.complex128),
            'J:add'    : alloc_cpu( (    3*Nm, 2*ng, Nr), dtype=np.complex128),
            'rho:add'  : alloc_cpu( (      Nm, 2*ng, Nr), dtype=np.complex128)}
        self.send_r = {
            'E:replace': alloc_cpu( (n_fld*Nm,   ng, Nr), dtype=np.complex128),
            'B:replace': alloc_cpu( (n_fld*Nm,   ng, Nr), dtype=np.complex128),
            'J:add'    : alloc_cpu( (    3*Nm, 2*ng, Nr), dtype=np.complex128),
            'rho:add'  : alloc_cpu( (      Nm, 2*ng, Nr), dtype=np.complex128)}
        self.recv_l = {
            'E:replace': alloc_cpu( (n_fld*Nm,   ng, Nr), dtype=np.complex128),
            'B:replace': alloc_cpu( (n_fld*Nm,   ng, Nr), dtype=np.complex128),
            'J:add'    : alloc_cpu( (    3*Nm, 2*ng, Nr), dtype=np.complex128),
            'rho:add'  : alloc_cpu( (      Nm, 2*ng, Nr), dtype=np.complex128)}
        self.recv_r = {
            'E:replace': alloc_cpu( (n_fld*Nm,   ng, Nr), dtype=np.complex128),
            'B:replace': alloc_cpu( (n_fld*Nm,   ng, Nr), dtype=np.complex128),
            'J:add'    : alloc_cpu( (    3*Nm, 2*ng, Nr), dtype=np.complex128),
            'rho:add'  : alloc_cpu( (      Nm, 2*ng, Nr), dtype=np.complex128)}

        # Allocate buffers on the GPU, for the different exchange types
        if cuda_installed:
            self.d_send_l = { key: cupy.asarray(value) for key, value in \
                                self.send_l.items() }
            self.d_send_r = { key: cupy.asarray(value) for key, value in \
                                self.send_r.items() }
            self.d_recv_l = { key: cupy.asarray(value) for key, value in \
                                self.recv_l.items() }
            self.d_recv_r = { key: cupy.asarray(value) for key, value in \
                                self.recv_r.items() }


    def handle_vec_buffer(self, grid_r, grid_t, grid_z,
                            pml_r, pml_t, method, exchange_type,
                            use_cuda, before_sending=False,
                            after_receiving=False, gpudirect=False ):
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

        pml_r, pml_t: lists of 2darrays, or None
            The 2d arrays that represent the PML components (if present)
            on the interpolation grid

        method: str
            Can either be 'replace' or 'add' depending on the type
            of field exchange that is needed

        exchange_type: str
            Can either be 'E:replace', 'B:replace', 'J:add' or 'rho:add'
            Determines which buffer array is used.

        use_cuda: bool
            Whether the simulation runs on GPUs. If True,
            the buffers are copied to the GPU arrays after the MPI exchange.

        before_sending: bool
            Whether to copy the inner part of the domain to the sending buffer

        after_receiving: bool
            Whether to copy the receiving buffer to the guard cells

        gpudirect: bool
            - if `gpudirect` is True:
              Uses the CUDA GPUDirect feature on clusters
              that have a working CUDA-aware MPI implementation.
            - if `gpudirect` is False: (default)
              Standard MPI communication is performed when using CUDA
              for computation. This involves a manual GPU to CPU memory
              copy before exchanging information between MPI domains.
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
                # Copy the inner regions of the domain to the buffers
                for m in range(self.Nm):
                    if pml_r is None:
                        # Copy only the regular components
                        copy_vec_to_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                            self.d_send_l[exchange_type],
                            self.d_send_r[exchange_type],
                            grid_r[m], grid_t[m], grid_z[m], m,
                            copy_left, copy_right, nz_start, nz_end )
                    else:
                        # Copy regular components + PML components
                        copy_pml_to_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                            self.d_send_l[exchange_type],
                            self.d_send_r[exchange_type],
                            grid_r[m], grid_t[m], grid_z[m],
                            pml_r[m], pml_t[m], m,
                            copy_left, copy_right, nz_start, nz_end )
                # If GPUDirect with CUDA-aware MPI is not used,
                # copy the GPU buffers to the sending CPU buffers
                if not gpudirect:
                    if copy_left:
                        self.d_send_l[exchange_type].get(
                            out=self.send_l[exchange_type] )
                    if copy_right:
                        self.d_send_r[exchange_type].get(
                            out=self.send_r[exchange_type] )

            elif after_receiving:
                # If GPUDirect with CUDA-aware MPI is not used,
                # copy the CPU receiving buffers to the GPU buffers
                if not gpudirect:
                    if copy_left:
                        self.d_recv_l[exchange_type].set(
                            self.recv_l[exchange_type] )
                    if copy_right:
                        self.d_recv_r[exchange_type].set(
                            self.recv_r[exchange_type] )
                if method == 'replace':
                    # Replace the guard cells of the domain with the buffers
                    for m in range(self.Nm):
                        if pml_r is None:
                            # Copy only the regular components
                            replace_vec_from_gpu_buffer \
                                [dim_grid_2d, dim_block_2d](
                                self.d_recv_l[exchange_type],
                                self.d_recv_r[exchange_type],
                                grid_r[m], grid_t[m], grid_z[m], m,
                                copy_left, copy_right, nz_start, nz_end )
                        else:
                            # Copy regular components + PML components
                            replace_pml_from_gpu_buffer \
                                [ dim_grid_2d, dim_block_2d ](
                                self.d_recv_l[exchange_type],
                                self.d_recv_r[exchange_type],
                                grid_r[m], grid_t[m], grid_z[m],
                                pml_r[m], pml_t[m], m,
                                copy_left, copy_right, nz_start, nz_end )
                elif method == 'add':
                    # Add the buffers to the domain
                    for m in range(self.Nm):
                        add_vec_from_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                            self.d_recv_l[exchange_type],
                            self.d_recv_r[exchange_type],
                            grid_r[m], grid_t[m], grid_z[m], m,
                            copy_left, copy_right, nz_start, nz_end )

        # Without GPU
        else:

            if before_sending:

                send_l = self.send_l[exchange_type]
                send_r = self.send_r[exchange_type]
                # Copy the inner regions of the domain to the buffers
                if copy_left:
                    for m in range(self.Nm):
                        if pml_r is None:
                            # Copy only the regular components
                            send_l[3*m+0,:,:]=grid_r[m][nz_start:nz_end,:]
                            send_l[3*m+1,:,:]=grid_t[m][nz_start:nz_end,:]
                            send_l[3*m+2,:,:]=grid_z[m][nz_start:nz_end,:]
                        else:
                            # Copy regular components + PML components
                            send_l[5*m+0,:,:]=grid_r[m][nz_start:nz_end,:]
                            send_l[5*m+1,:,:]=grid_t[m][nz_start:nz_end,:]
                            send_l[5*m+2,:,:]=grid_z[m][nz_start:nz_end,:]
                            send_l[5*m+3,:,:]=pml_r[m][nz_start:nz_end,:]
                            send_l[5*m+4,:,:]=pml_t[m][nz_start:nz_end,:]
                if copy_right:
                    for m in range(self.Nm):
                        if pml_r is None:
                            # Copy only the regular components
                            send_r[3*m+0,:,:]=grid_r[m][Nz-nz_end:Nz-nz_start,:]
                            send_r[3*m+1,:,:]=grid_t[m][Nz-nz_end:Nz-nz_start,:]
                            send_r[3*m+2,:,:]=grid_z[m][Nz-nz_end:Nz-nz_start,:]
                        else:
                            # Copy regular components + PML components
                            send_r[5*m+0,:,:]=grid_r[m][Nz-nz_end:Nz-nz_start,:]
                            send_r[5*m+1,:,:]=grid_t[m][Nz-nz_end:Nz-nz_start,:]
                            send_r[5*m+2,:,:]=grid_z[m][Nz-nz_end:Nz-nz_start,:]
                            send_r[5*m+3,:,:]=pml_r[m][Nz-nz_end:Nz-nz_start,:]
                            send_r[5*m+4,:,:]=pml_t[m][Nz-nz_end:Nz-nz_start,:]

            elif after_receiving:

                recv_l = self.recv_l[exchange_type]
                recv_r = self.recv_r[exchange_type]
                if method == 'replace':
                    # Replace the guard cells of the domain with the buffers
                    if copy_left:
                        if pml_r is None:
                            # Copy only the regular components
                            for m in range(self.Nm):
                                grid_r[m][:nz_end-nz_start,:]=recv_l[3*m+0,:,:]
                                grid_t[m][:nz_end-nz_start,:]=recv_l[3*m+1,:,:]
                                grid_z[m][:nz_end-nz_start,:]=recv_l[3*m+2,:,:]
                        else:
                            # Copy regular components + PML components
                            for m in range(self.Nm):
                                grid_r[m][:nz_end-nz_start,:]=recv_l[5*m+0,:,:]
                                grid_t[m][:nz_end-nz_start,:]=recv_l[5*m+1,:,:]
                                grid_z[m][:nz_end-nz_start,:]=recv_l[5*m+2,:,:]
                                pml_r[m][:nz_end-nz_start,:]=recv_l[5*m+3,:,:]
                                pml_t[m][:nz_end-nz_start,:]=recv_l[5*m+4,:,:]
                    if copy_right:
                        for m in range(self.Nm):
                            if pml_r is None:
                                # Copy only the regular components
                                grid_r[m][-(nz_end-nz_start):,:]=recv_r[3*m+0,:,:]
                                grid_t[m][-(nz_end-nz_start):,:]=recv_r[3*m+1,:,:]
                                grid_z[m][-(nz_end-nz_start):,:]=recv_r[3*m+2,:,:]
                            else:
                                # Copy regular components + PML components
                                grid_r[m][-(nz_end-nz_start):,:]=recv_r[5*m+0,:,:]
                                grid_t[m][-(nz_end-nz_start):,:]=recv_r[5*m+1,:,:]
                                grid_z[m][-(nz_end-nz_start):,:]=recv_r[5*m+2,:,:]
                                pml_r[m][-(nz_end-nz_start):,:]=recv_r[5*m+3,:,:]
                                pml_t[m][-(nz_end-nz_start):,:]=recv_r[5*m+4,:,:]
                elif method == 'add':
                    # Add buffers to the domain
                    if copy_left:
                        for m in range(self.Nm):
                            grid_r[m][:nz_end-nz_start,:]+=recv_l[3*m+0,:,:]
                            grid_t[m][:nz_end-nz_start,:]+=recv_l[3*m+1,:,:]
                            grid_z[m][:nz_end-nz_start,:]+=recv_l[3*m+2,:,:]
                    if copy_right:
                        for m in range(self.Nm):
                            grid_r[m][-(nz_end-nz_start):,:]+=recv_r[3*m+0,:,:]
                            grid_t[m][-(nz_end-nz_start):,:]+=recv_r[3*m+1,:,:]
                            grid_z[m][-(nz_end-nz_start):,:]+=recv_r[3*m+2,:,:]


    def handle_scal_buffer( self, grid, method, exchange_type, use_cuda,
                            before_sending=False, after_receiving=False,
                            gpudirect=False ):
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

        gpudirect: bool
            - if `gpudirect` is True:
              Uses the CUDA GPUDirect feature on clusters
              that have a working CUDA-aware MPI implementation.
            - if `gpudirect` is False: (default)
              Standard MPI communication is performed when using CUDA
              for computation. This involves a manual GPU to CPU memory
              copy before exchanging information between MPI domains.
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
        Nz = grid[0].shape[0]

        # When using the GPU
        if use_cuda:
            # Calculate the number of blocks and threads per block
            dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d(
                nz_end - nz_start, self.Nr )

            if before_sending:
                # Copy the inner regions of the domain to the buffers
                for m in range(self.Nm):
                    copy_scal_to_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                        self.d_send_l[exchange_type],
                        self.d_send_r[exchange_type],
                        grid[m], m, copy_left, copy_right, nz_start, nz_end)
                # If GPUDirect with CUDA-aware MPI is not used,
                # copy the GPU buffers to the sending CPU buffers
                if not gpudirect:
                    if copy_left:
                        self.d_send_l[exchange_type].get(
                            out=self.send_l[exchange_type] )
                    if copy_right:
                        self.d_send_r[exchange_type].get(
                            out=self.send_r[exchange_type] )

            elif after_receiving:
                # If GPUDirect with CUDA-aware MPI is not used,
                # copy the CPU receiving buffers to the GPU buffers
                if not gpudirect:
                    if copy_left:
                        self.d_recv_l[exchange_type].set(
                            self.recv_l[exchange_type] )
                    if copy_right:
                        self.d_recv_r[exchange_type].set(
                            self.recv_r[exchange_type] )
                if method == 'replace':
                    # Replace the guard cells of the domain with the buffers
                    for m in range(self.Nm):
                        replace_scal_from_gpu_buffer[dim_grid_2d, dim_block_2d](
                            self.d_recv_l[exchange_type],
                            self.d_recv_r[exchange_type],
                            grid[m], m, copy_left, copy_right, nz_start, nz_end)
                elif method == 'add':
                    # Add the buffers to the domain
                    for m in range(self.Nm):
                        add_scal_from_gpu_buffer[ dim_grid_2d, dim_block_2d ](
                            self.d_recv_l[exchange_type],
                            self.d_recv_r[exchange_type],
                            grid[m], m, copy_left, copy_right, nz_start, nz_end)

        # Without GPU
        else:

            if before_sending:

                send_l = self.send_l[exchange_type]
                send_r = self.send_r[exchange_type]
                # Copy the inner regions of the domain to the buffer
                if copy_left:
                    for m in range(self.Nm):
                        send_l[m,:,:]=grid[m][nz_start:nz_end,:]
                if copy_right:
                    for m in range(self.Nm):
                        send_r[m,:,:]=grid[m][Nz-nz_end:Nz-nz_start,:]

            elif after_receiving:

                recv_l = self.recv_l[exchange_type]
                recv_r = self.recv_r[exchange_type]
                if method == 'replace':
                    # Replace the guard cells of the domain with the buffers
                    if copy_left:
                        for m in range(self.Nm):
                            grid[m][:nz_end-nz_start,:]=recv_l[m,:,:]
                    if copy_right:
                        for m in range(self.Nm):
                            grid[m][-(nz_end-nz_start):,:]=recv_r[m,:,:]

                if method == 'add':
                    # Add buffers to the domain
                    if copy_left:
                        for m in range(self.Nm):
                            grid[m][:nz_end-nz_start,:]+=recv_l[m,:,:]
                    if copy_right:
                        for m in range(self.Nm):
                            grid[m][-(nz_end-nz_start):,:]+=recv_r[m,:,:]
