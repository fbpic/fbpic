"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to handle mpi buffers, in parallel.py
"""
try :
    from fbpic.cuda_utils import cuda_tpb_bpg_2d
    from .cuda_methods import \
        copy_EB_to_gpu_buffers, copy_EB_from_gpu_buffers, \
        copy_J_to_gpu_buffers, add_J_from_gpu_buffers, \
        copy_rho_to_gpu_buffers, add_rho_from_gpu_buffers
    cuda_installed = True
except ImportError :
    cuda_installed = False

def copy_EB_buffers( comm, interp, before_sending=False,
                     after_receiving=False):
    """
    Either copy the inner part of the domain to the sending buffer for E & B,
    or copy the receving buffer for E & B to the guard cells of the domain.

    Depending on whether the field data is initially on the CPU
    or on the GPU, this function will do the appropriate exchange
    with the device.

    Parameters
    ----------
    comm: an MPI_Communicator object
        Contains the sending/receiving buffers and number of guard cells

    interp: a list of InterpolationGrid objects
        (one element per azimuthal mode)

    before_sending: bool
        Whether to copy the inner part of the domain to the sending buffer

    after_receiving: bool
        Whether to copy the receiving buffer to the guard cells of the domain
    """
    # Shortcut for the guard cells
    ng = comm.n_guard
    copy_left = (comm.left_proc is not None)
    copy_right = (comm.right_proc is not None)
        
    # When using the GPU
    if interp[0].use_cuda:

        # Calculate the number of blocks and threads per block
        dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d( ng, interp[0].Nr )
        
        if before_sending:
            # Copy the inner regions of the domain to the GPU buffers
            copy_EB_to_gpu_buffers[ dim_grid_2d, dim_block_2d ](
                comm.d_EB_l, comm.d_EB_r,
                interp[0].Er, interp[0].Et, interp[0].Ez,
                interp[0].Br, interp[0].Bt, interp[0].Bz,
                interp[1].Er, interp[1].Et, interp[1].Ez,
                interp[1].Br, interp[1].Bt, interp[1].Bz,
                copy_left, copy_right, ng )
            # Copy the GPU buffers to the sending CPU buffers
            if copy_left:
                comm.d_EB_l.copy_to_host( comm.EB_send_l )
            if copy_right:
                comm.d_EB_r.copy_to_host( comm.EB_send_r )

        elif after_receiving:
            # Copy the CPU receiving buffers to the GPU buffers
            if copy_left:
                comm.d_EB_l.copy_to_device( comm.EB_recv_l )
            if copy_right:
                comm.d_EB_r.copy_to_device( comm.EB_recv_r )
            # Copy the GPU buffers to the guard cells of the domain
            copy_EB_from_gpu_buffers[ dim_grid_2d, dim_block_2d ](
                comm.d_EB_l, comm.d_EB_r,
                interp[0].Er, interp[0].Et, interp[0].Ez,
                interp[0].Br, interp[0].Bt, interp[0].Bz,
                interp[1].Er, interp[1].Et, interp[1].Ez,
                interp[1].Br, interp[1].Bt, interp[1].Bz,
                copy_left, copy_right, ng )

    # Without GPU
    else:
        for m in range(comm.Nm):
            offset = 6*m

            if before_sending:
                # Copy the inner regions of the domain to the buffer
                if copy_left:
                    comm.EB_send_l[0+offset,:,:] = interp[m].Er[ng:2*ng,:]
                    comm.EB_send_l[1+offset,:,:] = interp[m].Et[ng:2*ng,:]
                    comm.EB_send_l[2+offset,:,:] = interp[m].Ez[ng:2*ng,:]
                    comm.EB_send_l[3+offset,:,:] = interp[m].Br[ng:2*ng,:]
                    comm.EB_send_l[4+offset,:,:] = interp[m].Bt[ng:2*ng,:]
                    comm.EB_send_l[5+offset,:,:] = interp[m].Bz[ng:2*ng,:]
                if copy_right:
                    comm.EB_send_r[0+offset,:,:] = interp[m].Er[-2*ng:-ng,:]
                    comm.EB_send_r[1+offset,:,:] = interp[m].Et[-2*ng:-ng,:]
                    comm.EB_send_r[2+offset,:,:] = interp[m].Ez[-2*ng:-ng,:]
                    comm.EB_send_r[3+offset,:,:] = interp[m].Br[-2*ng:-ng,:]
                    comm.EB_send_r[4+offset,:,:] = interp[m].Bt[-2*ng:-ng,:]
                    comm.EB_send_r[5+offset,:,:] = interp[m].Bz[-2*ng:-ng,:]
                    
            elif after_receiving:
                # Copy the buffer to the guard cells of the domain
                if copy_left:
                    interp[m].Er[:ng,:] = comm.EB_recv_l[0+offset,:,:]
                    interp[m].Et[:ng,:] = comm.EB_recv_l[1+offset,:,:]
                    interp[m].Ez[:ng,:] = comm.EB_recv_l[2+offset,:,:]
                    interp[m].Br[:ng,:] = comm.EB_recv_l[3+offset,:,:]
                    interp[m].Bt[:ng,:] = comm.EB_recv_l[4+offset,:,:] 
                    interp[m].Bz[:ng,:] = comm.EB_recv_l[5+offset,:,:]
                if copy_right:
                    interp[m].Er[-ng:,:] = comm.EB_recv_r[0+offset,:,:]
                    interp[m].Et[-ng:,:] = comm.EB_recv_r[1+offset,:,:]
                    interp[m].Ez[-ng:,:] = comm.EB_recv_r[2+offset,:,:]
                    interp[m].Br[-ng:,:] = comm.EB_recv_r[3+offset,:,:]
                    interp[m].Bt[-ng:,:] = comm.EB_recv_r[4+offset,:,:] 
                    interp[m].Bz[-ng:,:] = comm.EB_recv_r[5+offset,:,:]

def copy_J_buffers( comm, interp, before_sending=False,
                    after_receiving=False):
    """
    Either copy the inner part of the domain to the sending buffer for J,
    or add the receving buffer for J to the guard cells of the domain.

    Depending on whether the field data is initially on the CPU
    or on the GPU, this function will do the appropriate exchange
    with the device.

    Parameters
    ----------
    comm: an MPI_Communicator object
        Contains the sending/receiving buffers and number of guard cells

    interp: a list of InterpolationGrid objects
        (one element per azimuthal mode)

    before_sending: bool
        Whether to copy the inner part of the domain to the sending buffer

    after_receiving: bool
        Whether to add the receiving buffer to the guard cells of the domain
    """
    # Shortcut for the guard cells
    ng = comm.n_guard
    copy_left = (comm.left_proc is not None)
    copy_right = (comm.right_proc is not None)
        
    # When using the GPU
    if interp[0].use_cuda:

        # Calculate the number of blocks and threads per block
        dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d( 2*ng, interp[0].Nr )

        if before_sending:
            # Copy the inner regions of the domain to the GPU buffers
            copy_J_to_gpu_buffers[ dim_grid_2d, dim_block_2d ](
                comm.d_J_l, comm.d_J_r,
                interp[0].Jr, interp[0].Jt, interp[0].Jz,
                interp[1].Jr, interp[1].Jt, interp[1].Jz,
                copy_left, copy_right, ng )
            # Copy the GPU buffers to the CPU sending buffers
            if copy_left:
                comm.d_J_l.copy_to_host( comm.J_send_l )
            if copy_right:
                comm.d_J_r.copy_to_host( comm.J_send_r )

        elif after_receiving:
            # Copy the CPU receiving buffers to the GPU buffers
            if copy_left:
                comm.d_J_l.copy_to_device( comm.J_recv_l )
            if copy_right:
                comm.d_J_r.copy_to_device( comm.J_recv_r )
            # Add the GPU buffers to the guard cells of the domain
            add_J_from_gpu_buffers[ dim_grid_2d, dim_block_2d ](
                comm.d_J_l, comm.d_J_r,
                interp[0].Jr, interp[0].Jt, interp[0].Jz,
                interp[1].Jr, interp[1].Jt, interp[1].Jz,
                copy_left, copy_right, ng )

    # Without GPU
    else:
        for m in range(comm.Nm):
            offset = 3*m

            if before_sending:
                # Copy the inner region of the domain to the buffer
                if copy_left:
                    comm.J_send_l[0+offset,:,:] = interp[m].Jr[:2*ng,:]
                    comm.J_send_l[1+offset,:,:] = interp[m].Jt[:2*ng,:]
                    comm.J_send_l[2+offset,:,:] = interp[m].Jz[:2*ng,:]
                if copy_right:
                    comm.J_send_r[0+offset,:,:] = interp[m].Jr[-2*ng:,:]
                    comm.J_send_r[1+offset,:,:] = interp[m].Jt[-2*ng:,:]
                    comm.J_send_r[2+offset,:,:] = interp[m].Jz[-2*ng:,:]

            elif after_receiving:
                # Add the buffer to the guard cells of the domain
                if copy_left:
                    interp[m].Jr[:2*ng,:] += comm.J_recv_l[0+offset,:,:]
                    interp[m].Jt[:2*ng,:] += comm.J_recv_l[1+offset,:,:]
                    interp[m].Jz[:2*ng,:] += comm.J_recv_l[2+offset,:,:]
                if copy_right:
                    interp[m].Jr[-2*ng:,:] += comm.J_recv_r[0+offset,:,:]
                    interp[m].Jt[-2*ng:,:] += comm.J_recv_r[1+offset,:,:]
                    interp[m].Jz[-2*ng:,:] += comm.J_recv_r[2+offset,:,:]

def copy_rho_buffers( comm, interp, before_sending=False,
                      after_receiving=False):
    """
    Either copy the inner part of the domain to the sending buffer for rho,
    or add the receving buffer for rho to the guard cells of the domain.

    Depending on whether the field data is initially on the CPU
    or on the GPU, this function will do the appropriate exchange
    with the device.

    Parameters
    ----------
    comm: an MPI_Communicator object
        Contains the sending/receiving buffers and number of guard cells

    interp: a list of InterpolationGrid objects
        (one element per azimuthal mode)

    before_sending: bool
        Whether to copy the inner part of the domain to the sending buffer

    after_receiving: bool
        Whether to add the receiving buffer to the guard cells of the domain
    """
    # Shortcut for the guard cells
    ng = comm.n_guard
    copy_left = (comm.left_proc is not None)
    copy_right = (comm.right_proc is not None)
        
    # When using the GPU
    if interp[0].use_cuda:

        # Calculate the number of blocks and threads per block
        dim_grid_2d, dim_block_2d = cuda_tpb_bpg_2d( 2*ng, interp[0].Nr )

        if before_sending:
            # Copy the inner regions of the domain to the GPU buffers
            copy_rho_to_gpu_buffers[ dim_grid_2d, dim_block_2d ](
                comm.d_rho_l, comm.d_rho_r,
                interp[0].rho, interp[1].rho, copy_left, copy_right, ng )
            # Copy the GPU buffers to the sending CPU buffers
            if copy_left:
                comm.d_rho_l.copy_to_host( comm.rho_send_l )
            if copy_right:
                comm.d_rho_r.copy_to_host( comm.rho_send_r )

        elif after_receiving:
            # Copy the receiving CPU buffers to the GPU buffers
            if copy_left:
                comm.d_rho_l.copy_to_device( comm.rho_recv_l )
            if copy_right:
                comm.d_rho_r.copy_to_device( comm.rho_recv_r )
            # Add the GPU buffers to the guard cells of the domain
            add_rho_from_gpu_buffers[ dim_grid_2d, dim_block_2d ](
                comm.d_rho_l, comm.d_rho_r,
                interp[0].rho, interp[1].rho, copy_left, copy_right, ng )

    # Without GPU
    else:
        for m in range(comm.Nm):
            offset = 1*m

            if before_sending:
                # Copy the inner regions of the domain to the buffer
                if copy_left:
                    comm.rho_send_l[0+offset,:,:] = interp[m].rho[:2*ng,:]
                if copy_right:
                    comm.rho_send_r[0+offset,:,:] = interp[m].rho[-2*ng:,:]

            elif after_receiving:
                # Add the buffer to the guard cells of the domain
                if copy_left:
                    interp[m].rho[:2*ng,:] += comm.rho_recv_l[0+offset,:,:]
                if copy_right:
                    interp[m].rho[-2*ng:,:] += comm.rho_recv_r[0+offset,:,:]
