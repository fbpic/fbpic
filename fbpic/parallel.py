"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to implement the moving window.
"""
import numpy as np
from mpi4py import MPI as mpi
from fields.fields import InterpolationGrid

try :
    from numba import cuda
    cuda_installed = True
except ImportError :
    cuda_installed = False

class MPI_Communicator(object) :
    """
    Class that handles MPI communication.

    Attributes
    ----------
    -

    Methods
    -------
    - 
    """
    
    def __init__( self, Nz, Nr, zmin, zmax, n_guard, Nm) :
        """
        Initializes a communicator object.

        Parameters
        ----------

        Nz, Nr : int
            The initial global number of cells

        zmin, zmax : float
            The size of the global simulation box

        n_guard : int
            The number of guard cells at the 
            left and right edge of the domain

        Nm : int
            The total number of modes
        """
        # Initialize global number of cells
        self.Nz = Nz
        self.Nr = Nr

        # Initialize number of modes
        self.Nm = Nm

        # Initialize global box size
        self.zmin = zmin
        self.zmax = zmax

        # Initialize number of guadrd cells
        self.n_guard = n_guard

        # Initialize the mpi communicator
        self.mpi_comm = mpi.COMM_WORLD

        # Initialize the rank and the total
        # number of mpi threads
        self.rank = self.mpi_comm.rank
        self.size = self.mpi_comm.size

        # Initialize local number of cells
        if self.rank == (self.size-1):
            # Last domain gets extra cells in case Nz/self.size returns float
            # Last domain = domain at the right edge of the Simulation
            self.Nz_add_last = Nz % self.size
            self.Nz_local = int(Nz/self.size) + self.Nz_add_last + 2*n_guard
        else:
            # Other domains get all the same domain size
            self.Nz_local = int(Nz/self.size) + 2*n_guard
            self.Nz_add_last = 0

        # Initialize the guard cell buffers for the fields 
        # for both sides of the domain

        # Er, Et, Ez, Br, Bt, Bz for all modes m
        # Send right and left
        self.EB_send_r = np.empty((6*Nm, n_guard, Nr), 
                            dtype = np.complex128)
        self.EB_send_l = np.empty((6*Nm, n_guard, Nr), 
                            dtype = np.complex128)
        # Receive right and left
        self.EB_recv_r = np.empty((6*Nm, n_guard, Nr), 
                            dtype = np.complex128)
        self.EB_recv_l = np.empty((6*Nm, n_guard, Nr), 
                            dtype = np.complex128)

        # Jr, Jt, Jz for all modes m
        # Send right and left
        self.J_send_r = np.empty((3*Nm, n_guard, Nr), 
                            dtype = np.complex128)
        self.J_send_l = np.empty((3*Nm, n_guard, Nr), 
                            dtype = np.complex128)
        # Receive right and left
        self.J_recv_r = np.empty((3*Nm, n_guard, Nr), 
                            dtype = np.complex128)
        self.J_recv_l = np.empty((3*Nm, n_guard, Nr), 
                            dtype = np.complex128)

        # rho for all modes m
        # Send right and left
        self.rho_send_r = np.empty((Nm, n_guard, Nr), 
                            dtype = np.complex128)
        self.rho_send_l = np.empty((Nm, n_guard, Nr), dtype = np.complex128)
        # Receive right and left
        self.rho_recv_r = np.empty((Nm, n_guard, Nr), dtype = np.complex128)
        self.rho_recv_l = np.empty((Nm, n_guard, Nr), dtype = np.complex128)

    def divide_into_domain( self, zmin, zmax, p_zmin, p_zmax ):
        """
        Divide the global simulation into domain.
        Modifies the length of the box in z (zmin, zmax)
        and the boundaries of the initial plasma (p_zmin, p_zmax).
        """  

        dz = (zmax - zmin)/self.Nz
        Nz_delta = int(self.Nz/self.size)

        zmin += ((self.rank)*Nz_delta - self.n_guard)*dz
        zmax = zmin + (Nz_delta + self.Nz_add_last + 2*self.n_guard)*dz

        p_zmin = max(zmin+self.n_guard*dz-0.5*dz, p_zmin)
        p_zmax = min(zmax-self.n_guard*dz-0.5*dz, p_zmax)

        return zmin, zmax, p_zmin, p_zmax

    def exchange_domains( self, send_left, send_right, recv_left, recv_right ) :

        # Get the rank of the left and the right domain
        left_domain = self.rank-1
        right_domain = self.rank+1

        # Periodic boundary conditions for the domains
        if left_domain < 0: 
            left_domain = (self.size-1)
        if right_domain > (self.size-1):
            right_domain = 0

        # Send to left domain and receive from right domain
        if self.rank % 2 == 0:
            self.mpi_comm.Send(send_left, 
                        dest=left_domain, tag=1)
            self.mpi_comm.Recv(recv_right,
                        source=right_domain, tag=2)
        else:
            self.mpi_comm.Recv(recv_right,
                        source=right_domain, tag=1)
            self.mpi_comm.Send(send_left, 
                        dest=left_domain, tag=2)

        self.barrier()

        # Send to left domain and receive from right domain
        if self.rank % 2 == 0:
            self.mpi_comm.Send(send_right,
                        dest=right_domain, tag=3)

            self.mpi_comm.Recv(recv_left,
                        source=left_domain, tag=4)
        else:
            self.mpi_comm.Recv(recv_left,
                        source=left_domain, tag=3)

            self.mpi_comm.Send(send_right,
                        dest=right_domain, tag=4)

        self.barrier()


        # Wait for the non-blocking sends to be received
        #re_1 = mpi.Request.Wait(req_1)
        #re_2 = mpi.Request.Wait(req_2)

        """
        # Send to left domain and receive from right domain
        self.mpi_comm.Isend(send_left, 
                    dest=left_domain, tag=1)
        req_1 = self.mpi_comm.Irecv(recv_right,
                    source=right_domain, tag=1)
        # Send to right domain and receive from left domain
        self.mpi_comm.Isend(send_right,
                    dest=right_domain, tag=2)
        req_2 = self.mpi_comm.Irecv(recv_left,
                    source=left_domain, tag=2)
        # Wait for the non-blocking sends to be received
        re_1 = mpi.Request.Wait(req_1)
        re_2 = mpi.Request.Wait(req_2)
        """

    def exchange_fields( self, interp, fieldtype ):
        ng = self.n_guard
        # Check for fieldtype
        if fieldtype == 'EB':
            # Copy to buffer
            for m in range(self.Nm):
                offset = 6*m
                # Buffer for sending to left
                self.EB_send_l[0+offset,:,:] = interp[m].Er[ng:2*ng,:]
                self.EB_send_l[1+offset,:,:] = interp[m].Et[ng:2*ng,:]
                self.EB_send_l[2+offset,:,:] = interp[m].Ez[ng:2*ng,:]
                self.EB_send_l[3+offset,:,:] = interp[m].Br[ng:2*ng,:]
                self.EB_send_l[4+offset,:,:] = interp[m].Bt[ng:2*ng,:]
                self.EB_send_l[5+offset,:,:] = interp[m].Bz[ng:2*ng,:]
                # Buffer for sending to right
                self.EB_send_r[0+offset,:,:] = interp[m].Er[-2*ng:-ng,:]
                self.EB_send_r[1+offset,:,:] = interp[m].Et[-2*ng:-ng,:]
                self.EB_send_r[2+offset,:,:] = interp[m].Ez[-2*ng:-ng,:]
                self.EB_send_r[3+offset,:,:] = interp[m].Br[-2*ng:-ng,:]
                self.EB_send_r[4+offset,:,:] = interp[m].Bt[-2*ng:-ng,:]
                self.EB_send_r[5+offset,:,:] = interp[m].Bz[-2*ng:-ng,:]
            # Exchange the guard regions between the domains (MPI)
            self.exchange_domains(self.EB_send_l, self.EB_send_r,
                                 self.EB_recv_l, self.EB_recv_r)
            # Copy from buffer
            for m in range(self.Nm):
                offset = 6*m
                # Buffer for receiving from left
                interp[m].Er[:ng,:] = self.EB_recv_l[0+offset,:,:]
                interp[m].Et[:ng,:] = self.EB_recv_l[1+offset,:,:]
                interp[m].Ez[:ng,:] = self.EB_recv_l[2+offset,:,:] 
                interp[m].Br[:ng,:] = self.EB_recv_l[3+offset,:,:] 
                interp[m].Bt[:ng,:] = self.EB_recv_l[4+offset,:,:] 
                interp[m].Bz[:ng,:] = self.EB_recv_l[5+offset,:,:]
                # Buffer for receiving from right
                interp[m].Er[-ng:,:] = self.EB_recv_r[0+offset,:,:]
                interp[m].Et[-ng:,:] = self.EB_recv_r[1+offset,:,:]
                interp[m].Ez[-ng:,:] = self.EB_recv_r[2+offset,:,:]
                interp[m].Br[-ng:,:] = self.EB_recv_r[3+offset,:,:] 
                interp[m].Bt[-ng:,:] = self.EB_recv_r[4+offset,:,:] 
                interp[m].Bz[-ng:,:] = self.EB_recv_r[5+offset,:,:]

        if fieldtype == 'J':
            # Copy to buffer
            for m in range(self.Nm):
                offset = 3*m
                # Buffer for sending to left
                self.J_send_l[0+offset,:,:] = interp[m].Jr[ng:2*ng,:]
                self.J_send_l[1+offset,:,:] = interp[m].Jt[ng:2*ng,:]
                self.J_send_l[2+offset,:,:] = interp[m].Jz[ng:2*ng,:]
                # Buffer for sending to right
                self.J_send_r[0+offset,:,:] = interp[m].Jr[-2*ng:-ng,:]
                self.J_send_r[1+offset,:,:] = interp[m].Jt[-2*ng:-ng,:]
                self.J_send_r[2+offset,:,:] = interp[m].Jz[-2*ng:-ng,:]
            # Exchange the guard regions between the domains (MPI)
            self.exchange_domains(self.J_send_l, self.J_send_r,
                                 self.J_recv_l, self.J_recv_r)
            # Copy from buffer
            for m in range(self.Nm):
                offset = 3*m
                # Buffer for receiving from left
                interp[m].Jr[:ng,:] += self.J_recv_l[0+offset,:,:]
                interp[m].Jt[:ng,:] += self.J_recv_l[1+offset,:,:] 
                interp[m].Jz[:ng,:] += self.J_recv_l[2+offset,:,:] 
                # Buffer for receiving from right
                interp[m].Jr[-ng:,:] += self.J_recv_r[0+offset,:,:]
                interp[m].Jt[-ng:,:] += self.J_recv_r[1+offset,:,:] 
                interp[m].Jz[-ng:,:] += self.J_recv_r[2+offset,:,:]

        if fieldtype == 'rho':
            # Copy to buffer
            for m in range(self.Nm):
                offset = 1*m
                # Buffer for sending to left
                self.rho_send_l[0+offset,:,:] = interp[m].rho[ng:2*ng,:]
                # Buffer for sending to right
                self.rho_send_r[0+offset,:,:] = interp[m].rho[-2*ng:-ng,:]
            # Exchange the guard regions between the domains (MPI)
            self.exchange_domains(self.rho_send_l, self.rho_send_r,
                                 self.rho_recv_l, self.rho_recv_r)
            # Copy from buffer
            for m in range(self.Nm):
                offset = 1*m
                # Buffer for receiving from left
                interp[m].rho[:ng,:] += self.rho_recv_l[0+offset,:,:]
                # Buffer for receiving from right
                interp[m].rho[-ng:,:] += self.rho_recv_r[0+offset,:,:]

    def gather_grid( self, grid, root = 0):

        if self.rank == root:
            z = np.linspace(self.zmin+0.5, self.zmax+0.5, self.Nz)
            gathered_grid = InterpolationGrid(z = z, r = grid.r, m = grid.m )
        else:
            gathered_grid = None

        for field in ['Er', 'Et', 'Ez', 'Br', 'Bt', 'Bz']:
            array = getattr(grid, field)
            gathered_array = self.gather_grid_array_2D(array, root)
            if self.rank == root:
                setattr(gathered_grid, field, gathered_array)

        return gathered_grid

    def gather_grid_array_2D(self, array, root = 0):

        if self.rank == 0:
            gathered_array = np.zeros((self.Nz, self.Nr), dtype = array.dtype)
        else:
            gathered_array = None

        ng = self.n_guard

        Nz_d = int(self.Nz/self.size)
        Nz_last = self.Nz % self.size

        domain_sizes = ()

        for domain in range(self.size-1):
            domain_sizes += (Nz_d*self.Nr, )

        domain_sizes += ((Nz_d + Nz_last)*self.Nr, )

        self.mpi_comm.Gatherv(
            sendbuf = array[ng:-ng,:], 
            recvbuf = [gathered_array,    
                       domain_sizes, 
                       None],
            root = root)

        if self.rank == root:
            return gathered_array
        else:
            return


    def mpi_finalize( self ) :
        mpi.Finalize()

    def barrier( self ) :
        self.mpi_comm.Barrier()







