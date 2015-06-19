"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to implement the moving window.
"""
import numpy as np
from mpi4py import MPI as mpi
from fields.fields import InterpolationGrid
from particles.particles import Particles
try :
    from numba import cuda
    cuda_installed = True
except ImportError :
    cuda_installed = False

class MPI_Communicator(object) :
    """
    Class that handles the MPI communication between domains,
    when carrying out a simulation in parallel. It also handles
    the initial domain decomposition.

    Main attributes
    ----------
    - Nz, Nr : int
        The global number of cells of the simulation (without guard cells)

    - Nz_local : int
        The local number of cells on the MPI process (with guard cells)
    
    - Nz_domain : int
        The local number of cells without the guard regions

    - Nz_add_last : int 
        The additional number of cells in the last (right) domain, that
        need to be added if the grid cannot be divided by n domains evenly.

    - Ltot : float
        The global size (length) of the simulation box in z
    
    - exchange_part_period : int
        The period for exchanging the particles between domains.
        Needs to be smaller than the number of guard cells, when
        advancing the simulation with a timestep dt = dz/c.

    - n_guard : int
        The number of guard cells that are added on the left and 
        the right side of each local (one MPI process) domain.

    - rank : int
        Identifier of MPI thread

    - size : int 
        Total number of MPI threads

    - mpi_comm : object
        The mpi4py communicator object (defaults to COMM_WORLD)
    """

    def __init__( self, Nz, Nr, n_guard, Nm) :
        """
        Initializes a communicator object.

        Parameters
        ----------

        Nz, Nr : int
            The initial global number of cells

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
        # Initialize number of guard cells
        self.n_guard = n_guard
        # Initialize the period of the particle exchange
        # Particles are only exchanged every exchange_part_period timesteps.
        # (Cannot be higer than the number of guard cells)
        self.exchange_part_period = int(n_guard/2)

        # MPI Setup
        # Initialize the mpi communicator
        self.mpi_comm = mpi.COMM_WORLD
        # Initialize the rank and the total number of mpi processes
        self.rank = self.mpi_comm.rank
        self.size = self.mpi_comm.size

        # Initialize local number of cells
        # (Splits the global simulation and
        # adds guard cells to the local domain)
        if self.rank == (self.size-1):
            # Last domain gets extra cells in case Nz/self.size returns float
            # Last domain = domain at the right edge of the Simulation
            self.Nz_add_last = Nz % self.size
            self.Nz_local = int(Nz/self.size) + self.Nz_add_last + 2*n_guard
        else:
            # Other domains get all the same domain size
            self.Nz_local = int(Nz/self.size) + 2*n_guard
            self.Nz_add_last = 0

        # Check if the local domain size is large enough
        if self.Nz_local < 4*self.n_guard:
            raise ValueError( 'Number of local cells in z is smaller \
                               than 4 times n_guard. Use fewer domains or \
                               a smaller number of guard cells.')
            
        # Initialize the guard cell buffers for the fields 
        # for both sides of the domain. These buffers are used 
        # to exchange the fields between the neighboring domains.
        # For the E and B fields: Only the guard cells are exchanged
        # For J and rho: 2 * the guard cells are exchanged
        # Er, Et, Ez, Br, Bt, Bz for all modes m
        # Send right and left
        self.EB_send_r = np.empty((6*Nm, n_guard, Nr), dtype = np.complex128)
        self.EB_send_l = np.empty((6*Nm, n_guard, Nr), dtype = np.complex128)
        # Receive right and left
        self.EB_recv_r = np.empty((6*Nm, n_guard, Nr), dtype = np.complex128)
        self.EB_recv_l = np.empty((6*Nm, n_guard, Nr), dtype = np.complex128)
        # Jr, Jt, Jz for all modes m
        # Send to right and left
        self.J_send_r = np.empty((3*Nm, 2*n_guard, Nr), dtype = np.complex128)
        self.J_send_l = np.empty((3*Nm, 2*n_guard, Nr), dtype = np.complex128)
        # Receive from right and left
        self.J_recv_r = np.empty((3*Nm, 2*n_guard, Nr), dtype = np.complex128)
        self.J_recv_l = np.empty((3*Nm, 2*n_guard, Nr), dtype = np.complex128)
        # rho for all modes m
        # Send to right and left
        self.rho_send_r = np.empty((Nm, 2*n_guard, Nr), dtype = np.complex128)
        self.rho_send_l = np.empty((Nm, 2*n_guard, Nr), dtype = np.complex128)
        # Receive from right and left
        self.rho_recv_r = np.empty((Nm, 2*n_guard, Nr), dtype = np.complex128)
        self.rho_recv_l = np.empty((Nm, 2*n_guard, Nr), dtype = np.complex128)
        
        # Create damping array which is used to damp
        # the fields in the guard cells to reduce the error
        # generated by cutting off the infinite stencil
        # The n_guard/2 last cells on both sides of the 
        # domain are damped by default.
        self.create_damp_array(ncells_damp = int(n_guard/2))

    def divide_into_domain( self, zmin, zmax, p_zmin, p_zmax ):
        """
        Divide the global simulation into domain and add local guard cells.

        Return the new size of the local domain (zmin, zmax) and the
        boundaries of the initial plasma (p_zmin, p_zmax)

        Parameters :
        ------------
        zmin, zmax : floats
            Positions of the edges of the global simulation box
            (without guard cells)

        p_zmin, p_zmax : floats
            Positions between which the plasma will be initialized, in
            the global simulation box.

        Returns :
        ---------
        A tuple with 
        zmin, zmax : floats
            Positions of the edges of the local simulation box
            (with guard cells)

        p_zmin, p_zmax : floats
           Positions between which the plasma will be initialized, in
           the local simulation box.
           (NB : no plasma will be initialized in the guard cells)
        """
        # Initilize global box size
        self.Ltot = (zmax-zmin)
        # Get the distance dz between the cells
        # (longitudinal spacing of the grid)
        dz = (zmax - zmin)/self.Nz
        # Get the number of local cells without the guard cells
        # in each domain (except the last domain, which has extra cells)
        Nz_domain = int(self.Nz/self.size)
        # Calculate the local boundaries (zmin and zmax)
        # of this local simulation box including the guard cells.
        zmin_local = zmin + ((self.rank)*Nz_domain - self.n_guard)*dz
        zmax_local = zmin_local + self.Nz_local*dz
        # Calculate the new limits (p_zmin and p_zmax)
        # for adding particles to this domain
        p_zmin = max( zmin_local + self.n_guard*dz, p_zmin)
        p_zmax = min( zmax_local - self.n_guard*dz, p_zmax)
        # Initilaize domain specific parameters
        self.dz = dz
        self.Nz_domain = Nz_domain
        # Return the new boundaries to the simulation object
        return( zmin_local, zmax_local, p_zmin, p_zmax )

    def exchange_domains( self, send_left, send_right, recv_left, recv_right ):
        """
        Send the arrays send_left and send_right to the left and right
        processes respectively.
        Receive the arrays from the neighboring processes into recv_left
        and recv_right.

        Parameters :
        ------------
        - send_left, send_right, recv_left, recv_right : arrays 
             Sending and receiving buffers
        """
        # Get the rank of the left and the right domain
        left_domain = self.rank-1
        right_domain = self.rank+1
        # Periodic boundary conditions for the domains
        # (Left side of first (left) domain is added to right
        # side of last (right) domain)
        if left_domain < 0: 
            left_domain = (self.size-1)
        if right_domain > (self.size-1):
            right_domain = 0
        
        # MPI-Exchange: Uses non-blocking send and receive, 
        # which return directly and need to be synchronized later.
        # Send to left domain and receive from right domain
        self.mpi_comm.Isend(send_left, dest=left_domain, tag=1)
        req_1 = self.mpi_comm.Irecv(recv_right, source=right_domain, tag=1)
        # Send to right domain and receive from left domain
        self.mpi_comm.Isend(send_right, dest=right_domain, tag=2)
        req_2 = self.mpi_comm.Irecv(recv_left, source=left_domain, tag=2)
        # Wait for the non-blocking sends to be received (synchronization)
        re_1 = mpi.Request.Wait(req_1)
        re_2 = mpi.Request.Wait(req_2)

    def exchange_fields( self, interp, fieldtype ):
        """
        Send and receive the proper fields, depending on fieldtype
        Copy/add them consistently to the local grid.

        The layout of the local domain and a neighboring domain 
        can be visualised like this:
                      ---------------------
                      |ng|nc|       |nc|ng|    <- Local domain (rank)
                      ---------------------
        ---------------------
        |ng|nc|       |nc|ng|                  <- Neighboring domain (rank-1)
        ---------------------
        The area "ng" defines the region of the guard cells with length n_guard
        The area "nc" defines a region within the domain, that contains the 
        correct simulation data and has also a length of n_guard cells. 
        This region overlaps with the guard cells of the neighboring domain.

        Exchange of E and B fields:

        - Copy the correct part "nc" of the local domain to the guard cells
          of the neighboring domain.
        - [The fields in the guard cells are then damped (separate method)]

        Exchange of the currents J and the charge density rho:

        - Copy the guard cell region "ng" and the correct part "nc" and 
          add it to the same region (ng + nc) of the neighboring domain.

        Parameters :
        ------------
        interp : list
            A list of InterpolationGrid objects
            (one element per azimuthal mode)

        fieldtype : str
            An identifier for the field to send
            (Either 'EB', 'J' or 'rho')
        """
        # Shortcut for number of guard cells
        ng = self.n_guard
        
        # Check for fieldtype (E and B fields)
        if fieldtype == 'EB':
            # Copy the inner region of the domain to the buffer
            for m in range(self.Nm):
                offset = 6*m
                # Copy to buffer for sending to left
                self.EB_send_l[0+offset,:,:] = interp[m].Er[ng:2*ng,:]
                self.EB_send_l[1+offset,:,:] = interp[m].Et[ng:2*ng,:]
                self.EB_send_l[2+offset,:,:] = interp[m].Ez[ng:2*ng,:]
                self.EB_send_l[3+offset,:,:] = interp[m].Br[ng:2*ng,:]
                self.EB_send_l[4+offset,:,:] = interp[m].Bt[ng:2*ng,:]
                self.EB_send_l[5+offset,:,:] = interp[m].Bz[ng:2*ng,:]
                # Copy to buffer for sending to right
                self.EB_send_r[0+offset,:,:] = interp[m].Er[-2*ng:-ng,:]
                self.EB_send_r[1+offset,:,:] = interp[m].Et[-2*ng:-ng,:]
                self.EB_send_r[2+offset,:,:] = interp[m].Ez[-2*ng:-ng,:]
                self.EB_send_r[3+offset,:,:] = interp[m].Br[-2*ng:-ng,:]
                self.EB_send_r[4+offset,:,:] = interp[m].Bt[-2*ng:-ng,:]
                self.EB_send_r[5+offset,:,:] = interp[m].Bz[-2*ng:-ng,:]
            # Exchange the guard regions between the domains (MPI)
            # The inner (undistorted) region of the current rank is sent
            # to the left and the right and at the same time received from
            # the neigboring domains. (for E and B fields)
            self.exchange_domains(self.EB_send_l, self.EB_send_r,
                                  self.EB_recv_l, self.EB_recv_r)
            # An MPI barrier is needed here so that a single rank 
            # does not perform two sends and receives before all 
            # the other MPI connections within this exchange are completed.
            self.mpi_comm.Barrier()

            # Copy the received buffer (from the neighbors) to the 
            # local guard cell region
            for m in range(self.Nm):
                offset = 6*m
                # Copy buffer received from left to guard region
                interp[m].Er[:ng,:] = self.EB_recv_l[0+offset,:,:]
                interp[m].Et[:ng,:] = self.EB_recv_l[1+offset,:,:]
                interp[m].Ez[:ng,:] = self.EB_recv_l[2+offset,:,:]
                interp[m].Br[:ng,:] = self.EB_recv_l[3+offset,:,:]
                interp[m].Bt[:ng,:] = self.EB_recv_l[4+offset,:,:] 
                interp[m].Bz[:ng,:] = self.EB_recv_l[5+offset,:,:]
                # Copy buffer received from right to guard region
                interp[m].Er[-ng:,:] = self.EB_recv_r[0+offset,:,:]
                interp[m].Et[-ng:,:] = self.EB_recv_r[1+offset,:,:]
                interp[m].Ez[-ng:,:] = self.EB_recv_r[2+offset,:,:]
                interp[m].Br[-ng:,:] = self.EB_recv_r[3+offset,:,:]
                interp[m].Bt[-ng:,:] = self.EB_recv_r[4+offset,:,:] 
                interp[m].Bz[-ng:,:] = self.EB_recv_r[5+offset,:,:]
                
        # Check for fieldtype (currents, J)
        elif fieldtype == 'J':
            # Copy the inner and the guard region of the domain to the buffer
            for m in range(self.Nm):
                offset = 3*m
                # Copy to buffer for sending to left
                self.J_send_l[0+offset,:,:] = interp[m].Jr[:2*ng,:]
                self.J_send_l[1+offset,:,:] = interp[m].Jt[:2*ng,:]
                self.J_send_l[2+offset,:,:] = interp[m].Jz[:2*ng,:]
                # Copy to buffer for sending to right
                self.J_send_r[0+offset,:,:] = interp[m].Jr[-2*ng:,:]
                self.J_send_r[1+offset,:,:] = interp[m].Jt[-2*ng:,:]
                self.J_send_r[2+offset,:,:] = interp[m].Jz[-2*ng:,:]
            # Exchange the guard regions and the inner regions of the 
            # current rank between the neighboring domains (MPI).
            # The data is sent to the left and the right  domain and 
            # at the same time received from the neigboring domains. (for J)
            self.exchange_domains(self.J_send_l, self.J_send_r,
                                  self.J_recv_l, self.J_recv_r)
            # An MPI barrier is needed here so that a single rank 
            # does not perform two sends and receives before all 
            # the other MPI connections within this exchange are completed.
            self.mpi_comm.Barrier()

            # Add from buffer
            for m in range(self.Nm):
                offset = 3*m
                # Add the buffer received from the left domain 
                # to the inner region and the guard region
                interp[m].Jr[:2*ng,:] += self.J_recv_l[0+offset,:,:]
                interp[m].Jt[:2*ng,:] += self.J_recv_l[1+offset,:,:] 
                interp[m].Jz[:2*ng,:] += self.J_recv_l[2+offset,:,:] 
                # Add the buffer received from the right
                # to the inner region and the guard region
                interp[m].Jr[-2*ng:,:] += self.J_recv_r[0+offset,:,:]
                interp[m].Jt[-2*ng:,:] += self.J_recv_r[1+offset,:,:] 
                interp[m].Jz[-2*ng:,:] += self.J_recv_r[2+offset,:,:]
                
        # Check for fieldtype (charge density, rho)
        elif fieldtype == 'rho':
            # Copy the inner and the guard region of the domain to the buffer
            for m in range(self.Nm):
                offset = 1*m
                # Copy to buffer for sending to left
                self.rho_send_l[0+offset,:,:] = interp[m].rho[:2*ng,:]
                # Copy to buffer for sending to right
                self.rho_send_r[0+offset,:,:] = interp[m].rho[-2*ng:,:]
            # Exchange the guard regions and the inner regions of the 
            # current rank between the neighboring domains (MPI).
            # The data is sent to the left and the right  domain and 
            # at the same time received from the neigboring domains. (for rho)
            self.exchange_domains(self.rho_send_l, self.rho_send_r,
                                  self.rho_recv_l, self.rho_recv_r)
            # An MPI barrier is needed here so that a single rank 
            # does not perform two sends and receives before all 
            # the other MPI connections within this exchange are completed.
            self.mpi_comm.Barrier()

            # Copy from buffer
            for m in range(self.Nm):
                offset = 1*m
                # Add the buffer received from the left domain 
                # to the inner region and the guard region
                interp[m].rho[:2*ng,:] += self.rho_recv_l[0+offset,:,:]
                # Add the buffer received from the right
                # to the inner region and the guard region
                interp[m].rho[-2*ng:,:] += self.rho_recv_r[0+offset,:,:]
                
        else :
            raise ValueError('Unknown fieldtype : %s' %fieldtype)
            
    def exchange_particles(self, ptcl, zmin, zmax) :
        """
        Look for particles that are located inside the guard cells
        and exchange them with the corresponding neighboring processor.

        Parameters :
        ------------
        ptcl : list
            A list of Particles objects
        """
        # Shortcuts for number of guard cells and spacing between cells
        ng = self.n_guard
        dz = self.dz
        
        # Periodic boundary conditions for exchanging particles
        # Particles leaving at the right (left) side of the simulation box
        # are shifted by Ltot (zmax-zmin) to the left (right).
        if self.rank == 0:
            periodic_offset_left = self.Ltot
            periodic_offset_right = 0.
        elif self.rank == (self.size-1):
            periodic_offset_left = 0.
            periodic_offset_right = -self.Ltot
        else:
            periodic_offset_left = 0.
            periodic_offset_right = 0.

        # Select the particles that are in the left or right guard cells,
        # and those that stay on the local process
        selec_left = ( ptcl.z < (zmin + ng*dz) )
        selec_right = ( ptcl.z > (zmax - ng*dz) )
        selec_stay = (np.logical_not(selec_left) & np.logical_not(selec_right))
        # Count them, and convert the result into an array
        # so as to send them easily.
        N_send_l = np.array( sum(selec_left), dtype=np.int32)
        N_send_r = np.array( sum(selec_right), dtype=np.int32)
        N_stay = sum(selec_stay)

        # Initialize empty arrays to receive the number of particles that
        # will be send to this domain.
        N_recv_l = np.array(0, dtype = np.int32)
        N_recv_r = np.array(0, dtype = np.int32)
        # Send and receive the number of particles that are exchanged 
        self.exchange_domains(N_send_l, N_send_r, N_recv_l, N_recv_r)

        # Allocate sending buffers
        send_left = np.empty((8, N_send_l), dtype = np.float64)
        send_right = np.empty((8, N_send_r), dtype = np.float64)

        # Fill the sending buffers
        # Left guard region
        send_left[0,:] = ptcl.x[selec_left]
        send_left[1,:] = ptcl.y[selec_left]
        send_left[2,:] = ptcl.z[selec_left]+periodic_offset_left
        send_left[3,:] = ptcl.ux[selec_left]
        send_left[4,:] = ptcl.uy[selec_left]
        send_left[5,:] = ptcl.uz[selec_left]
        send_left[6,:] = ptcl.inv_gamma[selec_left]
        send_left[7,:] = ptcl.w[selec_left]
        # Right guard region
        send_right[0,:] = ptcl.x[selec_right]
        send_right[1,:] = ptcl.y[selec_right]
        send_right[2,:] = ptcl.z[selec_right]+periodic_offset_right
        send_right[3,:] = ptcl.ux[selec_right]
        send_right[4,:] = ptcl.uy[selec_right]
        send_right[5,:] = ptcl.uz[selec_right]
        send_right[6,:] = ptcl.inv_gamma[selec_right]
        send_right[7,:] = ptcl.w[selec_right]

        # An MPI barrier is needed here so that a single rank 
        # does not perform two sends and receives before all 
        # the other MPI connections within this exchange are completed.
        # Barrier is not called directly after the exchange
        # to hide the allocation of buffer data
        self.mpi_comm.Barrier()

        # Allocate the receiving buffers and exchange particles
        recv_left = np.zeros((8, N_recv_l), dtype = np.float64)
        recv_right = np.zeros((8, N_recv_r), dtype = np.float64)
        self.exchange_domains(send_left, send_right, recv_left, recv_right)

        # An MPI barrier is needed here so that a single rank 
        # does not perform two sends and receives before all 
        # the other MPI connections within this exchange are completed.
        self.mpi_comm.Barrier()

        # Form the new particle arrays by adding the received particles
        # from the left and the right to the particles that stay in the domain
        ptcl.Ntot = N_stay + int(N_recv_l) + int(N_recv_r)
        ptcl.x = np.hstack((recv_left[0], ptcl.x[selec_stay], recv_right[0]))
        ptcl.y = np.hstack((recv_left[1], ptcl.y[selec_stay], recv_right[1]))
        ptcl.z = np.hstack((recv_left[2], ptcl.z[selec_stay], recv_right[2]))
        ptcl.ux = np.hstack((recv_left[3], ptcl.ux[selec_stay], recv_right[3]))
        ptcl.uy = np.hstack((recv_left[4], ptcl.uy[selec_stay], recv_right[4]))
        ptcl.uz = np.hstack((recv_left[5], ptcl.uz[selec_stay], recv_right[5]))
        ptcl.inv_gamma = \
          np.hstack((recv_left[6], ptcl.inv_gamma[selec_stay], recv_right[6]))
        ptcl.w = np.hstack((recv_left[7], ptcl.w[selec_stay], recv_right[7]))

        # Reallocate the particles field arrays. This needs to be done,
        # as the total number of particles in this domain has changed.
        ptcl.Ex = np.empty(ptcl.Ntot, dtype = np.float64)
        ptcl.Ey = np.empty(ptcl.Ntot, dtype = np.float64)
        ptcl.Ez = np.empty(ptcl.Ntot, dtype = np.float64)
        ptcl.Bx = np.empty(ptcl.Ntot, dtype = np.float64)
        ptcl.By = np.empty(ptcl.Ntot, dtype = np.float64)
        ptcl.Bz = np.empty(ptcl.Ntot, dtype = np.float64)
        # Reallocate the cell index and sorted index arrays
        ptcl.cell_idx = np.empty(ptcl.Ntot, dtype = np.int32)
        ptcl.sorted_idx = np.arange(ptcl.Ntot, dtype = np.uint32)

    def create_damp_array( self, ncells_damp = 0, damp_shape = 'cos'):
        """
        Create the damping array for the fields in the guard cells.
        The ncells_damp first cells are modified by the damping.

        Parameters :
        ------------
        ncells_damp : int
            The number of cells over which the field is damped,
            in the guard cells

        damp_shape : str
            An identifier of the damping function
            Either 'cos', 'None', 'linear' or 'sin'
        """
        if damp_shape == 'None' :
            self.damp_array = np.ones(ncells_damp)
        elif damp_shape == 'linear' :
            self.damp_array = np.linspace(0, 1, ncells_damp)
        elif damp_shape == 'sin' :
            self.damp_array = np.sin(np.linspace(0, np.pi/2, ncells_damp) )
        elif damp_shape == 'cos' :
            self.damp_array = 0.5-0.5*np.cos(
                np.linspace(0, np.pi, ncells_damp) )
        else :
            raise ValueError("Invalid string for damp_shape : %s"%damp_shape)

    def damp_guard_fields( self, interp ):
        """
        Apply the damping shape in the right and left guard cells.
        create_damp_array() needs to be called before this function
        is called, in order to initialize the damping array.

        Parameter :
        -----------
        interp : list
            A list of InterpolationGrid objects (one per azimuthal mode)
        """
        # Number of cells that are damped
        dc = len(self.damp_array) 
        # Damping of the fields in the guard cells.
        # Shape and length defined by self.damp_array
        # (create_damp_array needs to be called before this function)
        for m in range(self.Nm):
            # Damp the fields in left guard cells
            interp[m].Er[:dc,:] *= self.damp_array[:,np.newaxis]
            interp[m].Et[:dc,:] *= self.damp_array[:,np.newaxis]
            interp[m].Ez[:dc,:] *= self.damp_array[:,np.newaxis]
            interp[m].Br[:dc,:] *= self.damp_array[:,np.newaxis]
            interp[m].Bt[:dc,:] *= self.damp_array[:,np.newaxis]
            interp[m].Bz[:dc,:] *= self.damp_array[:,np.newaxis]
            # Damp the fields in right guard cells
            interp[m].Er[-dc:,:] *= self.damp_array[::-1,np.newaxis]
            interp[m].Et[-dc:,:] *= self.damp_array[::-1,np.newaxis]
            interp[m].Ez[-dc:,:] *= self.damp_array[::-1,np.newaxis]
            interp[m].Br[-dc:,:] *= self.damp_array[::-1,np.newaxis]
            interp[m].Bt[-dc:,:] *= self.damp_array[::-1,np.newaxis]
            interp[m].Bz[-dc:,:] *= self.damp_array[::-1,np.newaxis]

    def gather_grid( self, grid, root = 0):
        """
        Gather a grid object by combining the local domains
        without the guard regions to a new global grid object.

        Parameter :
        -----------
        grid : Grid object (InterpolationGrid)
            A grid object that is gathered on the root process

        root : int, optional
            Process that gathers the data

        Returns :
        ---------
        gathered_grid : Grid object (InterpolationGrid)
            A gathered grid that contains the global simulation data
        """
        if self.rank == root:
            # Calculate global edges of the simulation box on root process
            zmin_global = grid.zmin + self.dz * \
                            (self.n_guard - self.rank*self.Nz_domain)
            zmax_global = zmin_global + self.Ltot
            # Create new grid array that contains cell positions in z
            z = np.linspace(zmin_global, zmax_global, self.Nz) + 0.5*self.dz
            # Initialize new InterpolationGrid object that 
            # is used to gather the global grid data
            gathered_grid = InterpolationGrid(z = z, r = grid.r, m = grid.m )
        else:
            # Other processes do not need to initialize new InterpolationGrid
            gathered_grid = None
        # Loop over fields that need to be gathered
        for field in ['Er', 'Et', 'Ez',
                      'Br', 'Bt', 'Bz',
                      'Jr', 'Jt', 'Jz', 'rho']:
            # Get array of field attribute
            array = getattr(grid, field)
            # Gather array on process root
            gathered_array = self.gather_grid_array(array, root)
            if self.rank == root:
                # Write array to field attribute in the gathered grid object
                setattr(gathered_grid, field, gathered_array)
        # Return the gathered grid
        return(gathered_grid)

    def gather_grid_array(self, array, root = 0):
        """
        Gather a grid array on the root process by using the
        mpi4py routine Gatherv, that gathers arbitrary shape arrays
        by combining the first dimension in ascending order.

        Parameter :
        -----------
        array : array (grid array)
            A grid array of the local domain

        root : int, optional
            Process that gathers the data

        Returns :
        ---------
        gathered_array : array (global grid array)
            A gathered array that contains the global simulation data
        """
        if self.rank == root:
            # Root process creates empty numpy array of the shape 
            # (Nz, Nr), that is used to gather the data
            gathered_array = np.zeros((self.Nz, self.Nr), dtype = array.dtype)
        else:
            # Other processes do not need to initialize a new array
            gathered_array = None
        # Shortcut for the guard cells
        ng = self.n_guard
        # Call the mpi4py routine Gartherv and pass the local grid array
        # as sending buffer without the guard region. The receiving buffer
        # on process root (0) is used to gather the data.
        self.mpi_comm.Gatherv(
            sendbuf = array[ng:-ng,:], 
            recvbuf = gathered_array,
            root = root)
        # return the gathered_array only on process root
        if self.rank == root:
            return(gathered_array)

    def gather_ptcl( self, ptcl, root = 0):
        """
        Gather a particle object by receiving the total number of particles
        Ntot (uses parallel sum reduction) in order to gather (mpi4py Gatherv) 
        the local particle arrays to global particle arrays with a length Ntot.

        Parameter :
        -----------
        ptcl : Particle object
            A particle object that is gathered on the root process

        root : int, optional
            Process that gathers the data

        Returns :
        ---------
        gathered_ptcl : Particle object
            A gathered particle object that contains the global simulation data
        """
        if self.rank == root:
            # Initialize new Particle object that 
            # is used to gather the global grid data
            gathered_ptcl = Particles(ptcl.q, ptcl.m, ptcl.n, 0, self.zmin,
                self.zmax, 0, ptcl.rmin, ptcl.rmax, ptcl.dt)
        else:
            # Other processes do not need to initialize new Particle object
            gathered_ptcl = None
        # Create a new array that contains the local number of particles
        Ntot_local = np.array(ptcl.Ntot, dtype = np.int32)
        # Create a new array that gathers the total number of particles
        Ntot = np.array([0], dtype = np.int32)
        # Use the mpi4py routine Reduce to perform a parallel sum reduction
        # in order to gather the total number of particles.
        self.mpi_comm.Reduce(Ntot_local, Ntot, op=SUM, root = root)
        # Loop over particle attributes that need to be gathered
        for particle_attr in ['x', 'y', 'z',
                              'ux', 'uy', 'uz',
                              'inv_gamma', 'w']:
            # Get array of particle attribute
            array = getattr(ptcl, particle_attr)
            # Gather array on process root
            gathered_array = self.gather_ptcl_array(array, Ntot, root)
            if self.rank == root:
                # Write array to particle attribute in the gathered object
                setattr(gathered_ptcl, particle_attr, gathered_array)
        # Return the gathered particle object
        return(gathered_ptcl)

    def gather_ptcl_array(self, array, length, root = 0):
        """
        Gather a particle array on the root process by using the
        mpi4py routine Gatherv, that gathers arbitrary shape arrays
        by combining the first dimension in ascending order.

        Parameter :
        -----------
        array : array (ptcl array)
            A particle array of the local domain

        length : int
            The length of the gathered array (total number of particles)

        root : int, optional
            Process that gathers the data

        Returns :
        ---------
        gathered_array : array (global ptcl array)
            A gathered array that contains the global simulation data
        """
        if self.rank == root:
            # Root process creates empty numpy array of the shape 
            # (length), that is used to gather the data.
            gathered_array = np.empty(length, dtype = array.dtype)
        else:
            # Other processes do not need to initialize a new array
            gathered_array = None
        # Call the mpi4py routine Gartherv and pass the local particle array.
        # The receiving buffer on process root (0) is used to gather the data.
        self.mpi_comm.Gatherv(
            sendbuf = array, 
            recvbuf = gathered_array,
            root = root)
        # return the gathered_array only on process root
        if self.rank == root:
            return(gathered_array)

