"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to implement the mpi exchanges.
"""
import numpy as np
from mpi4py import MPI as mpi
from fbpic.fields.fields import InterpolationGrid
from fbpic.particles.particles import Particles
from .buffer_handling import *

# Dictionary of correspondance between numpy types and mpi types
# (Necessary when calling Gatherv)
mpi_type_dict = { 'float32' : mpi.REAL4,
                  'float64' : mpi.REAL8,
                  'complex64' : mpi.COMPLEX8,
                  'complex128' : mpi.COMPLEX16 }
    
    
class MPI_Communicator(object) :
    """
    Class that handles the MPI communication between domains,
    when carrying out a simulation in parallel. It also handles
    the initial domain decomposition.

    Main attributes
    ---------------
    - Nz, Nr : int
        The global number of cells of the simulation (without guard cells)

    - Nz_enlarged, Nz_domain : int
        The local number of cells, with and without guard cells, respectively

    - Nz_enlarged_procs, Nz_domain_procs : list of ints
        The number of cells in each proc, with and without guard cells
        
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

    def __init__( self, Nz, Nr, n_guard, Nm, boundaries='periodic') :
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

        boundaries : str
            Indicates how to exchange the fields at the left and right
            boundaries of the global simulation box
            Either 'periodic' or 'open'
        """
        # Initialize global number of cells
        self.Nz = Nz
        self.Nr = Nr
        # Initialize number of modes
        self.Nm = Nm
        # Initialize number of guard cells
        self.n_guard = n_guard
        # Initialize the exchange boundaries
        self.boundaries = boundaries
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

        # Initialize the number of cells of each proc
        # (Splits the global simulation and
        # adds guard cells to the local domain)
        Nz_per_proc = int(Nz/self.size)
        # Get the number of cells in each domain
        # (The last proc gets extra cells, so as to have Nz cells in total)
        Nz_domain_procs = [ Nz_per_proc for k in range(self.size) ]
        Nz_domain_procs[-1] = Nz_domain_procs[-1] + Nz%(self.size)
        self.Nz_domain_procs = Nz_domain_procs
        # Get the starting index (for easy output)
        self.iz_start_procs = [ k*Nz_per_proc for k in range(self.size) ]
        # Get the enlarged number of cells in each domain (includes guards)
        self.Nz_enlarged_procs = [ n+2*n_guard for n in Nz_domain_procs ]

        # Get the local values of the above arrays
        self.Nz_domain = self.Nz_domain_procs[self.rank]
        self.Nz_enlarged = self.Nz_enlarged_procs[self.rank]

        # Check if the local domain size is large enough
        if self.Nz_enlarged < 4*self.n_guard:
            raise ValueError( 'Number of local cells in z is smaller \
                               than 4 times n_guard. Use fewer domains or \
                               a smaller number of guard cells.')
            
        # Initialize the guard cell buffers for the fields 
        # for both sides of the domain. These buffers are used 
        # to exchange the fields between the neighboring domains.
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

        # Create damping array which is used to damp
        # the fields in the guard cells to reduce the error
        # generated by cutting off the infinite stencil
        # The n_guard/2 last cells on both sides of the 
        # domain are damped by default.
        self.create_damp_array( ncells_damp = int(n_guard/2) )

        # Get the rank of the left and the right domain
        self.left_proc = self.rank-1
        self.right_proc = self.rank+1
        # Correct these initial values by taking into account boundaries
        if self.boundaries == 'periodic' :
            # Periodic boundary conditions for the domains
            if self.rank == 0 : 
                self.left_proc = (self.size-1)
            if self.rank == self.size-1 :
                self.right_proc = 0
        elif self.boundaries == 'open' :
            # Moving window
            if self.rank == 0 :
                self.left_proc = None
            if self.rank == self.size-1 :
                self.right_proc = None
        else :
            raise ValueError('Unrecognized boundaries: %s' %self.boundaries)
        
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

        Nz_enlarged : int
           The number of cells in the local simulation box (with guard cells)
        """
        # Initilize global box size
        self.Ltot = (zmax-zmin)
        
        # Get the distance dz between the cells
        # (longitudinal spacing of the grid)
        dz = (zmax - zmin)/self.Nz
        self.dz = dz
        
        # Calculate the local boundaries (zmin and zmax)
        # of this local simulation box including the guard cells.
        iz_start = self.iz_start_procs[self.rank]
        zmin_local = zmin + (iz_start - self.n_guard)*dz
        zmax_local = zmin_local + self.Nz_enlarged*dz
        
        # Calculate the new limits (p_zmin and p_zmax)
        # for adding particles to this domain
        p_zmin = max( zmin_local + self.n_guard*dz, p_zmin)
        p_zmax = min( zmax_local - self.n_guard*dz, p_zmax)
        # Return the new boundaries to the simulation object
        return( zmin_local, zmax_local, p_zmin, p_zmax, self.Nz_enlarged )

    def exchange_fields( self, interp, fieldtype ):
        """
        Send and receive the proper fields, depending on fieldtype
        Copy/add them consistently to the local grid.

        Depending on whether the field data is initially on the CPU
        or on the GPU, this function will do the appropriate exchange
        with the device.

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
        # Check for fieldtype

        if fieldtype == 'EB':

            # Copy the inner part of the domain to the sending buffer
            copy_EB_buffers( self, interp, before_sending=True )
            # Exchange the guard regions between the domains (MPI)
            self.exchange_domains( self.EB_send_l, self.EB_send_r,
                                  self.EB_recv_l, self.EB_recv_r)
            # An MPI barrier is needed here so that a single rank does not
            # do two sends and receives before this exchange is completed.
            self.mpi_comm.Barrier()
            # Copy the receiving buffer to the guard cells of the domain
            copy_EB_buffers( self, interp, after_receiving=True )

        elif fieldtype == 'J':

            # Copy the inner part of the domain to the sending buffer
            copy_J_buffers( self, interp, before_sending=True )
            # Exchange the guard regions between the domains (MPI)
            self.exchange_domains( self.J_send_l, self.J_send_r,
                                  self.J_recv_l, self.J_recv_r)
            # An MPI barrier is needed here so that a single rank does not
            # do two sends and receives before this exchange is completed.
            self.mpi_comm.Barrier()
            # Copy the receiving buffer to the guard cells of the domain
            copy_J_buffers( self, interp, after_receiving=True )

        elif fieldtype == 'rho':

            # Copy the inner part of the domain to the sending buffer
            copy_rho_buffers( self, interp, before_sending=True )
            # Exchange the guard regions between the domains (MPI)
            self.exchange_domains( self.rho_send_l, self.rho_send_r,
                                  self.rho_recv_l, self.rho_recv_r)
            # An MPI barrier is needed here so that a single rank does not
            # do two sends and receives before this exchange is completed.
            self.mpi_comm.Barrier()
            # Copy the receiving buffer to the guard cells of the domain
            copy_rho_buffers( self, interp, after_receiving=True )

        else:
            raise ValueError('Unknown fieldtype : %s' %fieldtype)


    def exchange_domains( self, send_left, send_right, recv_left, recv_right ):
        """
        Send the arrays send_left and send_right to the left and right
        processes respectively.
        Receive the arrays from the neighboring processes into recv_left
        and recv_right.
        Sending and receiving is done from CPU to CPU.

        Parameters :
        ------------
        - send_left, send_right, recv_left, recv_right : arrays 
             Sending and receiving buffers
        """
        # MPI-Exchange: Uses non-blocking send and receive, 
        # which return directly and need to be synchronized later.
        # Send to left domain and receive from right domain
        if self.left_proc is not None :
            self.mpi_comm.Isend(send_left, dest=self.left_proc, tag=1)
        if self.right_proc is not None :
            req_1 = self.mpi_comm.Irecv(recv_right,
                                        source=self.right_proc, tag=1)
        # Send to right domain and receive from left domain
        if self.right_proc is not None :
            self.mpi_comm.Isend(send_right, dest=self.right_proc, tag=2)
        if self.left_proc is not None :
            req_2 = self.mpi_comm.Irecv(recv_left,
                                        source=self.left_proc, tag=2)
        # Wait for the non-blocking sends to be received (synchronization)
        if self.right_proc is not None :
            re_1 = mpi.Request.Wait(req_1)
        if self.left_proc is not None :
            re_2 = mpi.Request.Wait(req_2)


    def exchange_particles(self, ptcl, zmin, zmax) :
        """
        Look for particles that are located inside the guard cells
        and exchange them with the corresponding neighboring processor.

        Parameters :
        ------------
        ptcl : a Particle object
            The object corresponding to a given species
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

        # If needed, copy the particles to the CPU
        if ptcl.use_cuda:
            ptcl.receive_particles_from_gpu()
            
        # Select the particles that are in the left or right guard cells,
        # and those that stay on the local process
        selec_left = ( ptcl.z < (zmin + ng*dz) )
        selec_right = ( ptcl.z > (zmax - ng*dz) )
        selec_stay = (np.logical_not(selec_left) & np.logical_not(selec_right))
        # Count them, and convert the result into an array
        # so as to send them easily.
        N_send_l = np.array( selec_left.sum(), dtype=np.int32)
        N_send_r = np.array( selec_right.sum(), dtype=np.int32)
        N_stay = selec_stay.sum()

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

        # If needed, copy the particles to the GPU
        if ptcl.use_cuda:
            ptcl.send_particles_to_gpu()

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

        # Copy the array to the GPU if possible
        if cuda_installed:
            self.d_damp_array = cuda.to_device(self.damp_array)

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
            gathered_array = np.zeros((self.Nz, self.Nr), dtype=array.dtype)
        else:
            # Other processes do not need to initialize a new array
            gathered_array = None
        # Shortcut for the guard cells
        ng = self.n_guard
        
        # Call the mpi4py routine Gartherv
        # First get the size and MPI type of the 2D arrays in each procs
        i_start_procs = tuple( self.Nr*iz for iz in self.iz_start_procs )
        N_domain_procs = tuple( self.Nr*nz for nz in self.Nz_domain_procs )
        mpi_type = mpi_type_dict[ str(array.dtype) ] 
        # Then send the arrays
        sendbuf = [ array[ng:-ng,:], N_domain_procs[self.rank] ]
        recvbuf = [ gathered_array, N_domain_procs, i_start_procs, mpi_type ]
        self.mpi_comm.Gatherv( sendbuf, recvbuf, root=root )
        # Return the gathered_array only on process root
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
        # Get the local number of particle on each proc, and the particle number
        n_rank = self.mpi_comm.allgather( ptcl.Ntot )
        Ntot = sum(n_rank)
        # Loop over particle attributes that need to be gathered
        for particle_attr in ['x', 'y', 'z', 'ux', 'uy', 'uz','inv_gamma', 'w']:
            # Get array of particle attribute
            array = getattr(ptcl, particle_attr)
            # Gather array on process root
            gathered_array = self.gather_ptcl_array(array, n_rank, Ntot, root)
            if self.rank == root:
                # Write array to particle attribute in the gathered object
                setattr(gathered_ptcl, particle_attr, gathered_array)
        # Return the gathered particle object
        return(gathered_ptcl)

    def gather_ptcl_array(self, array, n_rank, Ntot, root = 0):
        """
        Gather a particle array on the root process by using the
        mpi4py routine Gatherv, that gathers arbitrary shape arrays
        by combining the first dimension in ascending order.

        Parameter :
        -----------
        array : array (ptcl array)
            A particle array of the local domain

        n_rank : list of ints
            A list containing the number of particles to send on each proc
            
        Ntot : int
            The total number of particles for all the proc together

        root : int, optional
            Process that gathers the data

        Returns :
        ---------
        gathered_array : array (global ptcl array)
            A gathered array that contains the global simulation data
        """
        # Prepare the output array
        if self.rank == root:
            # Root process creates empty numpy array
            gathered_array = np.empty(Ntot, dtype=array.dtype)
        else:
            # Other processes do not need to initialize a new array
            gathered_array = None

        # Prepare the send and receive buffers
        i_start_procs = tuple( np.cumsum([0] + n_rank[:-1]) )
        n_rank_procs = tuple( n_rank )
        mpi_type = mpi_type_dict[ str(array.dtype) ]
        sendbuf = [ array, n_rank_procs[self.rank] ]
        recvbuf = [ gathered_array, n_rank_procs, i_start_procs, mpi_type ]

        # Send/receive the arrays
        self.mpi_comm.Gatherv( sendbuf, recvbuf, root=root )
        
        # Return the gathered_array only on process root
        if self.rank == root:
            return(gathered_array)
