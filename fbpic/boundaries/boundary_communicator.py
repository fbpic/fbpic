"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to implement the boundary exchanges.
"""
import numpy as np
from mpi4py import MPI as mpi
from fbpic.fields.fields import InterpolationGrid
from fbpic.particles.particles import Particles
from .mpi_buffer_handling import BufferHandler
from .guard_cell_damping import GuardCellDamper

# Dictionary of correspondance between numpy types and mpi types
# (Necessary when calling Gatherv)
mpi_type_dict = { 'float32': mpi.REAL4,
                  'float64': mpi.REAL8,
                  'complex64': mpi.COMPLEX8,
                  'complex128': mpi.COMPLEX16 }

class BoundaryCommunicator(object):
    """
    Class that handles the boundary conditions along z, esp.
    the moving window and MPI communication between domains.
    It also handles the initial domain decomposition.

    The functions of this object is:
    
    - At each timestep, to exchange the fields between MPI domains
      and damp the E and B fields in the guard cells

    - Every exchange_period iterations, to exchange the particles
      between MPI domains and (in the case of a moving window) shift the grid

    - When the diagnostics are called, to gather the fields and particles
    """

    def __init__( self, Nz, Nr, n_guard, Nm, boundaries='periodic',
                  exchange_period=None ):
        """
        Initializes a communicator object.

        Parameters
        ----------

        Nz, Nr: int
            The initial global number of cells

        n_guard: int
            The number of guard cells at the 
            left and right edge of the domain

        Nm: int
            The total number of modes

        boundaries: str
            Indicates how to exchange the fields at the left and right
            boundaries of the global simulation box
            Either 'periodic' or 'open'

        exchange_period: int
            Indicates how often to move the moving window and exchange
            the particles. (These 2 operations are done simultaneously.)

        v_moving: int
            Speed of the moving window. Use 0 for no moving window.
        """
        # Initialize global number of cells and modes
        self.Nz = Nz
        self.Nr = Nr
        self.Nm = Nm
        
        # MPI Setup
        self.mpi_comm = mpi.COMM_WORLD
        self.rank = self.mpi_comm.rank
        self.size = self.mpi_comm.size
        # Get the rank of the left and the right domain
        self.left_proc = self.rank-1
        self.right_proc = self.rank+1
        # Correct these initial values by taking into account boundaries
        if boundaries == 'periodic':
            # Periodic boundary conditions for the domains
            if self.rank == 0: 
                self.left_proc = (self.size-1)
            if self.rank == self.size-1:
                self.right_proc = 0
        elif boundaries == 'open':
            # None means that the boundary is open
            if self.rank == 0:
                self.left_proc = None
            if self.rank == self.size-1:
                self.right_proc = None
        else:
            raise ValueError('Unrecognized boundaries: %s' %self.boundaries)

        # Initialize number of guard cells
        # For single proc and periodic boundaries, no need for guard cells
        if boundaries=='periodic' and self.size==1:
            n_guard = 0
        self.n_guard = n_guard

        # Initialize the period of the particle exchange and moving window
        if exchange_period is None:
            self.exchange_period = max(1, int(n_guard/2))
        else:
            self.exchange_period = exchange_period

        # Initialize the moving window to None (See the method
        # set_moving_window in main.py to initialize a proper moving window)
        self.moving_win = None

        # Initialize a buffer handler object, for MPI communications
        if self.size > 1:
            self.mpi_buffers = BufferHandler( self.n_guard, Nr, Nm,
                                      self.left_proc, self.right_proc )

        # Create damping object for the guard cells
        if self.n_guard > 0:
            self.guard_damper = GuardCellDamper( self.n_guard,
                    self.left_proc, self.right_proc, self.exchange_period )

    def divide_into_domain( self, zmin, zmax, p_zmin, p_zmax ):
        """
        Divide the global simulation into domain and add local guard cells.

        Return the new size of the local domain (zmin, zmax) and the
        boundaries of the initial plasma (p_zmin, p_zmax)

        Parameters:
        ------------
        zmin, zmax: floats
            Positions of the edges of the global simulation box
            (without guard cells)

        p_zmin, p_zmax: floats
            Positions between which the plasma will be initialized, in
            the global simulation box.

        Returns:
        ---------
        A tuple with 
        zmin, zmax: floats
            Positions of the edges of the local simulation box
            (with guard cells)

        p_zmin, p_zmax: floats
           Positions between which the plasma will be initialized, in
           the local simulation box.
           (NB: no plasma will be initialized in the guard cells)

        Nz_enlarged: int
           The number of cells in the local simulation box (with guard cells)
        """
        # Initialize global box size
        self.Ltot = (zmax-zmin)
        # Get the distance dz between the cells
        # (longitudinal spacing of the grid)
        dz = (zmax - zmin)/self.Nz
        self.dz = dz
            
        # Initialize the number of cells of each proc
        # (Splits the global simulation and
        # adds guard cells to the local domain)
        Nz_per_proc = int(self.Nz/self.size)
        # Get the number of cells in each domain
        # (The last proc gets extra cells, so as to have Nz cells in total)
        Nz_domain_procs = [ Nz_per_proc for k in range(self.size) ]
        Nz_domain_procs[-1] = Nz_domain_procs[-1] + (self.Nz)%(self.size)
        self.Nz_domain_procs = Nz_domain_procs
        # Get the starting index (for easy output)
        self.iz_start_procs = [ k*Nz_per_proc for k in range(self.size) ]
        # Get the enlarged number of cells in each domain (includes guards)
        self.Nz_enlarged_procs = [ n+2*self.n_guard for n in Nz_domain_procs ]
        # Get the local values of the above arrays
        self.Nz_domain = self.Nz_domain_procs[self.rank]
        self.Nz_enlarged = self.Nz_enlarged_procs[self.rank]

        # Check if the local domain size is large enough
        if self.Nz_enlarged < 4*self.n_guard:
            raise ValueError( 'Number of local cells in z is smaller \
                               than 4 times n_guard. Use fewer domains or \
                               a smaller number of guard cells.')
        
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

    def move_grids( self, interp, dt ):
        """
        Calculate by how many cells the moving window should be moved.
        If this is non-zero, shift the fields on the interpolation grid,
        and add new particles.

        NB: the spectral grid is not modified, as it is automatically
        updated after damping the fields (see main.py)

        Parameters
        ----------
        interp: a list of Interpolation object
            Contains the fields data of the simulation
    
        dt: float (in seconds)
            Timestep of the simulation
        """
        self.moving_win.move_grids(interp, dt, self.mpi_comm)
           
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

        Parameters:
        ------------
        interp: list
            A list of InterpolationGrid objects
            (one element per azimuthal mode)

        fieldtype: str
            An identifier for the field to send
            (Either 'EB', 'J' or 'rho')
        """
        # Only perform the exchange if there is more than 1 proc
        if self.size > 1:

            if fieldtype == 'EB':

                # Copy the inner part of the domain to the sending buffer
                self.mpi_buffers.copy_EB_buffers( interp, before_sending=True )
                # Copy the sending buffers to the receiving buffers via MPI
                self.exchange_domains(
                    self.mpi_buffers.EB_send_l, self.mpi_buffers.EB_send_r,
                    self.mpi_buffers.EB_recv_l, self.mpi_buffers.EB_recv_r )
                # An MPI barrier is needed here so that a single rank does not
                # do two sends and receives before this exchange is completed.
                self.mpi_comm.Barrier()
                # Copy the receiving buffer to the guard cells of the domain
                self.mpi_buffers.copy_EB_buffers( interp, after_receiving=True )

            elif fieldtype == 'J':

                # Copy the inner part of the domain to the sending buffer
                self.mpi_buffers.copy_J_buffers( interp, before_sending=True )
                # Copy the sending buffers to the receiving buffers via MPI
                self.exchange_domains(
                    self.mpi_buffers.J_send_l, self.mpi_buffers.J_send_r,
                    self.mpi_buffers.J_recv_l, self.mpi_buffers.J_recv_r )
                # An MPI barrier is needed here so that a single rank does not
                # do two sends and receives before this exchange is completed.
                self.mpi_comm.Barrier()
                # Copy the receiving buffer to the guard cells of the domain
                self.mpi_buffers.copy_J_buffers( interp, after_receiving=True )

            elif fieldtype == 'rho':

                # Copy the inner part of the domain to the sending buffer
                self.mpi_buffers.copy_rho_buffers( interp, before_sending=True )
                # Copy the sending buffers to the receiving buffers via MPI
                self.exchange_domains(
                    self.mpi_buffers.rho_send_l, self.mpi_buffers.rho_send_r,
                    self.mpi_buffers.rho_recv_l, self.mpi_buffers.rho_recv_r )
                # An MPI barrier is needed here so that a single rank does not
                # do two sends and receives before this exchange is completed.
                self.mpi_comm.Barrier()
                # Copy the receiving buffer to the guard cells of the domain
                self.mpi_buffers.copy_rho_buffers(interp, after_receiving=True)

            else:
                raise ValueError('Unknown fieldtype: %s' %fieldtype)

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
            mpi.Request.Wait(req_1)
        if self.left_proc is not None :
            mpi.Request.Wait(req_2)
            
    def exchange_particles(self, ptcl, zmin, zmax):
        """
        Look for particles that are located inside the guard cells
        and exchange them with the corresponding neighboring processor.

        Parameters:
        ------------
        ptcl: a Particle object
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
        # Reallocate the cell index and sorted index arrays on the CPU
        ptcl.cell_idx = np.empty(ptcl.Ntot, dtype = np.int32)
        ptcl.sorted_idx = np.arange(ptcl.Ntot, dtype = np.uint32)
        ptcl.particle_buffer = np.arange(ptcl.Ntot, dtype = np.float64)
        # If needed, copy the particles to the GPU
        if ptcl.use_cuda:
            ptcl.send_particles_to_gpu()
        # The particles are unsorted after adding new particles.
        ptcl.sorted = False

    def damp_guard_EB( self, interp ):
        """
        Apply the damping shape in the right and left guard cells.

        Parameter:
        -----------
        interp: list
            A list of InterpolationGrid objects (one per azimuthal mode)
        """
        # Do not damp the fields for 0 guard cells (periodic, single proc)
        if self.n_guard != 0:
            self.guard_damper.damp_guard_EB( interp )

    def gather_grid( self, grid, root = 0):
        """
        Gather a grid object by combining the local domains
        without the guard regions to a new global grid object.

        Parameter:
        -----------
        grid: Grid object (InterpolationGrid)
            A grid object that is gathered on the root process

        root: int, optional
            Process that gathers the data

        Returns:
        ---------
        gathered_grid: Grid object (InterpolationGrid)
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

        Parameter:
        -----------
        array: array (grid array)
            A grid array of the local domain

        root: int, optional
            Process that gathers the data

        Returns:
        ---------
        gathered_array: array (global grid array)
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

        Parameter:
        -----------
        ptcl: Particle object
            A particle object that is gathered on the root process

        root: int, optional
            Process that gathers the data

        Returns:
        ---------
        gathered_ptcl: Particle object
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
        for particle_attr in ['x', 'y', 'z', 'ux', 'uy',
                              'uz', 'inv_gamma', 'w']:
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

        Parameter:
        -----------
        array: array (ptcl array)
            A particle array of the local domain

        n_rank: list of ints
            A list containing the number of particles to send on each proc
            
        Ntot: int
            The total number of particles for all the proc together

        root: int, optional
            Process that gathers the data

        Returns:
        ---------
        gathered_array: array (global ptcl array)
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


