# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to implement the boundary exchanges.
"""
import numpy as np
from scipy.constants import c
from mpi4py import MPI as mpi
from fbpic.fields.fields import InterpolationGrid
from fbpic.fields.utility_methods import get_stencil_reach
from fbpic.particles.particles import Particles
from .field_buffer_handling import BufferHandler
from .particle_buffer_handling import remove_outside_particles, \
     add_buffers_to_particles, shift_particles_periodic_subdomain
# Check if CUDA is available, then import CUDA functions
from fbpic.cuda_utils import cuda_installed
if cuda_installed:
    from fbpic.cuda_utils import cuda, cuda_tpb_bpg_2d
    from .cuda_methods import cuda_damp_EB_left, cuda_damp_EB_right

# Dictionary of correspondance between numpy types and mpi types
# (Necessary when calling Gatherv)
mpi_type_dict = { 'float32': mpi.REAL4,
                  'float64': mpi.REAL8,
                  'complex64': mpi.COMPLEX8,
                  'complex128': mpi.COMPLEX16,
                  'uint64': mpi.UINT64_T }

class BoundaryCommunicator(object):
    """
    Class that handles the boundary conditions along z, esp.
    the moving window and MPI communication between domains.
    It also handles the initial domain decomposition.

    The functions of this object are:

    - At each timestep, to exchange the fields between MPI domains
      in the guard cells (n_guard) and damp the E and B fields in the damping
      guard cells (n_damp)

    - Every iteration, to move the grid in case of a moving window

    - Every exchange_period iterations, to exchange the particles
      between MPI domains / or add and remove particles

    - When the diagnostics are called, to gather the fields and particles
    """

    # Initialization routines
    # -----------------------

    def __init__( self, Nz, zmin, zmax, Nr, rmax, Nm, dt,
            boundaries, n_order, n_guard = None, n_damp=30,
            exchange_period = None, use_all_mpi_ranks=True ):
        """
        Initializes a communicator object.

        Parameters
        ----------
        Nz, Nr: int
            The initial global number of cells

        zmin, zmax, rmax: float
            The position of the edges of the simulation box in z and r
            (More precisely, the position of the edge of the (first)
             last cell)

        Nm: int
            The total number of modes

        dt: float
            The timestep of the simulation

        boundaries: str
            Indicates how to exchange the fields at the left and right
            boundaries of the global simulation box
            Either 'periodic' or 'open'

        n_order: int
           The order of the stencil for the z derivatives.
           Use -1 for infinite order, otherwise use a positive, even
           number. In this case, the stencil extends up to approx.
           2*n_order cells on each side. (A finite order stencil
           is required to have a localized field push that allows
           to do simulations in parallel on multiple MPI ranks)

        n_guard: int
            Number of guard cells to use at the left and right of
            a domain, when using MPI.

        n_damp : int
            Number of damping guard cells at the left and right of a
            simulation box if a moving window is attached. The guard
            region at these areas (left / right of moving window) is
            extended by n_damp (N=n_guard+n_damp) in order to smoothly
            damp the fields such that they do not wrap around.
            (Defaults to 30)

        exchange_period: int, optional
            Number of iterations before which the particles are exchanged.
            If set to None, the minimum exchange period is calculated
            automatically: Within exchange_period timesteps, the
            particles should never be able to travel more than
            (n_guard - particle_shape order) cells.

        use_all_mpi_ranks: bool, optional
            - if `use_all_mpi_ranks` is True (default):
              All the MPI ranks will contribute to the same simulation,
              using domain-decomposition to share the work.
            - if `use_all_mpi_ranks` is False:
              Each MPI rank will run an independent simulation.
              This can be useful when running parameter scans.
        """
        # Initialize global number of cells and modes
        self.Nz = Nz
        self.Nr = Nr
        self.Nm = Nm

        # Get the distance dz between the cells
        # (longitudinal spacing of the grid)
        self.dz = (zmax - zmin)/self.Nz

        # MPI Setup
        if use_all_mpi_ranks:
            self.mpi_comm = mpi.COMM_WORLD
            self.rank = self.mpi_comm.rank
            self.size = self.mpi_comm.size
        else:
            self.mpi_comm = None
            self.rank = 0
            self.size = 1
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
        # Automatically calculate required guard cells
        # for given order (n_order)
        if n_guard == None:
            if n_order == -1:
                # Set n_guard to fixed value of 30 in case of
                # open boundaries and infinite order stencil
                # (if not defined otherwise by user)
                self.n_guard = 30
                # Raise error if user tries to use parallel MPI computation
                # with an infinite order stencil. This would give wrong results
                if self.size != 1:
                    raise ValueError('Non-local, infinite order stencil \
                        selected, while performing parallel computation.')
            else:
                # Automatic calculation of the guard region size,
                # depending on the stencil order (n_order)
                stencil = get_stencil_reach( self.Nz, self.dz, c*dt, n_order )
                # approx 2*n_order (+1 because the moving window
                # shifts the grid by one cell during the PIC loop
                # and therefore, the guard region needs to be larger
                # by one cell)
                self.n_guard = stencil + 1
        else:
            # Otherwise: Set user defined guard region size
            self.n_guard = n_guard
        # For single proc and periodic boundaries, no need for guard cells
        if boundaries=='periodic' and self.size==1:
            self.n_guard = 0
        # Register damping cells
        self.n_damp = n_damp
        # For periodic boundaries, no need for damping cells
        if boundaries=='periodic':
            self.n_damp = 0

        # Initialize the period of the particle exchange and moving window
        if exchange_period is None:
            # Maximum number of cells a particle can travel in one timestep
            # Safety factor of 2 needed if there is a moving window attached
            # to the simulation or in case a galilean frame is used.
            cells_per_step = 2.*c*dt/self.dz
            # Maximum number of timesteps before a particle can reach the end
            # of the guard region including the maximum number of cells (+/-3)
            # it can affect with a "cubic" particle shape_factor.
            self.exchange_period = int( (self.n_guard-3)/cells_per_step )
            # Set exchange_period to 1 in the case of single-proc
            # and periodic boundary conditions.
            if self.size == 1 and boundaries == 'periodic':
                self.exchange_period = 1
            # Check that calculated exchange_period is acceptable for given
            # simulation parameters (check that guard region is large enough).
            if self.exchange_period < 1:
                raise ValueError('Guard region size is too small for chosen \
                    timestep. In one timestep, a particle can travel more \
                    than n_guard region cells.')
        else:
            # User-defined exchange_period. Choose carefully.
            self.exchange_period = exchange_period

        # Initialize the moving window to None (See the method
        # set_moving_window in main.py to initialize a proper moving window)
        self.moving_win = None

        # Initialize a buffer handler object, for MPI communications
        if self.size > 1:
            self.mpi_buffers = BufferHandler( self.n_guard, Nr, Nm,
                                      self.left_proc, self.right_proc )

        # Create damping arrays for the damping cells at the left
        # and right of the box in the case of "open" boundaries.
        if self.n_damp > 0:
            if self.left_proc is None:
                # Create the damping arrays for left proc
                self.left_damp = self.generate_damp_array(
                    self.n_guard, self.n_damp )
                if cuda_installed:
                    self.d_left_damp = cuda.to_device( self.left_damp )
            if self.right_proc is None:
                # Create the damping arrays for right proc
                self.right_damp = self.generate_damp_array(
                    self.n_guard, self.n_damp )
                if cuda_installed:
                    self.d_right_damp = cuda.to_device( self.right_damp )

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
        # Add damping region to first and last domain
        # (Note: self.n_damp is zero in case of 'periodic' boundaries)
        self.Nz_enlarged_procs[0] += self.n_damp
        self.Nz_enlarged_procs[-1] += self.n_damp
        # Get the local values of the above arrays
        self.Nz_domain = self.Nz_domain_procs[self.rank]
        self.Nz_enlarged = self.Nz_enlarged_procs[self.rank]

        # Check if the local domain size is large enough
        if self.Nz_enlarged < 4*self.n_guard:
            raise ValueError( 'Number of local cells in z is smaller \
                               than 4 times n_guard. Use fewer domains or \
                               a smaller number of guard cells.')

        # Calculate the local boundaries,
        # zmin_local_domain and zmax_local_domain,
        # of this local simulation box.
        iz_start = self.iz_start_procs[self.rank]
        zmin_local_domain = zmin + iz_start*dz
        zmax_local_domain = zmin_local_domain + self.Nz_domain*dz
        # Calculate the new limits (p_zmin and p_zmax)
        # for adding particles to this domain
        p_zmin_local_domain = max( zmin_local_domain, p_zmin)
        p_zmax_local_domain = min( zmax_local_domain, p_zmax)

        # Calculate the enlarged boundaries (i.e. including guard cells
        # and damp cells), which are passed to the fields object.
        zmin_local_enlarged = zmin_local_domain - self.n_guard*dz
        zmax_local_enlarged = zmax_local_domain + self.n_guard*dz
        if self.left_proc is None:
            zmin_local_enlarged -= self.n_damp*dz
        if self.right_proc is None:
            zmax_local_enlarged += self.n_damp*dz

        # Return the new boundaries to the simulation object
        return( zmin_local_enlarged, zmax_local_enlarged,
                p_zmin_local_domain, p_zmax_local_domain,
                self.Nz_enlarged )

    # Exchange routines
    # -----------------

    def move_grids( self, fld, dt, time ):
        """
        Calculate by how many cells the moving window should be moved.
        If this is non-zero, shift the fields on the interpolation grid,
        and add new particles.

        NB: the spectral grid is not modified, as it is automatically
        updated after damping the fields (see main.py)

        Parameters
        ----------
        fld: a Fields object
            Contains the fields data of the simulation

        dt: float (in seconds)
            Timestep of the simulation

        time: float (seconds)
            The global time in the simulation
            This is used in order to determine how much the window should move
        """
        self.moving_win.move_grids(fld, self, time)

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
                self.mpi_buffers.copy_EB_buffers(interp, before_sending=True)
                # Copy the sending buffers to the receiving buffers via MPI
                self.exchange_domains(
                    self.mpi_buffers.EB_send_l, self.mpi_buffers.EB_send_r,
                    self.mpi_buffers.EB_recv_l, self.mpi_buffers.EB_recv_r )
                # An MPI barrier is needed here so that a single rank does not
                # do two sends and receives before this exchange is completed.
                self.mpi_comm.Barrier()
                # Copy the receiving buffer to the guard cells of the domain
                self.mpi_buffers.copy_EB_buffers(interp, after_receiving=True)

            elif fieldtype == 'J':

                # Copy the inner part of the domain to the sending buffer
                self.mpi_buffers.copy_J_buffers(interp, before_sending=True)
                # Copy the sending buffers to the receiving buffers via MPI
                self.exchange_domains(
                    self.mpi_buffers.J_send_l, self.mpi_buffers.J_send_r,
                    self.mpi_buffers.J_recv_l, self.mpi_buffers.J_recv_r )
                # An MPI barrier is needed here so that a single rank does not
                # do two sends and receives before this exchange is completed.
                self.mpi_comm.Barrier()
                # Copy the receiving buffer to the guard cells of the domain
                self.mpi_buffers.copy_J_buffers(interp, after_receiving=True)

            elif fieldtype == 'rho':

                # Copy the inner part of the domain to the sending buffer
                self.mpi_buffers.copy_rho_buffers(interp, before_sending=True)
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
            req_1.Wait()
        if self.left_proc is not None :
            req_2.Wait()

    def exchange_particles(self, species, fld, time ):
        """
        Look for particles that are located outside of the physical boundaries
        and:
         - for open boundaries: remove the particles at the edges of the global
           box and (when using a moving window) add the new particles from
           the moving window
         - for boundaries with neighboring processors: exchange particles that
           are outside of the local physical subdomain
         - for single-proc periodic simulation (periodic boundaries on both
           sides): simply shift the particle positions by an integer number
           of box length, so that outside particle are back inside the
           physical domain

        Parameters:
        ------------
        species: a Particle object
            The object corresponding to a given species

        fld: a Fields object
            Contains information about the dimension of the grid,
            and the prefix sum (when using the GPU).
            The object itself is not modified by this routine.

        time: float (seconds)
            The global time of the simulation
            (Needed in the case of a flowing plasma which is generate
            from a density profile: in the case the time is used in
            order to infer how much the plasma has moved)
        """
        # For single-proc periodic simulation (periodic boundaries)
        # simply shift the particle positions by an integer number
        if self.n_guard == 0:
            shift_particles_periodic_subdomain( species,
                    fld.interp[0].zmin, fld.interp[0].zmax )
        # Otherwise, remove particles that are outside of the local physical
        # subdomain and send them to neighboring processors
        else:
            self.exchange_particles_aperiodic_subdomain( species, fld, time )

    def exchange_particles_aperiodic_subdomain(self, species, fld, time ):
        """
        Look for particles that are located outside of the physical boundaries
        of the local subdomain and exchange them with the corresponding
        neighboring processor.
        Also remove the particles that are below the left boundary of the
        global box, and (when using the moving window) add particles at the
        right boundary of the global box.

        Parameters:
        ------------
        species: a Particle object
            The object corresponding to a given species

        fld: a Fields object
            Contains information about the dimension of the grid,
            and the prefix sum (when using the GPU).
            The object itself is not modified by this routine.

        time: float (seconds)
            The global time of the simulation
            (Needed in the case of a flowing plasma which is generate
            from a density profile: in the case the time is used in
            order to infer how much the plasma has moved)
        """
        # Remove out-of-domain particles from particle arrays (either on
        # CPU or GPU) and store them in sending buffers on the CPU
        float_send_left, float_send_right, uint_send_left, uint_send_right = \
            remove_outside_particles( species, fld, self.n_guard,
                                    self.left_proc, self.right_proc )

        # Send/receive the number of particles (need to be stored in arrays)
        N_send_l = np.array( float_send_left.shape[1], dtype=np.uint32 )
        N_send_r = np.array( float_send_right.shape[1], dtype=np.uint32 )
        N_recv_l = np.array( 0, dtype=np.uint32 )
        N_recv_r = np.array( 0, dtype=np.uint32 )
        self.exchange_domains(N_send_l, N_send_r, N_recv_l, N_recv_r)
        # Note: if left_proc or right_proc is None, the
        # corresponding N_recv remains 0 (no exchange)
        if self.size > 1:
            self.mpi_comm.Barrier()

        # Allocate the receiving buffers and exchange particles
        n_float = float_send_left.shape[0]
        float_recv_left = np.zeros((n_float, N_recv_l), dtype=np.float64)
        float_recv_right = np.zeros((n_float, N_recv_r), dtype=np.float64)
        self.exchange_domains( float_send_left, float_send_right,
                                float_recv_left, float_recv_right )
        # Integers (e.g. particle id), if any
        n_int = uint_send_left.shape[0]
        uint_recv_left = np.zeros((n_int, N_recv_l), dtype=np.uint64 )
        uint_recv_right = np.zeros((n_int, N_recv_r), dtype=np.uint64 )
        if n_int > 0:
            self.exchange_domains( uint_send_left, uint_send_right,
                                    uint_recv_left, uint_recv_right )

        # When using a moving window, create new particles in recv_right
        # (Overlap this with the exchange of domains, since recv_right
        # will not be affected by the exchange at this open boundary)
        if (self.moving_win is not None) and (self.rank == self.size-1):
            float_recv_right, uint_recv_right = \
              self.moving_win.generate_particles(species,fld.interp[0].dz,time)

        # An MPI barrier is needed here so that a single rank
        # does not perform two sends and receives before all
        # the other MPI connections within this exchange are completed.
        if self.size > 1:
            self.mpi_comm.Barrier()

        # Periodic boundary conditions for exchanging particles
        # Particles received at the right (resp. left) end of the simulation
        # box are shifted by Ltot (zmax-zmin) to the right (resp. left).
        if self.right_proc == 0:
            # The index 2 corresponds to z
            float_recv_right[2,:] = float_recv_right[2,:] + self.Ltot
        if self.left_proc == self.size-1:
            # The index 2 corresponds to z
            float_recv_left[2,:] = float_recv_left[2,:] - self.Ltot

        # Add the exchanged buffers to the particles on the CPU or GPU
        # and resize the auxiliary field-on-particle and sorting arrays
        add_buffers_to_particles( species, float_recv_left, float_recv_right,
                                    uint_recv_left, uint_recv_right )

    def damp_guard_EB( self, interp ):
        """
        Apply the damping shape in the right and left guard cells.
        Damp the fields E and B in the guard cells.

        Parameter:
        -----------
        interp: list of InterpolationGrid objects (one per azimuthal mode)
            Objects that contain the fields to be damped.
        """
        # Do not damp the fields for 0 n_damp cells (periodic)
        if self.n_damp != 0:
            if self.left_proc is None:
                # Damp the fields on the CPU or the GPU
                if interp[0].use_cuda:
                    # Damp the fields on the GPU
                    dim_grid, dim_block = cuda_tpb_bpg_2d(
                        self.n_guard+self.n_damp, interp[0].Nr )

                    cuda_damp_EB_left[dim_grid, dim_block](
                        interp[0].Er, interp[0].Et, interp[0].Ez,
                        interp[0].Br, interp[0].Bt, interp[0].Bz,
                        interp[1].Er, interp[1].Et, interp[1].Ez,
                        interp[1].Br, interp[1].Bt, interp[1].Bz,
                        self.d_left_damp, self.n_guard, self.n_damp)
                else:
                    # Damp the fields on the CPU
                    nd = self.n_guard + self.n_damp
                    for m in range(len(interp)):
                        # Damp the fields in left guard cells
                        interp[m].Er[:nd,:]*=self.left_damp[:,np.newaxis]
                        interp[m].Et[:nd,:]*=self.left_damp[:,np.newaxis]
                        interp[m].Ez[:nd,:]*=self.left_damp[:,np.newaxis]
                        interp[m].Br[:nd,:]*=self.left_damp[:,np.newaxis]
                        interp[m].Bt[:nd,:]*=self.left_damp[:,np.newaxis]
                        interp[m].Bz[:nd,:]*=self.left_damp[:,np.newaxis]

            if self.right_proc is None:
                # Damp the fields on the CPU or the GPU
                if interp[0].use_cuda:
                    # Damp the fields on the GPU
                    dim_grid, dim_block = cuda_tpb_bpg_2d(
                        self.n_guard+self.n_damp, interp[0].Nr )

                    cuda_damp_EB_right[dim_grid, dim_block](
                        interp[0].Er, interp[0].Et, interp[0].Ez,
                        interp[0].Br, interp[0].Bt, interp[0].Bz,
                        interp[1].Er, interp[1].Et, interp[1].Ez,
                        interp[1].Br, interp[1].Bt, interp[1].Bz,
                        self.d_right_damp, self.n_guard, self.n_damp)
                else:
                    # Damp the fields on the CPU
                    nd = self.n_guard + self.n_damp
                    for m in range(len(interp)):
                        # Damp the fields in left guard cells
                        interp[m].Er[-nd:,:]*=self.right_damp[::-1,np.newaxis]
                        interp[m].Et[-nd:,:]*=self.right_damp[::-1,np.newaxis]
                        interp[m].Ez[-nd:,:]*=self.right_damp[::-1,np.newaxis]
                        interp[m].Br[-nd:,:]*=self.right_damp[::-1,np.newaxis]
                        interp[m].Bt[-nd:,:]*=self.right_damp[::-1,np.newaxis]
                        interp[m].Bz[-nd:,:]*=self.right_damp[::-1,np.newaxis]

    def generate_damp_array( self, n_guard, n_damp ):
        """
        Create a 1d damping array of length n_guard.

        The expression of the damping array depends on whether the guard cells
        correspond to an open boundary or a boundary with another processor.

        Parameters
        ----------
        n_guard: int
            Number of guard cells along z

        n_damp: int
            Number of damping cells along z

        Returns
        -------
        A 1darray of doubles, of length n_guard + n_damp,
        which represents the damping.
        """
        # Array of cell indices
        i_cell = np.arange( n_guard+n_damp )

        # Perform narrow damping, with the first n_guard of the cells at 0,
        # then 1/3*n_damp cells with a sinusoidal**2 rise, and finally
        # 2/3*n_damp cells at 1 (the damping array is defined such that it
        # can directly be multiplied with the fields at the left boundary of
        # the box - and needs to be inverted (damping_array[::-1]) before being
        # applied to the right boundary of the box.)
        damping_array = np.where( i_cell < n_guard+n_damp/3.,
                np.sin((i_cell - n_guard)*np.pi/(2*n_damp/3.))**2, 1. )
        damping_array = np.where( i_cell < n_guard, 0., damping_array )

        return( damping_array )

    # Gathering routines
    # ------------------

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
        # Calculate global edges of the simulation box on root process
        if self.rank == root:
            n_remove = self.n_guard
            if self.left_proc is None:
                # Add damp cells if root process is rank 0
                n_remove += self.n_damp
            # Calculate the global zmin without the guard (and damp) region
            zmin_global = grid.zmin + self.dz * \
                (n_remove - self.rank*self.Nz_domain)
            # Create new grid array that contains cell positions in z
            z = zmin_global + self.dz*( 0.5 + np.arange(self.Nz) )
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

        # Guard region cells to be removed at the left and right
        n_remove_l = self.n_guard
        n_remove_r = self.n_guard
        # Remove n_damp cells from the left proc's output region
        if self.left_proc is None:
            n_remove_l += self.n_damp
        # Remove n_damp cells from the right proc's output region
        if self.right_proc is None:
            n_remove_r += self.n_damp
        # Select the physical region of the local box
        local_array = array[n_remove_l:len(array)-n_remove_r,:]

        # Then send the arrays
        if self.size > 1:
            # First get the size and MPI type of the 2D arrays in each procs
            i_start_procs = tuple( self.Nr*iz for iz in self.iz_start_procs )
            N_domain_procs = tuple( self.Nr*nz for nz in self.Nz_domain_procs )
            mpi_type = mpi_type_dict[ str(array.dtype) ]
            sendbuf = [ local_array, N_domain_procs[self.rank] ]
            recvbuf = [ gathered_array, N_domain_procs,
                        i_start_procs, mpi_type ]
            self.mpi_comm.Gatherv( sendbuf, recvbuf, root=root )
        else:
            gathered_array[:,:] = local_array

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

        if self.size > 1:
            # Prepare the send and receive buffers
            i_start_procs = tuple( np.cumsum([0] + n_rank[:-1]) )
            n_rank_procs = tuple( n_rank )
            mpi_type = mpi_type_dict[ str(array.dtype) ]
            sendbuf = [ array, n_rank_procs[self.rank] ]
            recvbuf = [ gathered_array, n_rank_procs, i_start_procs, mpi_type ]
            # Send/receive the arrays
            self.mpi_comm.Gatherv( sendbuf, recvbuf, root=root )
        else:
            gathered_array[:] = array[:]

        # Return the gathered_array only on process root
        if self.rank == root:
            return(gathered_array)
