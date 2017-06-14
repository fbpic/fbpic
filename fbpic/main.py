# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters, Soeren Jalas
# License: 3-Clause-BSD-LBNL
"""
Fourier-Bessel Particle-In-Cell (FB-PIC) main file

This file steers and controls the simulation.
"""
# When cuda is available, select one GPU per mpi process
# (This needs to be done before the other imports,
# as it sets the cuda context)
from mpi4py import MPI
# Check if CUDA is available, then import CUDA functions
from .cuda_utils import cuda_installed
if cuda_installed:
    from .cuda_utils import send_data_to_gpu, \
                receive_data_from_gpu, mpi_select_gpus
    mpi_select_gpus( MPI )

# Import the rest of the requirements
import sys, time
from scipy.constants import m_e, m_p, e, c
from .particles import Particles
from .lpa_utils.boosted_frame import BoostConverter
from .fields import Fields
from .boundaries import BoundaryCommunicator, MovingWindow

class Simulation(object):
    """
    Top-level simulation class that contains all the simulation
    data, as well as the methods to perform the PIC cycle.

    The `Simulation` class has several important attributes:

    - `fld`, a `Fields` object which contains the field information
    - `ptcl`, a list of `Particles` objects (one per species)
    - `diags`, a list of diagnostics to be run during the simulation
    - `comm`, a `BoundaryCommunicator`, which contains the MPI decomposition
    """

    def __init__(self, Nz, zmax, Nr, rmax, Nm, dt, p_zmin, p_zmax,
                 p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e, zmin=0.,
                 n_order=-1, dens_func=None, filter_currents=True,
                 v_comoving=None, use_galilean=False,
                 initialize_ions=False, use_cuda=False,
                 n_guard=50, exchange_period=None, boundaries='periodic',
                 gamma_boost=None, use_all_mpi_ranks=True,
                 particle_shape='linear' ):
        """
        Initializes a simulation, by creating the following structures:

        - the `Fields` object, which contains the field data on the grids
        - a set of electrons
        - a set of ions (if initialize_ions is True)

        Parameters
        ----------
        Nz: int
            The number of gridpoints along z
        Nr: int
            The number of gridpoints along r

        zmax: float
            The position of the edge of the simulation in z
            (More precisely, the position of the edge of the last cell)
        rmax: float
            The position of the edge of the simulation in r
            (More precisely, the position of the edge of the last
            cell)

        Nm: int
            The number of azimuthal modes taken into account

        dt: float
            The timestep of the simulation

        p_zmin: float
            The minimal z position above which the particles are initialized
        p_zmax: float
            The maximal z position below which the particles are initialized
        p_rmin: float
            The minimal r position above which the particles are initialized
        p_rmax: float
            The maximal r position below which the particles are initialized

        p_nz: int
            The number of macroparticles per cell along the z direction
        p_nr: int
            The number of macroparticles per cell along the r direction
        p_nt: int
            The number of macroparticles along the theta direction

        n_e: float (in particles per m^3)
           Peak density of the electrons

        n_order: int, optional
           The order of the stencil for the z derivatives.
           Use -1 for infinite order, otherwise use a positive, even
           number. In this case, the stencil extends up to n_order/2
           cells on each side.

        zmin: float, optional
           The position of the edge of the simulation box.
           (More precisely, the position of the edge of the first cell)

        dens_func: callable, optional
           A function of the form:
           def dens_func( z, r ) ...
           where z and r are 1d arrays, and which returns
           a 1d array containing the density *relative to n*
           (i.e. a number between 0 and 1) at the given positions

        initialize_ions: bool, optional
           Whether to initialize the neutralizing ions
        filter_currents: bool, optional
            Whether to filter the currents and charge in k space

        v_comoving: float or None, optional
            If this variable is None, the standard PSATD is used (default).
            Otherwise, the current is assumed to be "comoving",
            i.e. constant with respect to (z - v_comoving * t).
            This can be done in two ways: either by
            - Using a PSATD scheme that takes this hypothesis into account
            - Solving the PSATD scheme in a Galilean frame
        use_galilean: bool, optional
            Determines which one of the two above schemes is used
            When use_galilean is true, the whole grid moves
            with a speed v_comoving

        use_cuda: bool, optional
            Wether to use CUDA (GPU) acceleration

        n_guard: int, optional
            Number of guard cells to use at the left and right of
            a domain, when using MPI.
        exchange_period: int, optional
            Number of iteration before which the particles are exchanged
            and the window is moved (the two operations are simultaneous)
            If set to None, the particles are exchanged every n_guard/2

        boundaries: string, optional
            Indicates how to exchange the fields at the left and right
            boundaries of the global simulation box.
            Either 'periodic' or 'open'

        gamma_boost : float, optional
            When initializing the laser in a boosted frame, set the
            value of `gamma_boost` to the corresponding Lorentz factor.
            All the other quantities (zmin, zmax, n_e, etc.) are to be given
            in the lab frame.

        use_all_mpi_ranks: bool, optional
            When launching the simulation with mpirun:

            - if `use_all_mpi_ranks` is True (default):
              All the MPI ranks will contribute to the same simulation,
              using domain-decomposition to share the work.
            - if `use_all_mpi_ranks` is False:
              Each MPI rank will run an independent simulation.
              This can be useful when running parameter scans. In this case,
              make sure that your input script is written so that the input
              parameters and output folder depend on the MPI rank.

        particle_shape: str, optional
            Set the particle shape for the charge/current deposition.
            Possible values are 'cubic', 'linear' and 'linear_non_atomic'.
            While 'cubic' corresponds to third order shapes and 'linear'
            to first order shapes, 'linear_non_atomic' uses an equivalent
            deposition scheme to 'linear' which avoids atomics on the GPU.
        """
        # Check whether to use cuda
        self.use_cuda = use_cuda
        if (use_cuda==True) and (cuda_installed==False):
            print('*** Cuda not available for the simulation.')
            print('*** Performing the simulation on CPU.')
            self.use_cuda = False

        # Register the comoving parameters
        self.v_comoving = v_comoving
        self.use_galilean = use_galilean
        if v_comoving is None:
            self.use_galilean = False

        # When running the simulation in a boosted frame, convert the arguments
        uz_m = 0.   # Mean normalized momentum of the particles
        if gamma_boost is not None:
            boost = BoostConverter( gamma_boost )
            zmin, zmax, dt = boost.copropag_length([ zmin, zmax, dt ])
            p_zmin, p_zmax = boost.static_length([ p_zmin, p_zmax ])
            n_e, = boost.static_density([ n_e ])
            uz_m, = boost.longitudinal_momentum([ uz_m ])

        # Initialize the boundary communicator
        self.comm = BoundaryCommunicator(Nz, Nr, n_guard, Nm,
            boundaries, n_order, exchange_period, use_all_mpi_ranks )
        print_simulation_setup( self.comm, self.use_cuda )
        # Modify domain region
        zmin, zmax, p_zmin, p_zmax, Nz = \
              self.comm.divide_into_domain(zmin, zmax, p_zmin, p_zmax)

        # Initialize the field structure
        self.fld = Fields( Nz, zmax, Nr, rmax, Nm, dt,
                    n_order=n_order, zmin=zmin,
                    v_comoving=v_comoving,
                    use_galilean=use_galilean,
                    use_cuda=self.use_cuda )

        # Modify the input parameters p_zmin, p_zmax, r_zmin, r_zmax, so that
        # they fall exactly on the grid, and infer the number of particles
        p_zmin, p_zmax, Npz = adapt_to_grid( self.fld.interp[0].z,
                                p_zmin, p_zmax, p_nz )
        p_rmin, p_rmax, Npr = adapt_to_grid( self.fld.interp[0].r,
                                p_rmin, p_rmax, p_nr )

        # Initialize the electrons and the ions
        grid_shape = self.fld.interp[0].Ez.shape
        self.ptcl = [
            Particles( q=-e, m=m_e, n=n_e, Npz=Npz, zmin=p_zmin,
                       zmax=p_zmax, Npr=Npr, rmin=p_rmin, rmax=p_rmax,
                       Nptheta=p_nt, dt=dt, dens_func=dens_func,
                       use_cuda=self.use_cuda, uz_m=uz_m,
                       grid_shape=grid_shape, particle_shape=particle_shape) ]
        if initialize_ions :
            self.ptcl.append(
                Particles(q=e, m=m_p, n=n_e, Npz=Npz, zmin=p_zmin,
                          zmax=p_zmax, Npr=Npr, rmin=p_rmin, rmax=p_rmax,
                          Nptheta=p_nt, dt=dt, dens_func=dens_func,
                          use_cuda=self.use_cuda, uz_m=uz_m,
                          grid_shape=grid_shape,
                          particle_shape=particle_shape ) )

        # Register the number of particles per cell along z, and dt
        # (Necessary for the moving window)
        self.dt = dt
        self.p_nz = p_nz
        # Register the time and the iteration
        self.time = 0.
        self.iteration = 0
        # Register the filtering flag
        self.filter_currents = filter_currents

        # Initialize an empty list of external fields
        self.external_fields = []
        # Initialize an empty list of diagnostics
        self.diags = []
        # Initialize an empty list of laser antennas
        self.laser_antennas = []

    def step(self, N=1, correct_currents=True,
            correct_divE=False, use_true_rho=False,
            move_positions=True, move_momenta=True, show_progress=True):
        """
        Perform N PIC cycles.

        Parameters
        ----------
        N: int, optional
            The number of timesteps to take
            Default: N=1

        correct_currents: bool, optional
            Whether to correct the currents in spectral space

        correct_divE: bool, optional
            Whether to correct the divergence of E in spectral space

        use_true_rho: bool, optional
            Wether to use the true rho deposited on the grid for the
            field push or not. (requires initialize_ions = True)

        move_positions: bool, optional
            Whether to move or freeze the particles' positions

        move_momenta: bool, optional
            Whether to move or freeze the particles' momenta

        show_progress: bool, optional
            Whether to show a progression bar
        """
        # Shortcuts
        ptcl = self.ptcl
        fld = self.fld
        # Measure the time taken by the PIC cycle
        measured_start = time.time()

        # Send simulation data to GPU (if CUDA is used)
        if self.use_cuda:
            send_data_to_gpu(self)

        # Loop over timesteps
        for i_step in range(N):

            # Messages and diagnostics
            # ------------------------

            # Show a progression bar
            if show_progress and self.comm.rank==0:
                progression_bar( i_step, N, measured_start )

            # Run the diagnostics
            for diag in self.diags:
                # Check if the fields should be written at
                # this iteration and do it if needed.
                # (Send the data to the GPU if needed.)
                diag.write( self.iteration )

            # Exchanges to prepare for this iteration
            # ---------------------------------------

            # Exchange the fields (EB) in the guard cells between domains
            self.comm.exchange_fields(fld.interp, 'EB')

            # Check whether this iteration involves
            # particle exchange / moving window
            if self.iteration % self.comm.exchange_period == 0 or i_step == 0:

                # Note: Particle exchange is imposed at the first iteration
                # of this loop (i_step == 0) in order to make sure that:
                # - All particles are inside the box initially
                # - The quantity `rho_prev` is properly defined

                # Move the grids if needed
                if self.comm.moving_win is not None:
                    # Damp the fields in the guard cells
                    self.comm.damp_guard_EB( fld.interp )
                    # Shift the fields, and prepare positions
                    # between which new particles should be added
                    self.comm.move_grids(fld, self.dt, self.time)
                    # Exchange the E and B fields via MPI if needed
                    # (Notice that the fields have not been damped since the
                    # last exchange, so fields are correct in the guard cells)
                    self.comm.exchange_fields(fld.interp, 'EB')

                # Particle exchange after moving window / mpi communications
                # This includes MPI exchange of particles, removal of
                # out-of-box particles and (if there is a moving window)
                # injection of new particles by the moving window.
                # (In the case of single-proc periodic simulations, particles
                # are shifted by one box length, so they remain inside the box.)
                for species in self.ptcl:
                    self.comm.exchange_particles(species, fld, self.time)

                # Reproject the charge on the interpolation grid
                # (Since particles have been added/suppressed)
                self.deposit('rho_prev')

            # Standard PIC loop
            # -----------------
            # Gather the fields from the grid at t = n dt
            for species in ptcl:
                species.gather( fld.interp )
            # Apply the external fields at t = n dt
            for ext_field in self.external_fields:
                ext_field.apply_expression( self.ptcl, self.time )

            # Ionize the particles at t = n dt
            # (if the species is not ionizable, `handle_ionization` skips it)
            for species in ptcl:
                species.handle_ionization()

            # Push the particles' positions and velocities to t = (n+1/2) dt
            if move_momenta:
                for species in ptcl:
                    species.push_p()
            if move_positions:
                for species in ptcl:
                    species.halfpush_x()
            # Get positions/velocities for antenna particles at t = (n+1/2) dt
            for antenna in self.laser_antennas:
                antenna.update_v( self.time + 0.5*self.dt )
                antenna.halfpush_x( self.dt )
            # Shift the boundaries of the grid for the Galilean frame
            if self.use_galilean:
                self.shift_galilean_boundaries()

            # Get the current at t = (n+1/2) dt
            self.deposit('J')

            # Push the particles' positions to t = (n+1) dt
            if move_positions:
                for species in ptcl:
                    species.halfpush_x()
            # Get positions for antenna particles at t = (n+1) dt
            for antenna in self.laser_antennas:
                antenna.halfpush_x( self.dt )
            # Shift the boundaries of the grid for the Galilean frame
            if self.use_galilean:
                self.shift_galilean_boundaries()

            # Get the charge density at t = (n+1) dt
            self.deposit('rho_next')
            # Correct the currents (requires rho at t = (n+1) dt )
            if correct_currents:
                fld.correct_currents()

            # Damp the fields in the guard cells
            self.comm.damp_guard_EB( fld.interp )
            # Get the damped fields on the spectral grid at t = n dt
            fld.interp2spect('E')
            fld.interp2spect('B')
            # Push the fields E and B on the spectral grid to t = (n+1) dt
            fld.push( use_true_rho )
            if correct_divE:
                fld.correct_divE()
            # Get the fields E and B on the interpolation grid at t = (n+1) dt
            fld.spect2interp('E')
            fld.spect2interp('B')

            # Increment the global time and iteration
            self.time += self.dt
            self.iteration += 1

        # Receive simulation data from GPU (if CUDA is used)
        if self.use_cuda:
            receive_data_from_gpu(self)

        # Print the measured time taken by the PIC cycle
        if show_progress and (self.comm.rank==0):
            measured_duration = time.time() - measured_start
            m, s = divmod(measured_duration, 60)
            h, m = divmod(m, 60)
            print('\n Time taken by the loop: %d:%02d:%02d\n' % (h, m, s))

    def deposit( self, fieldtype ):
        """
        Deposit the charge or the currents to the interpolation grid
        and then to the spectral grid.

        Parameters
        ----------
        fieldtype: str
            The designation of the spectral field that
            should be changed by the deposition
            Either 'rho_prev', 'rho_next' or 'J'
        """
        # Shortcut
        fld = self.fld

        # Deposit charge or currents on the interpolation grid

        # Charge
        if fieldtype in ['rho_prev', 'rho_next']:
            fld.erase('rho')
            # Deposit the particle charge
            for species in self.ptcl:
                species.deposit( fld, 'rho' )
            # Deposit the charge of the virtual particles in the antenna
            for antenna in self.laser_antennas:
                antenna.deposit( fld, 'rho', self.comm )
            # Divide by cell volume
            fld.divide_by_volume('rho')
            # Exchange the charge density of the guard cells between domains
            self.comm.exchange_fields(fld.interp, 'rho')

        # Currents
        elif fieldtype == 'J':
            fld.erase('J')
            # Deposit the particle current
            for species in self.ptcl:
                species.deposit( fld, 'J' )
            # Deposit the current of the virtual particles in the antenna
            for antenna in self.laser_antennas:
                antenna.deposit( fld, 'J', self.comm )
            # Divide by cell volume
            fld.divide_by_volume('J')
            # Exchange the current of the guard cells between domains
            self.comm.exchange_fields(fld.interp, 'J')
        else:
            raise ValueError('Unknown fieldtype: %s' %fieldtype)

        # Get the charge or currents on the spectral grid
        fld.interp2spect( fieldtype )
        if self.filter_currents:
            fld.filter_spect( fieldtype )

    def shift_galilean_boundaries(self):
        """
        Shift the interpolation grids by v_comoving over
        a half-timestep. (The arrays of values are unchanged,
        only position attributes are changed.)

        With the Galilean frame, in principle everything should
        be solved in variables xi = z - v_comoving t, and -v_comoving
        should be added to the motion of the particles. However, it
        is equivalent to, instead, shift the boundaries of the grid.
        """
        # Calculate shift distance over a half timestep
        shift_distance = self.v_comoving * 0.5 * self.dt
        # Shift the boundaries of the grid
        for m in range(self.fld.Nm):
            self.fld.interp[m].zmin += shift_distance
            self.fld.interp[m].zmax += shift_distance
            self.fld.interp[m].z += shift_distance

    def set_moving_window( self, v=c, ux_m=0., uy_m=0., uz_m=0.,
                  ux_th=0., uy_th=0., uz_th=0., gamma_boost=None ):
        """
        Initializes a moving window for the simulation.

        Parameters
        ----------
        v: float (in meters per seconds), optional
            The speed of the moving window

        ux_m: float (dimensionless), optional
           Normalized mean momenta of the injected particles along x
        uy_m: float (dimensionless), optional
           Normalized mean momenta of the injected particles along y
        uz_m: float (dimensionless), optional
           Normalized mean momenta of the injected particles along z

        ux_th: float (dimensionless), optional
           Normalized thermal momenta of the injected particles along x
        uy_th: float (dimensionless), optional
           Normalized thermal momenta of the injected particles along y
        uz_th: float (dimensionless), optional
           Normalized thermal momenta of the injected particles along z

        gamma_boost : float, optional
            When initializing a moving window in a boosted frame, set the
            value of `gamma_boost` to the corresponding Lorentz factor.
            Quantities like uz_m of the injected particles will be
            automatically Lorentz-transformed.
            (uz_m is to be given in the lab frame ; for the moment, this
            will not work if any of ux_th, uy_th, uz_th, ux_m, uy_m is nonzero)
        """
        # Attach the moving window to the boundary communicator
        self.comm.moving_win = MovingWindow( self.fld.interp, self.comm,
            self.ptcl, v, self.p_nz, self.time, ux_m, uy_m, uz_m,
            ux_th, uy_th, uz_th, gamma_boost )

def progression_bar( i, Ntot, measured_start, Nbars=50, char='-'):
    """
    Shows a progression bar with Nbars and the remaining
    simulation time.
    """
    nbars = int( (i+1)*1./Ntot*Nbars )
    sys.stdout.write('\r[' + nbars*char )
    sys.stdout.write((Nbars-nbars)*' ' + ']')
    sys.stdout.write(' %d/%d' %(i,Ntot))
    # Estimated time in seconds until it will finish (linear interpolation)
    eta = (((float(Ntot)/(i+1.))-1.)*(time.time()-measured_start))
    # Conversion to H:M:S
    m, s = divmod(eta, 60)
    h, m = divmod(m, 60)
    sys.stdout.write(', %d:%02d:%02d left' % (h, m, s))
    sys.stdout.flush()

def print_simulation_setup( comm, use_cuda ):
    """
    Print message about the number of proc and
    whether it is using GPU or CPU.

    Parameters
    ----------
    comm: an fbpic BoundaryCommunicator object
        Contains the information on the MPI decomposition

    use_cuda: bool
        Whether the simulation is set up to use CUDA
    """
    if comm.rank == 0:
        if use_cuda:
            message = "\nRunning FBPIC on GPU "
        else:
            message = "\nRunning FBPIC on CPU "
        message += "with %d proc.\n" %comm.size
        print( message )

def adapt_to_grid( x, p_xmin, p_xmax, p_nx, ncells_empty=0 ):
    """
    Adapt p_xmin and p_xmax, so that they fall exactly on the grid x
    Return the total number of particles, assuming p_nx particles
    per gridpoint

    Parameters
    ----------
    x: 1darray
        The positions of the gridpoints along the x direction

    p_xmin, p_xmax: float
        The minimal and maximal position of the particles
        These may not fall exactly on the grid

    p_nx: int
        Number of particle per gridpoint

    ncells_empty: int
        Number of empty cells at the righthand side of the box
        (Typically used when using a moving window)

    Returns
    -------
    A tuple with:
       - p_xmin: a float that falls exactly on the grid
       - p_xmax: a float that falls exactly on the grid
       - Npx: the total number of particles
    """

    # Find the max and the step of the array
    xmin = x.min()
    xmax = x.max()
    dx = x[1] - x[0]

    # Do not load particles below the lower bound of the box
    if p_xmin < xmin - 0.5*dx:
        p_xmin = xmin - 0.5*dx
    # Do not load particles in the two last upper cells
    # (This is because the charge density may extend over these cells
    # when it is smoothed. If particles are loaded closer to the right
    # boundary, this extended charge density can wrap around and appear
    # at the left boundary.)
    if p_xmax > xmax + (0.5-ncells_empty)*dx:
        p_xmax = xmax + (0.5-ncells_empty)*dx

    # Find the gridpoints on which the particles should be loaded
    x_load = x[ ( x > p_xmin ) & ( x < p_xmax ) ]
    # Deduce the total number of particles
    Npx = len(x_load) * p_nx
    # Reajust p_xmin and p_xmanx so that they match the grid
    if Npx > 0:
        p_xmin = x_load.min() - 0.5*dx
        p_xmax = x_load.max() + 0.5*dx

    return( p_xmin, p_xmax, Npx )
