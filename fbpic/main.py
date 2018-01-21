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
from fbpic.utils.mpi import MPI
# Check if threading is available
from .utils.threading import threading_enabled
# Check if CUDA is available, then import CUDA functions
from .utils.cuda import cuda_installed
if cuda_installed:
    from .utils.cuda import send_data_to_gpu, \
                receive_data_from_gpu, mpi_select_gpus
    mpi_select_gpus( MPI )

# Import the rest of the requirements
import numba
from scipy.constants import m_e, m_p, e, c
from .utils.printing import ProgressBar, print_simulation_setup
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
                 v_comoving=None, use_galilean=True, initialize_ions=False,
                 use_cuda=False, n_guard=None, n_damp=30, exchange_period=None,
                 boundaries='periodic', gamma_boost=None,
                 use_all_mpi_ranks=True, particle_shape='linear',
                 verbose_level=1 ):
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
           The order of the stencil for z derivatives in the Maxwell solver.
           Use -1 for infinite order, i.e. for exact dispersion relation in
           all direction (adviced for single-GPU/single-CPU simulation).
           Use a positive number (and multiple of 2) for a finite-order stencil
           (required for multi-GPU/multi-CPU with MPI). A large `n_order` leads
           to more overhead in MPI communications, but also to a more accurate
           dispersion relation for electromagnetic waves. (Typically,
           `n_order = 32` is a good trade-off.) See `this article
           <https://arxiv.org/abs/1611.05712>`_ for more information.

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
            a domain, when performing parallel (MPI) computation
            or when using open boundaries. Defaults to None, which
            calculates the required guard cells for n_order
            automatically (approx 2*n_order). If no MPI is used and
            in the case of open boundaries with an infinite order stencil,
            n_guard defaults to 30, if not set otherwise.
        n_damp : int, optional
            Number of damping guard cells at the left and right of a
            simulation box if a moving window is attached. The guard
            region at these areas (left / right of moving window) is
            extended by n_damp (N=n_guard+n_damp) in order to smoothly
            damp the fields such that they do not wrap around.
            (Defaults to 30)
        exchange_period: int, optional
            Number of iterations before which the particles are exchanged.
            If set to None, the maximum exchange period is calculated
            automatically: Within exchange_period timesteps, the
            particles should never be able to travel more than
            (n_guard - particle_shape order) cells. (Setting exchange_period
            to small values can substantially affect the performance)

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
            Possible values are 'cubic', 'linear'. ('cubic' corresponds to
            third order shapes and 'linear' to first order shapes).

        verbose_level: int, optional
            Print information about the simulation setup after
            initialization of the Simulation class.
            0 - Print no information
            1 (Default) - Print basic information
            2 - Print detailed information
        """
        # Check whether to use CUDA
        self.use_cuda = use_cuda
        if (self.use_cuda==True) and (cuda_installed==False):
            # Print warning if use_cuda = True but CUDA is not available
            print('*** Cuda not available for the simulation.')
            print('*** Performing the simulation on CPU.')
            self.use_cuda = False
        # CPU multi-threading
        self.use_threading = threading_enabled
        if self.use_threading:
            self.cpu_threads = numba.config.NUMBA_NUM_THREADS
        else:
            self.cpu_threads = 1

        # Register the comoving parameters
        self.v_comoving = v_comoving
        self.use_galilean = use_galilean
        if v_comoving is None:
            self.use_galilean = False

        # When running the simulation in a boosted frame, convert the arguments
        uz_m = 0.   # Mean normalized momentum of the particles
        if gamma_boost is not None:
            self.boost = BoostConverter( gamma_boost )
            zmin, zmax, dt = self.boost.copropag_length([ zmin, zmax, dt ])
            p_zmin, p_zmax = self.boost.static_length([ p_zmin, p_zmax ])
            n_e, = self.boost.static_density([ n_e ])
            uz_m, = self.boost.longitudinal_momentum([ uz_m ])
        else:
            self.boost = None

        # Initialize the boundary communicator
        self.comm = BoundaryCommunicator( Nz, zmin, zmax, Nr, rmax, Nm, dt,
            boundaries, n_order, n_guard, n_damp, exchange_period,
            use_all_mpi_ranks )
        # Modify domain region
        zmin, zmax, p_zmin, p_zmax, Nz = \
              self.comm.divide_into_domain( p_zmin, p_zmax )

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
        self.grid_shape = self.fld.interp[0].Ez.shape
        self.particle_shape = particle_shape
        self.ptcl = [
            Particles(q=-e, m=m_e, n=n_e, Npz=Npz, zmin=p_zmin,
                      zmax=p_zmax, Npr=Npr, rmin=p_rmin, rmax=p_rmax,
                      Nptheta=p_nt, dt=dt, dens_func=dens_func, uz_m=uz_m,
                      grid_shape=self.grid_shape,
                      particle_shape=self.particle_shape,
                      use_cuda=self.use_cuda ) ]
        if initialize_ions :
            self.ptcl.append(
                Particles(q=e, m=m_p, n=n_e, Npz=Npz, zmin=p_zmin,
                          zmax=p_zmax, Npr=Npr, rmin=p_rmin, rmax=p_rmax,
                          Nptheta=p_nt, dt=dt, dens_func=dens_func, uz_m=uz_m,
                          grid_shape=self.grid_shape,
                          particle_shape=self.particle_shape,
                          use_cuda=self.use_cuda ) )

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
        # Initialize an empty list of diagnostics and checkpoints
        # (Checkpoints are used for restarting the simulation)
        self.diags = []
        self.checkpoints = []
        # Initialize an empty list of laser antennas
        self.laser_antennas = []

        # Print simulation setup
        print_simulation_setup( self, verbose_level=verbose_level )

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
            Whether to use the true rho deposited on the grid for the
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
        # Sanity check
        # (This is because the guard cells of rho are never exchanged.)
        if self.comm.size > 1 and use_true_rho:
            raise ValueError('use_true_rho cannot be used in multi-proc mode.')
        if self.comm.size > 1 and correct_divE:
            raise ValueError('correct_divE cannot be used in multi-proc mode.')

        # Initialize variables to measure the time taken by the simulation
        if show_progress and self.comm.rank==0:
            progress_bar = ProgressBar( N )

        # Send simulation data to GPU (if CUDA is used)
        if self.use_cuda:
            send_data_to_gpu(self)

        # Get the E and B fields in spectral space initially
        # (In the rest of the loop, E and B will only be transformed
        # from spectal space to real space, but never the other way around)
        self.comm.exchange_fields(fld.interp, 'E', 'replace')
        self.comm.exchange_fields(fld.interp, 'B', 'replace')
        self.comm.damp_EB_open_boundary( fld.interp )
        fld.interp2spect('E')
        fld.interp2spect('B')

        # Beginning of the N iterations
        # -----------------------------

        # Loop over timesteps
        for i_step in range(N):

            # Show a progression bar and calculate ETA
            if show_progress and self.comm.rank==0:
                progress_bar.time( i_step )
                progress_bar.print_progress()

            # Particle exchanges to prepare for this iteration
            # ------------------------------------------------

            # Check whether this iteration involves particle exchange.
            # Note: Particle exchange is imposed at the first iteration
            # of this loop (i_step == 0) in order to ensure that all
            # particles are inside the box, and that 'rho_prev' is correct
            if self.iteration % self.comm.exchange_period == 0 or i_step == 0:
                # Particle exchange includes MPI exchange of particles, removal
                # of out-of-box particles and (if there is a moving window)
                # injection of new particles by the moving window.
                # (In the case of single-proc periodic simulations, particles
                # are shifted by one box length, so they remain inside the box)
                for species in self.ptcl:
                    self.comm.exchange_particles(species, fld, self.time)
                # Set again the number of cells to be injected to 0
                # (This number is incremented when `move_grids` is called)
                if self.comm.moving_win is not None:
                    self.comm.moving_win.nz_inject = 0

                # Reproject the charge on the interpolation grid
                # (Since particles have been removed / added to the simulation;
                # otherwise rho_prev is obtained from the previous iteration.
                # Note that the guard cells of rho are never exchanged.)
                self.deposit('rho_prev', exchange=False)

            # For the field diagnostics of the first step: deposit J
            # (Note however that this is not the *corrected* current)
            if i_step == 0:
                self.deposit('J', exchange=True)

            # Diagnostics
            # -----------

            # Run the diagnostics
            # (E, B, rho, x are defined at time n; J, p at time n-1/2)
            for diag in self.diags:
                # Check if the diagnostic should be written at this iteration
                # and write it, if it is the case.
                # (If needed: bring rho/J from spectral space, where they
                # were smoothed/corrected, and copy the data from the GPU.)
                diag.write( self.iteration )

            # Main PIC iteration
            # ------------------

            # Gather the fields from the grid at t = n dt
            for species in ptcl:
                species.gather( fld.interp )
            # Apply the external fields at t = n dt
            for ext_field in self.external_fields:
                ext_field.apply_expression( self.ptcl, self.time )

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
            # (Guard cell exchange done either now or after current correction)
            self.deposit('J', exchange=(correct_currents is False))

            # Handle elementary processes at t = (n + 1/2)dt
            # i.e. when the particles' velocity and position are synchronized
            # (e.g. ionization, Compton scattering, ...)
            for species in ptcl:
                species.handle_elementary_processes( self.time + 0.5*self.dt )

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
            self.deposit('rho_next', exchange=False)
            # Correct the currents (requires rho at t = (n+1) dt )
            if correct_currents:
                fld.correct_currents()
                if self.comm.size > 1:
                    # Exchange the guard cells of corrected J between domains
                    # (If correct_currents is False, the exchange of J
                    # is done in the function `deposit`)
                    fld.spect2partial_interp('J')
                    self.comm.exchange_fields(fld.interp, 'J', 'add')
                    fld.partial_interp2spect('J')
                    fld.exchanged_source['J'] = True

            # Push the fields E and B on the spectral grid to t = (n+1) dt
            fld.push( use_true_rho )
            if correct_divE:
                fld.correct_divE()
            # Move the grids if needed
            if self.comm.moving_win is not None:
                # Shift the fields is spectral space and update positions of
                # the interpolation grids
                self.comm.move_grids(fld, self.dt, self.time)

            # Get the MPI-exchanged and damped E and B field in both
            # spectral space and interpolation space
            # (Since exchange/damp operation is purely along z, spectral fields
            # are updated by doing an iFFT/FFT instead of a full transform)
            fld.spect2partial_interp('E')
            fld.spect2partial_interp('B')
            self.comm.exchange_fields(fld.interp, 'E', 'replace')
            self.comm.exchange_fields(fld.interp, 'B', 'replace')
            self.comm.damp_EB_open_boundary( fld.interp )
            fld.partial_interp2spect('E')
            fld.partial_interp2spect('B')
            # Get the corresponding fields in interpolation space
            fld.spect2interp('E')
            fld.spect2interp('B')

            # Increment the global time and iteration
            self.time += self.dt
            self.iteration += 1

            # Write the checkpoints if needed
            for checkpoint in self.checkpoints:
                checkpoint.write( self.iteration )

        # End of the N iterations
        # -----------------------

        # Finalize PIC loop
        # Get the charge density and the current from spectral space.
        fld.spect2interp('J')
        if (not fld.exchanged_source['J']) and (self.comm.size > 1):
            self.comm.exchange_fields(self.fld.interp, 'J', 'add')
        fld.spect2interp('rho_prev')
        if (not fld.exchanged_source['rho_prev']) and (self.comm.size > 1):
            self.comm.exchange_fields(self.fld.interp, 'rho', 'add')

        # Receive simulation data from GPU (if CUDA is used)
        if self.use_cuda:
            receive_data_from_gpu(self)

        # Print the measured time taken by the PIC cycle
        if show_progress and (self.comm.rank==0):
            progress_bar.print_summary()

    def deposit( self, fieldtype, exchange=False ):
        """
        Deposit the charge or the currents to the interpolation grid
        and then to the spectral grid.

        Parameters
        ----------
        fieldtype: str
            The designation of the spectral field that
            should be changed by the deposition
            Either 'rho_prev', 'rho_next' or 'J'

        exchange: bool
            Whether to exchange guard cells via MPI before transforming
            the fields to the spectral grid. (The corresponding flag in
            fld.exchanged_source is set accordingly.)
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
            # Exchange guard cells if requested by the user
            if exchange and self.comm.size > 1:
                self.comm.exchange_fields(fld.interp, 'rho', 'add')

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
            # Exchange guard cells if requested by the user
            if exchange and self.comm.size > 1:
                self.comm.exchange_fields(fld.interp, 'J', 'add')

        else:
            raise ValueError('Unknown fieldtype: %s' %fieldtype)

        # Get the charge or currents on the spectral grid
        fld.interp2spect( fieldtype )
        if self.filter_currents:
            fld.filter_spect( fieldtype )
        # Set the flag to indicate whether these fields have been exchanged
        fld.exchanged_source[ fieldtype ] = (exchange and self.comm.size > 1)

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
        # Shift the boundaries of the global domain
        self.comm.shift_global_domain_positions( shift_distance )
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
            self.dt, self.ptcl, v, self.p_nz, self.time,
            ux_m, uy_m, uz_m, ux_th, uy_th, uz_th, gamma_boost )

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
