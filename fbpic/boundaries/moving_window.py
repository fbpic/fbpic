# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to implement the moving window.
"""
from fbpic.utils.threading import njit_parallel, prange
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    from fbpic.utils.cuda import cuda, cuda_tpb_bpg_2d, compile_cupy

class MovingWindow(object):
    """
    Class that contains the moving window's variables and methods
    """
    def __init__( self, comm, dt, v, time ):
        """
        Initializes a moving window object.

        Parameters
        ----------
        comm: a BoundaryCommunicator object
            Contains information about the MPI decomposition
            and about the longitudinal boundaries

        dt: float
            The timestep of the simulation.

        v: float (meters per seconds), optional
            The speed of the moving window

        time: float (seconds)
            The time (in the simulation) at which the moving
            window was initialized
        """
        # Check that the boundaries are open
        if ((comm.rank == comm.size-1) and (comm.right_proc is not None)) \
          or ((comm.rank == 0) and (comm.left_proc is not None)):
          raise ValueError('The simulation is using a moving window, but '
                    'the boundaries are periodic.\n Please select open '
                    'boundaries when initializing the Simulation object.')

        # Attach moving window speed
        self.v = v
        # Attach time of last move
        self.t_last_move = time - dt

        # Get the positions of the global physical domain
        zmin_global_domain, zmax_global_domain = comm.get_zmin_zmax(
                            local=False, with_damp=False, with_guard=False )

        # Attach reference position of moving window (only for the first proc)
        # (Determines by how many cells the window should be moved)
        if comm.rank == 0:
            self.zmin = zmin_global_domain


    def move_grids(self, fld, ptcl, comm, time):
        """
        Calculate by how many cells the moving window should be moved.
        If this is non-zero, shift the fields on the interpolation grid,
        and increment the positions between which the continuously-injected
        particles will be generated.

        Parameters
        ----------
        fld: a Fields object
            Contains the fields data of the simulation

        ptcl: a list of Particles object
            This is passed in order to increment the positions between
            which the continuously-injection particles will be generated

        comm: an fbpic BoundaryCommunicator object
            Contains the information on the MPI decomposition

        time: float (seconds)
            The global time in the simulation
            This is used in order to determine how much the window should move
        """
        # To avoid discrepancies between processors, only the first proc
        # decides whether to send the data, and broadcasts the information.
        dz = comm.dz
        if comm.rank==0:
            # Move the continuous position of the moving window object
            self.zmin += self.v * (time - self.t_last_move)
            # Find the number of cells by which the window should move
            zmin_global_domain, zmax_global_domain = comm.get_zmin_zmax(
                            local=False, with_damp=False, with_guard=False )
            n_move = int( (self.zmin - zmin_global_domain)/dz )
        else:
            n_move = None
        # Broadcast the information to all proc
        if comm.size > 1:
            n_move = comm.mpi_comm.bcast( n_move )

        # Move the grids
        if n_move != 0:
            # Move the global domain
            comm.shift_global_domain_positions( n_move*dz )
            # Shift the fields
            Nm = len(fld.interp)
            for m in range(Nm):
                # Modify the values of the corresponding z's
                fld.interp[m].zmin += n_move*fld.interp[m].dz
                fld.interp[m].zmax += n_move*fld.interp[m].dz
                # Shift/move fields by n_move cells in spectral space
                self.shift_spect_grid( fld.spect[m], n_move )

        # Because the grids have just been shifted, there is a shift
        # in the cell indices that are used for the prefix sum.
        for species in ptcl:
            if species.use_cuda:
                species.prefix_sum_shift += n_move
                # This quantity is reset to 0 whenever prefix_sum is recalculated

        # Prepare the positions of injection for the particles
        # (The actual creation of particles is done when the routine
        # exchange_particles of boundary_communicator.py is called)
        if comm.rank == comm.size-1:
            for species in ptcl:
                if species.continuous_injection:
                    # Increment the positions for the generation of particles
                    # (Particles are generated when `generate_particles` is called)
                    species.injector.increment_injection_positions(
                            self.v, time-self.t_last_move )

        # Change the time of the last move
        self.t_last_move = time


    def shift_spect_grid( self, grid, n_move,
                          shift_rho=True, shift_currents=True ):
        """
        Shift the spectral fields by n_move cells (with respect to the
        spatial grid). Shifting is done either on the CPU or the GPU,
        if use_cuda is True. (Typically n_move is positive, and the
        fields are shifted backwards)

        Parameters
        ----------
        grid: an SpectralGrid corresponding to one given azimuthal mode
            Contains the values of the fields in spectral space,
            and is modified by this function.

        n_move: int
            The number of cells by which the grid should be shifted

        shift_rho: bool, optional
            Whether to also shift the charge density
            Default: True, since rho is only recalculated from
            scratch when the particles are exchanged

        shift_currents: bool, optional
            Whether to also shift the currents
            Default: False, since the currents are recalculated from
            scratch at each PIC cycle
        """
        if grid.use_cuda:
            shift = grid.d_field_shift
            # Get a 2D CUDA grid of the size of the grid
            tpb, bpg = cuda_tpb_bpg_2d( grid.Ep.shape[0], grid.Ep.shape[1] )
            # Shift all the fields on the GPU
            shift_spect_array_gpu[tpb, bpg]( grid.Ep, shift, n_move )
            shift_spect_array_gpu[tpb, bpg]( grid.Em, shift, n_move )
            shift_spect_array_gpu[tpb, bpg]( grid.Ez, shift, n_move )
            shift_spect_array_gpu[tpb, bpg]( grid.Bp, shift, n_move )
            shift_spect_array_gpu[tpb, bpg]( grid.Bm, shift, n_move )
            shift_spect_array_gpu[tpb, bpg]( grid.Bz, shift, n_move )
            if grid.use_pml:
                shift_spect_array_gpu[tpb, bpg]( grid.Ep_pml, shift, n_move )
                shift_spect_array_gpu[tpb, bpg]( grid.Em_pml, shift, n_move )
                shift_spect_array_gpu[tpb, bpg]( grid.Bp_pml, shift, n_move )
                shift_spect_array_gpu[tpb, bpg]( grid.Bm_pml, shift, n_move )
            if shift_rho:
                shift_spect_array_gpu[tpb, bpg]( grid.rho_prev, shift, n_move )
            if shift_currents:
                shift_spect_array_gpu[tpb, bpg]( grid.Jp, shift, n_move )
                shift_spect_array_gpu[tpb, bpg]( grid.Jm, shift, n_move )
                shift_spect_array_gpu[tpb, bpg]( grid.Jz, shift, n_move )
        else:
            shift = grid.field_shift
            # Shift all the fields on the CPU
            shift_spect_array_cpu( grid.Ep, shift, n_move )
            shift_spect_array_cpu( grid.Em, shift, n_move )
            shift_spect_array_cpu( grid.Ez, shift, n_move )
            shift_spect_array_cpu( grid.Bp, shift, n_move )
            shift_spect_array_cpu( grid.Bm, shift, n_move )
            shift_spect_array_cpu( grid.Bz, shift, n_move )
            if grid.use_pml:
                shift_spect_array_cpu( grid.Ep_pml, shift, n_move )
                shift_spect_array_cpu( grid.Em_pml, shift, n_move )
                shift_spect_array_cpu( grid.Bp_pml, shift, n_move )
                shift_spect_array_cpu( grid.Bm_pml, shift, n_move )
            if shift_rho:
                shift_spect_array_cpu( grid.rho_prev, shift, n_move )
            if shift_currents:
                shift_spect_array_cpu( grid.Jp, shift, n_move )
                shift_spect_array_cpu( grid.Jm, shift, n_move )
                shift_spect_array_cpu( grid.Jz, shift, n_move )

@njit_parallel
def shift_spect_array_cpu( field_array, shift_factor, n_move ):
    """
    Shift the field 'field_array' by n_move cells on CPU.
    This is done in spectral space and corresponds to multiplying the
    fields with the factor exp(i*kz_true*dz)**n_move .

    Parameters
    ----------
    field_array: 2darray of complexs
        Contains the value of the fields, and is modified by
        this function

    shift_factor: 1darray of complexs
        Contains the shift array, that is multiplied to the fields in
        spectral space to shift them by one cell in spatial space
        ( exp(i*kz_true*dz) )

    n_move: int
        The number of cells by which the grid should be shifted
    """
    Nz, Nr = field_array.shape

    # Loop over the 2D array (in parallel over z if threading is enabled)
    for iz in prange( Nz ):
        power_shift = 1. + 0.j
        # Calculate the shift factor (raising to the power n_move ;
        # for negative n_move, we take the complex conjugate, since
        # shift_factor is of the form e^{i k dz})
        for i in range( abs(n_move) ):
            power_shift *= shift_factor[iz]
        if n_move < 0:
            power_shift = power_shift.conjugate()
        # Shift the fields
        for ir in range( Nr ):
            field_array[iz, ir] *= power_shift

if cuda_installed:

    @compile_cupy
    def shift_spect_array_gpu( field_array, shift_factor, n_move ):
        """
        Shift the field 'field_array' by n_move cells on the GPU.
        This is done in spectral space and corresponds to multiplying the
        fields with the factor exp(i*kz_true*dz)**n_move .

        Parameters
        ----------
        field_array: 2darray of complexs
            Contains the value of the fields, and is modified by
            this function

        shift_factor: 1darray of complexs
            Contains the shift array, that is multiplied to the fields in
            spectral space to shift them by one cell in spatial space
            ( exp(i*kz_true*dz) )

        n_move: int
            The number of cells by which the grid should be shifted
        """
        # Get a 2D CUDA grid
        iz, ir = cuda.grid(2)

        # Only access values that are actually in the array
        if ir < field_array.shape[1] and iz < field_array.shape[0]:
            power_shift = 1. + 0.j
            # Calculate the shift factor (raising to the power n_move ;
            # for negative n_move, we take the complex conjugate, since
            # shift_factor is of the form e^{i k dz})
            for i in range( abs(n_move) ):
                power_shift *= shift_factor[iz]
            if n_move < 0:
                power_shift = power_shift.conjugate()
            # Shift fields
            field_array[iz, ir] *= power_shift
