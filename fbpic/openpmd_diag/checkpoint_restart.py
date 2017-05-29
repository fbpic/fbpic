# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)

It defines functions that can save checkpoints,
as well as reload a simulation from a set of checkpoints.
"""
import os, re
import numpy as np
from .field_diag import FieldDiagnostic
from .particle_diag import ParticleDiagnostic
from mpi4py.MPI import COMM_WORLD as comm

def set_periodic_checkpoint( sim, period ):
    """
    Set up periodic checkpoints of the simulation

    The checkpoints are saved in openPMD format, in the directory
    `./checkpoints`, with one subdirectory per process.
    All the field and particle information of each processor is saved.

    NB: Checkpoints are registered among the list of diagnostics
    `diags` of the Simulation object `sim`.

    Parameters
    ----------
    sim: a Simulation object
       The simulation that is to be saved in checkpoints

    period: integer
       The number of PIC iteration between each checkpoint.
    """
    # Only processor 0 creates a directory where checkpoints will be stored
    # Make sure that all processors wait until this directory is created
    # (Use the global MPI communicator instead of the `BoundaryCommunicator`
    # so that this still works in the case `use_all_ranks=False`)
    if comm.rank == 0:
        if os.path.exists('./checkpoints') is False:
            os.mkdir('./checkpoints')
    comm.Barrier()

    # Choose the name of the directory: one directory per processor
    write_dir = 'checkpoints/proc%d/' %comm.rank

    # Register a periodic FieldDiagnostic in the diagnostics of the simulation
    sim.diags.append(
        FieldDiagnostic( period, sim.fld, write_dir=write_dir) )

    # Register a periodic ParticleDiagnostic, which contains all
    # the particles which are present in the simulation
    particle_dict = {}
    for i in range(len(sim.ptcl)):
        particle_dict[ 'species %d' %i ] = sim.ptcl[i]
    sim.diags.append(
        ParticleDiagnostic( period, particle_dict, write_dir=write_dir ) )

def restart_from_checkpoint( sim, iteration=None ):
    """
    Fills the Simulation object `sim` with data saved in a checkpoint.

    More precisely, the following data from `sim` is overwritten:

    - Current time and iteration number of the simulation
    - Position of the boundaries of the simulation box
    - Values of the field arrays
    - Size and values of the particle arrays

    Any other information (e.g. diagnostics of the simulation, presence of a
    moving window, presence of a laser antenna, etc.) need to be set by hand.

    For this reason, a successful restart will often require to modify the
    original input script that produced the checkpoint, rather than to start
    a new input script from scratch.

    NB: This function should always be called *before* the initialization
    of the moving window, since the moving window infers the position of
    particle injection from the existing particle data.

    Parameters
    ----------
    sim: a Simulation object
       The Simulation object into which the checkpoint should be loaded

    iteration: integer (optional)
       The iteration number of the checkpoint from which to restart
       If None, the latest checkpoint available will be used.
    """
    # Import openPMD-viewer
    try:
        from opmd_viewer import OpenPMDTimeSeries
    except ImportError:
        raise ImportError(
        'The package `opmd_viewer` is required to restart from checkpoints.'
        '\nPlease install it from https://github.com/openPMD/openPMD-viewer')

    # Verify that the restart is valid (only for the first processor)
    # (Use the global MPI communicator instead of the `BoundaryCommunicator`,
    # so that this also works for `use_all_ranks=False`)
    if comm.rank == 0:
        check_restart( sim, iteration )
    comm.Barrier()

    # Choose the name of the directory from which to restart:
    # one directory per processor
    checkpoint_dir = 'checkpoints/proc%d/hdf5' %comm.rank
    ts = OpenPMDTimeSeries( checkpoint_dir )
    # Select the iteration, and its index
    if iteration is None:
        iteration = ts.iterations[-1]
    i_iteration = ts.iterations.index( iteration )

    # Modify parameters of the simulation
    sim.iteration = iteration
    sim.time = ts.t[ i_iteration ]

    # Load the particles
    # Loop through the different species
    for i in range(len(sim.ptcl)):
        name = 'species %d' %i
        load_species( sim.ptcl[i], name, ts, iteration, sim.comm )

    # Load the fields
    # Loop through the different modes
    for m in range( sim.fld.Nm ):
        # Load the fields E and B
        for fieldtype in ['E', 'B', 'J']:
            for coord in ['r', 't', 'z']:
                load_fields( sim.fld.interp[m], fieldtype,
                             coord, ts, iteration )

def check_restart( sim, iteration ):
    """Verify that the restart is valid."""

    # Check that the checkpoint directory exists
    if os.path.exists('./checkpoints') is False:
        raise RuntimeError('The directory ./checkpoints, which is '
         'required to restart a simulation, does not exist.')

    # Infer the number of processors that were used for the checkpoint
    # and check that it is the same as the current number of processors
    nproc = 0
    regex_matcher = re.compile('proc\d+')
    for directory in os.listdir('./checkpoints'):
        if regex_matcher.match(directory) is not None:
            nproc += 1
    if nproc != comm.size:
        raise RuntimeError('For a valid restart, the current simulation '
        'should use %d MPI processes.' %nproc)

    # Check that the moving window was not yet initialized
    if sim.comm.moving_win is not None:
        raise RuntimeError('The moving window has already been initialized.\n'
        'For valid restart, the moving window should be initialized *after*\n'
        'calling `restart_from_checkpoint`.')


def load_fields( grid, fieldtype, coord, ts, iteration ):
    """
    Load the field information from the checkpoint `ts` into
    the InterpolationGrid `grid`.

    Parameters
    ----------
    grid: an InterpolationGrid object
       The object into which data should be loaded

    fieldtype: string
       Either 'E', 'B', 'J' or 'rho'. Indicates which field to load.

    coord: string
       Either 'r', 't' or 'z'. Indicates which field to load.

    ts: an OpenPMDTimeSeries object
       Points to the checkpoint data

    iteration: integer
       The iteration of the checkpoint to be loaded.
    """
    Nr = grid.Nr
    m = grid.m

    # Extract the field from the restart file using opmd_viewer
    if m==0:
        field_data, info = ts.get_field( fieldtype, coord,
                                         m=m, iteration=iteration )
        # Select a half-plane and transpose it to conform to FBPIC format
        field_data = field_data[Nr:,:].T
    elif m==1:
        # Extract the real and imaginary part by selecting the angle
        field_data_real, info = ts.get_field( fieldtype, coord,
                            iteration=iteration, m=m, theta=0)
        field_data_imag, _ = ts.get_field( fieldtype, coord,
                            iteration=iteration, m=m, theta=np.pi/2)
        # Select a half-plane and transpose it to conform to FBPIC format
        field_data_real = field_data_real[Nr:,:].T
        field_data_imag = field_data_imag[Nr:,:].T
        # Add the complex and imaginary part to create a complex field
        field_data = field_data_real + 1.j*field_data_imag
        # For the mode 1, there is an additional factor 0.5 to conform
        # to FBPIC's field representation (see field_diag.py)
        field_data *= 0.5

    # Affect the extracted field to the simulation
    if coord is not None:
        field_name = fieldtype+coord
    else:
        field_name = fieldtype
    # Perform a copy from field_data to the field in the simulation
    field = getattr( grid, field_name )
    field[:,:] = field_data[:,:]

    # Get the new positions of the bounds of the simulation
    # (and check that the box keeps the same length)
    length_old = grid.zmax - grid.zmin
    dz = length_old/grid.Nz
    grid.zmin = info.zmin
    grid.zmax = info.zmax+dz
    length_new = grid.zmax - grid.zmin
    assert np.allclose( length_old, length_new )

def load_species( species, name, ts, iteration, comm ):
    """
    Read the species data from the checkpoint `ts`
    and load it into the Species object `species`

    Parameters:
    -----------
    species: a Species object
        The object into which data is loaded

    name: string
        The name of the corresponding species in the checkpoint

    ts: an OpenPMDTimeSeries object
        Points to the data in the checkpoint

    iteration: integer
        The iteration at which to load the checkpoint

    comm: an fbpic.BoundaryCommunicator object
        Contains information about the number of procs
    """
    # Get the particles' positions (convert to meters)
    x, y, z = ts.get_particle(
                ['x', 'y', 'z'], iteration=iteration, species=name )
    species.x, species.y, species.z = 1.e-6*x, 1.e-6*y, 1.e-6*z
    # Get the particles' momenta
    species.ux, species.uy, species.uz = ts.get_particle(
        ['ux', 'uy', 'uz' ], iteration=iteration, species=name )
    # Get the weight (multiply it by the charge to conform with FBPIC)
    w, = ts.get_particle( ['w'], iteration=iteration, species=name )
    species.w = species.q * w
    # Get the inverse gamma
    species.inv_gamma = 1./np.sqrt(
        1 + species.ux**2 + species.uy**2 + species.uz**2 )
    # Take into account the fact that the arrays are resized
    Ntot = len(species.w)
    species.Ntot = Ntot

    # Check if the particles where tracked
    if "id" in ts.avail_record_components[name]:
        pid, = ts.get_particle( ['id'], iteration=iteration, species=name )
        species.track( comm )
        species.tracker.overwrite_ids( pid, comm )

    # As a safe-guard, check that the loaded data is in float64
    for attr in ['x', 'y', 'z', 'ux', 'uy', 'uz', 'w', 'inv_gamma' ]:
        assert getattr( species, attr ).dtype == np.float64

    # Field arrays
    species.Ez = np.zeros( Ntot )
    species.Ex = np.zeros( Ntot )
    species.Ey = np.zeros( Ntot )
    species.Bz = np.zeros( Ntot )
    species.Bx = np.zeros( Ntot )
    species.By = np.zeros( Ntot )
    # Sorting arrays
    if species.use_cuda:
        species.cell_idx = np.empty( Ntot, dtype=np.int32)
        species.sorted_idx = np.arange( Ntot, dtype=np.uint32)
        species.sorting_buffer = np.arange( Ntot, dtype=np.float64)
        species.sorted = False
