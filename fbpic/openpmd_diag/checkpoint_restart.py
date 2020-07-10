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
from scipy.constants import e
from .field_diag import FieldDiagnostic
from .particle_diag import ParticleDiagnostic
from fbpic.utils.mpi import comm

# Check if CUDA is available, then import CUDA
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    import cupy

def set_periodic_checkpoint( sim, period, checkpoint_dir='./checkpoints' ):
    """
    Set up periodic checkpoints of the simulation

    The checkpoints are saved in openPMD format, in the specified
    directory, with one subdirectory per process.
    The E and B fields and particle information of each processor is saved.

    NB: Checkpoints are registered in the list `checkpoints` of the Simulation
    object `sim`, and written at the end of the PIC loop (whereas regular
    diagnostics are written at the beginning of the PIC loop).

    Parameters
    ----------
    sim: a Simulation object
       The simulation that is to be saved in checkpoints

    period: integer
       The number of PIC iteration between each checkpoint.

    checkpoint_dir: string, optional
        The path to the directory in which the checkpoints are stored
        (When running a simulation with several MPI ranks, use the
        same path for all ranks.)
    """
    # Only processor 0 creates a directory where checkpoints will be stored
    # Make sure that all processors wait until this directory is created
    # (Use the global MPI communicator instead of the `BoundaryCommunicator`
    # so that this still works in the case `use_all_ranks=False`)
    if comm.rank == 0:
        if os.path.exists(checkpoint_dir) is False:
            os.mkdir(checkpoint_dir)
    comm.barrier()

    # Choose the name of the directory: one directory per processor
    write_dir = os.path.join(checkpoint_dir, 'proc%d/' %comm.rank)

    # Register a periodic FieldDiagnostic in the diagnostics of the simulation
    # This saves only the E and B field (and their PML components, if used)
    fieldtypes = ["E", "B"]
    if sim.use_pml:
        fieldtypes += ["Er_pml", "Et_pml", "Br_pml", "Bt_pml"]
    sim.checkpoints.append( FieldDiagnostic( period, sim.fld,
                        fieldtypes=fieldtypes, write_dir=write_dir ) )

    # Register a periodic ParticleDiagnostic, which contains all
    # the particles which are present in the simulation
    particle_dict = {}
    for i in range(len(sim.ptcl)):
        particle_dict[ 'species %d' %i ] = sim.ptcl[i]

    if len(particle_dict)>0:
        sim.checkpoints.append(
            ParticleDiagnostic( period, particle_dict, write_dir=write_dir ) )

def restart_from_checkpoint( sim, iteration=None,
                            checkpoint_dir='./checkpoints' ):
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

    iteration: integer, optional
       The iteration number of the checkpoint from which to restart
       If None, the latest checkpoint available will be used.

    checkpoint_dir: string, optional
        The path to the directory that contains the checkpoints to be loaded.
        (When running a simulation with several MPI ranks, use the
        same path for all ranks.)
    """
    # Try to import openPMD-viewer, version 1
    try:
        from openpmd_viewer import OpenPMDTimeSeries
        openpmd_viewer_version = 1
    except ImportError:
        # If not available, try to import openPMD-viewer, version 0
        try:
            from opmd_viewer import OpenPMDTimeSeries
            openpmd_viewer_version = 0
        except ImportError:
            openpmd_viewer_version = None
    # Otherwise, raise an error
    if openpmd_viewer_version is None:
        raise ImportError(
        'The package openPMD-viewer is required to restart from checkpoints.'
        '\nPlease install it from https://github.com/openPMD/openPMD-viewer')

    # Verify that the restart is valid (only for the first processor)
    # (Use the global MPI communicator instead of the `BoundaryCommunicator`,
    # so that this also works for `use_all_ranks=False`)
    if comm.rank == 0:
        check_restart( sim, iteration, checkpoint_dir )
    comm.barrier()

    # Choose the name of the directory from which to restart:
    # one directory per processor
    data_dir = os.path.join( checkpoint_dir, 'proc%d/hdf5' %comm.rank )
    ts = OpenPMDTimeSeries( data_dir )
    # Select the iteration, and its index
    if iteration is None:
        iteration = ts.iterations[-1]
    # Find the index of the closest iteration
    i_iteration = np.argmin( abs(np.array(ts.iterations) - iteration) )

    # Modify parameters of the simulation
    sim.iteration = iteration
    sim.time = ts.t[ i_iteration ]

    # Export available species as a list
    avail_species = ts.avail_species
    if avail_species is None:
        avail_species = []

    # Load the particles
    # Loop through the different species
    if len(avail_species) == len(sim.ptcl):
        for i in range(len(sim.ptcl)):
            name = 'species %d' %i
            load_species( sim.ptcl[i], name, ts, iteration,
                            sim.comm, openpmd_viewer_version )
    else:
        raise RuntimeError( \
"""Species numbers in checkpoint and simulation should be same, but
got {:d} and {:d}. Use add_new_species method to add species to
simulation or sim.ptcl = [] to remove them""".format(len(avail_species),
                                                     len(sim.ptcl)) )
    # Record position of grid before restart
    zmin_old = sim.fld.interp[0].zmin

    # Load the fields
    # Loop through the different modes
    for m in range( sim.fld.Nm ):
        # Load the fields E and B
        for fieldtype in ['E', 'B']:
            for coord in ['r', 't', 'z']:
                load_fields( sim.fld.interp[m], fieldtype,
                             coord, ts, iteration )
        # Load the PML components if needed
        if sim.use_pml:
            for fieldtype in ['Er_pml', 'Et_pml', 'Br_pml', 'Bt_pml']:
                load_fields(sim.fld.interp[m], fieldtype, None, ts, iteration)

    # Record position after restart (`zmin` is modified by `load_fields`)
    # and shift the global domain position in the BoundaryCommunicator
    zmin_new = sim.fld.interp[0].zmin
    sim.comm.shift_global_domain_positions( zmin_new - zmin_old )

def check_restart( sim, iteration, checkpoint_dir ):
    """Verify that the restart is valid."""

    # Check that the checkpoint directory exists
    if os.path.exists(checkpoint_dir) is False:
        raise RuntimeError('The directory %s, which is '
         'required to restart a simulation, does not exist.' %checkpoint_dir)

    # Infer the number of processors that were used for the checkpoint
    # and check that it is the same as the current number of processors
    nproc = 0
    regex_matcher = re.compile('proc\d+')
    for directory in os.listdir(checkpoint_dir):
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
       Either 'E', 'B', 'J', 'rho', or 'Er_pml', 'Et_pml', etc.
       Indicates which field to load.

    coord: string or None
       Either 'r', 't' or 'z'. Indicates which field to load.

    ts: an OpenPMDTimeSeries object
       Points to the checkpoint data

    iteration: integer
       The iteration of the checkpoint to be loaded.
    """
    Nr = grid.Nr
    m = grid.m

    # Extract the field from the restart file using openpmd_viewer
    if m==0:
        field_data, info = ts.get_field( fieldtype, coord,
                                         m=m, iteration=iteration )
        # Select a half-plane and transpose it to conform to FBPIC format
        field_data = field_data[Nr:,:].T
    else:
        # Extract the real and imaginary part by selecting the angle
        field_data_real, info = ts.get_field( fieldtype, coord,
                            iteration=iteration, m=m, theta=0)
        field_data_imag, _ = ts.get_field( fieldtype, coord,
                            iteration=iteration, m=m, theta=np.pi/(2*m))
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
    dz = info.dz
    grid.zmin = info.zmin - 0.5*dz
    grid.zmax = info.zmax + 0.5*dz
    length_new = grid.zmax - grid.zmin
    assert np.allclose( length_old, length_new )

def load_species( species, name, ts, iteration, comm, openpmd_viewer_version ):
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

    openpmd_viewer_version: int
        Version of openPMD-viewer that was imported
        (needed in order to properly read the particle positions)
    """
    # Get the particles' positions (convert to meters)
    x, y, z = ts.get_particle(
                ['x', 'y', 'z'], iteration=iteration, species=name )
    if openpmd_viewer_version == 0:
        # Version 0: Convert from microns to meters
        species.x, species.y, species.z = 1.e-6*x, 1.e-6*y, 1.e-6*z
    else:
        # Version 1: Positions are directly given in meters
        species.x, species.y, species.z = x, y, z
    # Get the particles' momenta
    species.ux, species.uy, species.uz = ts.get_particle(
        ['ux', 'uy', 'uz' ], iteration=iteration, species=name )
    # Get the weight (multiply it by the charge to conform with FBPIC)
    species.w, = ts.get_particle( ['w'], iteration=iteration, species=name )
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

    # If the species is ionizable, set the proper arrays
    if species.ionizer is not None:
        # Reallocate the ionization_level, and reset it with the right value
        species.ionizer.ionization_level = np.empty( Ntot, dtype=np.uint64 )
        q, = ts.get_particle( ['charge'], iteration=iteration, species=name)
        species.ionizer.ionization_level[:] = np.uint64( np.round( q/e ) )
        # Set the auxiliary array
        species.ionizer.w_times_level = \
                    species.w * species.ionizer.ionization_level

    # Reset the injection positions (for continuous injection)
    if species.continuous_injection:
        species.injector.reset_injection_positions()

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
        # cell_idx and sorted_idx always stay on GPU
        species.cell_idx = cupy.empty( Ntot, dtype=np.int32)
        species.sorted_idx = cupy.empty( Ntot, dtype=np.intp)
        # sorting buffers are initialized on CPU
        # (because they are swapped with other particle arrays during sorting)
        species.sorting_buffer = np.empty( Ntot, dtype=np.float64)
        if hasattr( species, 'int_sorting_buffer'):
            species.int_sorting_buffer = np.empty( Ntot, dtype=np.uint64 )
        species.sorted = False
