"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)

It defines functions that can save checkpoints,
as well as reload a simulation from a set of checkpoints.
"""
import os
import numpy as np
from .field_diag import FieldDiagnostic
from .particle_diag import ParticleDiagnostic

def set_periodic_checkpoint( sim, period ):
    """
    Write one file per proc, in openPMD format

    ### DOC

    """
    # Only processor 0 creates a directory where checkpoints will be stored
    # Make sure that all processors wait until this directory is created
    if sim.comm.rank == 0:
        if os.path.exists('./checkpoints') is False:
            os.mkdir('./checkpoints')
    sim.comm.mpi_comm.Barrier()
    
    # Choose the name of the directory: one directory per processor
    write_dir = 'checkpoints/proc%d/' %sim.comm.rank

    ### Same number of proc
    
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
    ### DOC

    Should be called before setting the moving window
    """
    # Import openPMD-viewer
    try:
        from opmd_viewer import OpenPMDTimeSeries
    except ImportError:
        raise ImportError(
        'The package `opmd_viewer` is required to restart from checkpoints.'
        '\nPlease install it from https://github.com/openPMD/openPMD-viewer')

    # Choose the name of the directory from which to restart:
    # one directory per processor
    checkpoint_dir = 'checkpoints/proc%d/hdf5' %sim.comm.rank
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
    for i in xrange(len(sim.ptcl)):
        name = 'species %d' %i        
        load_species( sim.ptcl[i], name, ts, iteration)

    # Load the fields
    # Loop through the different modes
    for m in range( sim.fld.Nm ):
        # Load the fields E and B
        for fieldtype in ['E', 'B']:
            for coord in ['r', 't', 'z']:
                load_fields( sim.fld.interp[m], fieldtype,
                             coord, ts, iteration )
        # Load the charge density (to prepare charge conservation)
        load_fields( sim.fld.interp[m], 'rho', None, ts, iteration )
        # Loading J is not necessary

    # Convert 'rho' to spectral space ('rho_prev')
    # to prepare charge conservation for the first timestep
    sim.fld.interp2spect( 'rho_prev' )
    if sim.filter_currents:
        sim.fld.filter_spect( 'rho_prev' )
    

def load_fields( grid, fieldtype, coord, ts, iteration ):
    """
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

def load_species( species, name, ts, iteration ):
    """
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
