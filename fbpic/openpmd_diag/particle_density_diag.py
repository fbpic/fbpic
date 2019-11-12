# Copyright 2018, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file defines the class ParticleDensityDiagnostic.
"""
import os
import numpy as np
from .field_diag import FieldDiagnostic

class ParticleChargeDensityDiagnostic(FieldDiagnostic):
    """
    Class that defines a diagnostic for particle density
    """

    def __init__(self, period=None, sim=None, species={},
                write_dir=None, iteration_min=0, iteration_max=np.inf,
                dt_period=None ):
        """
        Writes the charge density of the specified species in the
        openPMD file (one dataset per species)

        Parameters
        ----------
        period : int, optional
            The period of the diagnostics, in number of timesteps.
            (i.e. the diagnostics are written whenever the number
            of iterations is divisible by `period`). Specify either this or
            `dt_period`.

        dt_period : float (in seconds), optional
            The period of the diagnostics, in physical time of the simulation.
            Specify either this or `period`

        sim: an fbpic :any:`Simulation` object
            Contains the information of the simulation

        species: a dictionary of :any:`Particles` objects
            Similar to the corresponding object for `ParticleDiagnostic`
            Specifies the density of which species should be written

        write_dir : string, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory.

        iteration_min, iteration_max: ints
            The iterations between which data should be written
            (`iteration_min` is inclusive, `iteration_max` is exclusive)
        """
        # Check the arguments
        if sim is None:
            raise ValueError("You need to pass the argument `sim`.")
        if len(species) == 0:
            raise ValueError("You need to pass a valid `species` dictionary.")

        # Build the list of fieldtypes
        fieldtypes = []
        for species_name in species.keys():
            fieldtypes.append( 'rho_%s' %species_name )

        # General setup
        FieldDiagnostic.__init__(self, period, fldobject=sim.fld,
                    comm=sim.comm, fieldtypes=fieldtypes, write_dir=write_dir,
                    iteration_min=iteration_min, iteration_max=iteration_max,
                    dt_period=dt_period )

        # Register the arguments
        self.sim = sim
        self.species = species

    def write_hdf5( self, iteration ) :
        """
        Write an HDF5 file that complies with the OpenPMD standard

        Parameter
        ---------
        iteration : int
             The current iteration number of the simulation.
        """
        sim = self.sim

        # Extract information needed for the openPMD attributes
        dt = self.fld.dt
        time = iteration * dt
        dz = self.fld.interp[0].dz
        zmin, _ = self.comm.get_zmin_zmax(
                local=False, with_damp=False, with_guard=False )
        Nz, _ = self.comm.get_Nz_and_iz(
                local=False, with_damp=False, with_guard=False )
        Nr = self.comm.get_Nr(with_damp=False)

        # Create the file with these attributes
        filename = "data%08d.h5" %iteration
        fullpath = os.path.join( self.write_dir, "hdf5", filename )
        self.create_file_empty_meshes(
            fullpath, iteration, time, Nr, Nz, zmin, dz, dt )

        # Loop over the requested species
        for species_name in self.species.keys():

            # Deposit the charge density for this species ; this does not
            # affect the spectral space, and therefore it does not affect
            # the simulation
            species_object = self.species[species_name]
            # Deposit and overwrite rho_next in spectral space as it will be
            # correctly updated anyways later in this iteration by the PIC loop
            sim.deposit( 'rho_next', species_list=[species_object],
                        update_spectral=True, exchange=False )
            # Bring filtered particle density back to the intermediate grid
            self.fld.spect2interp('rho_next')
            # Exchange (add) the particle density between domains
            if (sim.comm is not None) and (sim.comm.size > 1):
                sim.comm.exchange_fields(sim.fld.interp, 'rho', 'add')

            # If needed: Receive data from the GPU
            if self.fld.use_cuda :
                self.fld.receive_fields_from_gpu()

            # Open the file again, and get the field path
            f = self.open_file( fullpath )
            # (f is None if this processor does not participate in writing data)
            if f is not None:
                field_path = "/data/%d/fields/" %iteration
                field_grp = f[field_path]
            else:
                field_grp = None

            # Loop over the different quantities that should be written
            fieldtype = "rho_%s" %species_name
            self.write_dataset( field_grp, fieldtype, "rho" )

            # Close the file (only the first proc does this)
            if f is not None:
                f.close()

            # Send data to the GPU if needed
            if self.fld.use_cuda :
                self.fld.send_fields_to_gpu()
