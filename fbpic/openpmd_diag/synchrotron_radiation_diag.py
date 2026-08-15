# Copyright 2023, FBPIC contributors
# Authors: Igor A Andriyash, Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file defines the class SRDiagnostic.
"""
import os
import numpy as np
from scipy.constants import hbar
from .generic_diag import OpenPMDDiagnostic
from fbpic.utils.mpi import comm as comm_simple

class SynchrotronRadiationDiagnostic(OpenPMDDiagnostic):
    """
    Class that defines the synchrotron radiation diagnostics to be performed.
    """

    def __init__(self, period=None, dt_period=None, species={}, comm=None,
                 write_dir=None,iteration_min=0, iteration_max=np.inf ):
        """
        Initialize the synchrotron radiation diagnostic

        Parameters
        ----------
        period : int, optional
            The period of the diagnostics, in number of timesteps.
            (i.e. the diagnostics are written whenever the number
            of iterations is divisible by `period`). Specify either this or
            `dt_period`

        dt_period : float (in seconds), optional
            The period of the diagnostics, in physical time of the simulation.
            Specify either this or `period`

        species: a dictionary of :any:`Particles` objects
            The object that is written (e.g. elec)
            is assigned to the particle name of this species.
            (e.g. {"electrons": elec }). All species must have synchrotron
            radiation activated with the same specral-angular grids.

        comm : an fbpic BoundaryCommunicator object or None
            If this is not None, the data is gathered on the first proc,
            and the guard cells are removed from the output.
            Otherwise, each proc writes its own data, including guard cells
            (Make sure to use different write_dir in this case)

        write_dir : string, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory

        iteration_min, iteration_max: ints, optional
            The iterations between which data should be written
            (`iteration_min` is inclusive, `iteration_max` is exclusive)
        """
        # Check input
        if len(species) == 0:
            raise ValueError(
            "`SRDiagnostic` requires the dictionary with the species.")

        # Register the arguments
        self.species = species
        self.species_names = list( species.keys() )

        for species_name in self.species_names:
            if species[ species_name ].synchrotron_radiator is None:
                raise ValueError(
                    f"{species_name} must have synchrotron radiation active")

        sr_object = species[ self.species_names[0] ].synchrotron_radiator

        self.use_cuda = sr_object.use_cuda
        self.dt_sim = sr_object.dt
        self.mesh_shape = (
            sr_object.N_theta_x, sr_object.N_theta_y, sr_object.N_omega
        )

        self.mesh_spacing = np.array([
            sr_object.d_theta_x, sr_object.d_theta_y,
            sr_object.d_omega * hbar ]
        )

        self.mesh_origin = np.array([
            sr_object.theta_x_min, sr_object.theta_x_min,
            sr_object.omega_min * hbar
        ])

        # General setup
        OpenPMDDiagnostic.__init__(self, period, comm, write_dir,
                            iteration_min, iteration_max,
                            dt_period=dt_period, dt_sim=self.dt_sim )


    def write_hdf5( self, iteration ):
        """
        Write an HDF5 file that complies with the OpenPMD standard

        Parameter
        ---------
        iteration : int
             The current iteration number of the simulation.
        """

        # If needed: Receive data from the GPU
        if self.use_cuda :
            for specie_name in self.species_names:
                self.species[specie_name].synchrotron_radiator\
                    .receive_from_gpu()

        # Extract information needed for the openPMD attributes
        time = iteration * self.dt_sim

        # Create the file with these attributes
        filename = "data%08d.h5" %iteration
        fullpath = os.path.join( self.write_dir, "hdf5", filename )
        self.create_file_empty_meshes(
            fullpath, iteration, time )

        # Open the file again, and get the field path
        f = self.open_file( fullpath )
        # (f is None if this processor does not participate in writing data)
        if f is not None:
            field_path = "/data/%d/fields/" %iteration
            field_grp = f[field_path]
        else:
            field_grp = None

        self.write_dataset( field_grp, "radiation" )

        # Close the file (only the first proc does this)
        if f is not None:
            f.close()

        # Send data to the GPU if needed
        if self.use_cuda :
            for specie_name in self.species_names:
                self.species[specie_name].synchrotron_radiator.send_to_gpu()

    # Writing methods
    # ---------------
    def write_dataset( self, field_grp, path) :
        """
        Write a given dataset

        Parameters
        ----------
        field_grp : an h5py.Group object
            The group that corresponds to the path indicated in meshesPath

        path : string
            The relative path where to write the dataset, in field_grp
        """

        for specie_name in self.species_names:
            path_specie = path + '_' + specie_name
            data_array = self.get_dataset( self.species[specie_name] )
            if field_grp is not None:
                dset = field_grp[path_specie]
                dset[:] =  data_array
            else:
                dset = None

    def get_dataset( self, specie ):
        """
        Copy and gather radation data on the first proc, in MPI mode
        """
        # Get the data on each individual proc
        data_one_proc = specie.synchrotron_radiator.radiation_data.copy()

        # Gather the data
        if self.comm.size>1:
            data_all_proc = self.mpi_reduce_radiation( data_one_proc )
        else:
            data_all_proc = data_one_proc

        return( data_all_proc )

    def mpi_reduce_radiation(self, data):
        """
        MPI operation to gather the radiation data
        """
        sendbuf = data
        if self.rank == 0:
            recvbuf = np.empty_like(data)
        else:
            recvbuf = None

        comm_simple.Reduce(sendbuf, recvbuf, root=0)
        return recvbuf

    # OpenPMD setup methods
    # ---------------------

    def create_file_empty_meshes( self, fullpath, iteration, time ):
        """
        Create an openPMD file with empty meshes and setup all its attributes

        Parameters
        ----------
        fullpath: string
            The absolute path to the file to be created

        iteration: int
            The iteration number of this diagnostic

        time: float (seconds)
            The physical time at this iteration
        """
        # Create the file
        f = self.open_file( fullpath )

        # Setup the different layers of the openPMD file
        # (f is None if this processor does not participate is writing data)
        if f is not None:

            # Setup the attributes of the top level of the file
            self.setup_openpmd_file( f, iteration, time, self.dt_sim )

            # Setup the meshes group (contains all the fields)
            field_path = "/data/%d/fields/" %iteration
            field_grp = f.require_group(field_path)

            for specie_name in self.species_names:
                dset = field_grp.require_dataset(
                    f"radiation_{specie_name}", self.mesh_shape, dtype='f8')
                # Setup the record and the component to which it belongs
                self.setup_openpmd_mesh_component_record( dset, "radiation" )
            # Close the file
            f.close()

    def setup_openpmd_mesh_component_record( self, dset, quantity ) :
        """
        Sets the attributes that are specific to a mesh record

        Parameter
        ---------
        dset : an h5py.Dataset or h5py.Group object

        quantity : string
           The name of the record (e.g. "radiation")
        """
        # Generic record attributes
        self.setup_openpmd_record( dset, quantity )

        # Geometry parameters
        dset.attrs['geometry'] = np.bytes_("cartesian")
        dset.attrs['axisLabels'] = np.array([ b'x', b'y', b'z' ])
        dset.attrs['gridSpacing'] = self.mesh_spacing
        dset.attrs["gridGlobalOffset"] = self.mesh_origin

        # Generic attributes
        dset.attrs["dataOrder"] = np.bytes_("C")
        dset.attrs["gridUnitSI"] = 1.
        dset.attrs["fieldSmoothing"] = np.bytes_("none")

        # Generic setup of the component
        self.setup_openpmd_component( dset )

        # Field positions
        dset.attrs["position"] = np.array([0.0, 0.0, 0.0])
