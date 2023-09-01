# Copyright 2023, FBPIC contributors
# Authors: Igor A Andriyash, Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file defines the class SRDiagnostic.
"""
import os
import numpy as np
from scipy.constants import e, hbar
from .generic_diag import OpenPMDDiagnostic
from fbpic.utils.mpi import comm as comm_simple

class SRDiagnostic(OpenPMDDiagnostic):
    """
    Class that defines the synchrotron radiation diagnostics to be performed.
    """

    def __init__(self, period=None, dt_period=None, sr_object=None, comm=None,
                 write_dir=None,iteration_min=0, iteration_max=np.inf ):
        """
        Initialize the synchrotron radiation diagnostic.

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

        sr_object : a Synchrotron Radiation object
            Points to the data that has to be written at each output

        comm : an fbpic BoundaryCommunicator object or None
            If this is not None, the data is gathered on the first proc,
            and the guard cells are removed from the output.
            Otherwise, each proc writes its own data, including guard cells.
            (Make sure to use different write_dir in this case.)

        write_dir : string, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory.

        iteration_min, iteration_max: ints
            The iterations between which data should be written
            (`iteration_min` is inclusive, `iteration_max` is exclusive)
        """
        # Check input
        if sr_object is None:
            raise ValueError(
            "You need to pass the argument `sr_object` to `SRDiagnostic`.")

        # General setup
        OpenPMDDiagnostic.__init__(self, period, comm, write_dir,
                            iteration_min, iteration_max,
                            dt_period=dt_period, dt_sim=sr_object.dt )

        # Register the arguments
        self.fld = sr_object

    def write_hdf5( self, iteration ):
        """
        Write an HDF5 file that complies with the OpenPMD standard

        Parameter
        ---------
        iteration : int
             The current iteration number of the simulation.
        """

        # If needed: Receive data from the GPU
        if self.fld.use_cuda :
            self.fld.receive_from_gpu()

        # Extract information needed for the openPMD attributes
        dt = self.fld.dt
        time = iteration * dt

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
        if self.fld.use_cuda :
            self.fld.send_to_gpu()

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
        # Extract the correct dataset
        data_array = self.get_dataset()
        if field_grp is not None:
            dset = field_grp[path]
            dset[:] =  data_array
        else:
            dset = None

    def get_dataset( self ):
        """
        Copy and dathers radation data on the first proc, in MPI mode
        """
        # Get the data on each individual proc
        data_one_proc = self.fld.radiation_data.copy()

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
        # Determine the shape of the datasets that will be written
        data_shape = ( self.fld.N_theta_x, self.fld.N_theta_y,
                       self.fld.N_omega )

        # Create the file
        f = self.open_file( fullpath )

        # Setup the different layers of the openPMD file
        # (f is None if this processor does not participate is writing data)
        if f is not None:

            # Setup the attributes of the top level of the file
            self.setup_openpmd_file( f, iteration, time, self.fld.dt )

            # Setup the meshes group (contains all the fields)
            field_path = "/data/%d/fields/" %iteration
            field_grp = f.require_group(field_path)
            self.setup_openpmd_meshes_group(field_grp)

            dset = field_grp.require_dataset(
                "radiation", data_shape, dtype='f8')
            self.setup_openpmd_mesh_component( dset, "radiation" )
            # Setup the record to which it belongs
            self.setup_openpmd_mesh_record( dset, "radiation" )
            # Close the file
            f.close()

    def setup_openpmd_meshes_group( self, dset ) :
        """
        Set the attributes that are specific to the mesh path

        Parameter
        ---------
        dset : an h5py.Group object that contains all the mesh quantities
        """
        # Field Solver
        dset.attrs["fieldSolver"] = np.string_("PSATD")
        # Field boundary
        dset.attrs["fieldBoundary"] = np.array([
            np.string_("reflecting"), np.string_("reflecting"),
            np.string_("reflecting"), np.string_("reflecting") ])
        # Particle boundary
        dset.attrs["particleBoundary"] = np.array([
            np.string_("absorbing"), np.string_("absorbing"),
            np.string_("absorbing"), np.string_("absorbing") ])
        # Current Smoothing
        dset.attrs["currentSmoothing"] = np.string_("Binomial")
        dset.attrs["currentSmoothingParameters"] = \
          np.string_("period=1;numPasses=1;compensator=false")
        # Charge correction
        dset.attrs["chargeCorrection"] = np.string_("spectral")
        dset.attrs["chargeCorrectionParameters"] = np.string_("period=1")

    def setup_openpmd_mesh_record( self, dset, quantity ) :
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
        dset.attrs['geometry'] = np.string_("cartesian")
        dset.attrs['axisLabels'] = np.array([ b'x', b'y', b'z' ])

        #omega_keV = hbar / e * 1e-3
        omega_J = hbar
        dset.attrs['gridSpacing'] = np.array([
                self.fld.d_theta_x, self.fld.d_theta_y,
                self.fld.d_omega * omega_J ])

        dset.attrs["gridGlobalOffset"] = np.array([
            self.fld.theta_x_min, self.fld.theta_x_min,
            self.fld.omega_min * omega_J ])

        # Generic attributes
        dset.attrs["dataOrder"] = np.string_("C")
        dset.attrs["gridUnitSI"] = 1.
        dset.attrs["fieldSmoothing"] = np.string_("none")

    def setup_openpmd_mesh_component( self, dset, quantity ) :
        """
        Set up the attributes of a mesh component

        Parameter
        ---------
        dset : an h5py.Dataset or h5py.Group object

        quantity : string
            The field that is being written
        """
        # Generic setup of the component
        self.setup_openpmd_component( dset )

        # Field positions
        dset.attrs["position"] = np.array([0.0, 0.0, 0.0])
