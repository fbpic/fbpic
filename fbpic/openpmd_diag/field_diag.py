# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file defines the class FieldDiagnostic.
"""
import os
import numpy as np
from .generic_diag import OpenPMDDiagnostic

class FieldDiagnostic(OpenPMDDiagnostic):
    """
    Class that defines the field diagnostics to be performed.
    """

    def __init__(self, period=None, fldobject=None, comm=None,
                 fieldtypes=["rho", "E", "B", "J"], write_dir=None,
                 iteration_min=0, iteration_max=np.inf, dt_period=None ) :
        """
        Initialize the field diagnostic.

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

        fldobject : a Fields object
            Points to the data that has to be written at each output

        comm : an fbpic BoundaryCommunicator object or None
            If this is not None, the data is gathered on the first proc,
            and the guard cells are removed from the output.
            Otherwise, each proc writes its own data, including guard cells.
            (Make sure to use different write_dir in this case.)

        fieldtypes : a list of strings, optional
            The strings are either "rho", "E", "B" or "J"
            and indicate which field should be written.
            Default : all fields are written

        write_dir : string, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory.

        iteration_min, iteration_max: ints
            The iterations between which data should be written
            (`iteration_min` is inclusive, `iteration_max` is exclusive)
        """
        # Check input
        if fldobject is None:
            raise ValueError(
            "You need to pass the argument `fldobject` to `FieldDiagnostic`.")

        # General setup
        OpenPMDDiagnostic.__init__(self, period, comm, write_dir,
                            iteration_min, iteration_max,
                            dt_period=dt_period, dt_sim=fldobject.dt )

        # Register the arguments
        self.fld = fldobject
        self.fieldtypes = fieldtypes
        self.coords = ['r', 't', 'z']

    def write_hdf5( self, iteration ):
        """
        Write an HDF5 file that complies with the OpenPMD standard

        Parameter
        ---------
        iteration : int
             The current iteration number of the simulation.
        """
        # If needed: Bring rho/J from spectral space (where they where
        # smoothed/corrected) to real space
        if "rho" in self.fieldtypes:
            # Get 'rho_prev', since it correspond to rho at time n
            self.fld.spect2interp('rho_prev')
            # Exchange rho in real space if needed
            if (self.comm is not None) and (self.comm.size > 1) \
                and (not self.fld.exchanged_source['rho_prev']):
                    self.comm.exchange_fields(self.fld.interp, 'rho', 'add')
        if "J" in self.fieldtypes:
            self.fld.spect2interp('J')
            # Exchange J in real space if needed
            if (self.comm is not None) and (self.comm.size > 1) \
                and (not self.fld.exchanged_source['J']):
                    self.comm.exchange_fields(self.fld.interp, 'J', 'add')

        # If needed: Receive data from the GPU
        if self.fld.use_cuda :
            self.fld.receive_fields_from_gpu()

        # Extract information needed for the openPMD attributes
        dt = self.fld.dt
        time = iteration * dt
        dz = self.fld.interp[0].dz
        if self.comm is None:
            # No communicator: dump all the present subdomain
            # (including damp cells and guard cells)
            zmin = self.fld.interp[0].zmin
            Nz = self.fld.interp[0].Nz
            Nr = self.fld.interp[0].Nr
        else:
            # Communicator present: only dump physical cells
            zmin, _ = self.comm.get_zmin_zmax(
                    local=False, with_damp=False, with_guard=False )
            Nz, _ = self.comm.get_Nz_and_iz(
                    local=False, with_damp=False, with_guard=False )
            Nr = self.comm.get_Nr( with_damp=False )

        # Create the file with these attributes
        filename = "data%08d.h5" %iteration
        fullpath = os.path.join( self.write_dir, "hdf5", filename )
        self.create_file_empty_meshes(
            fullpath, iteration, time, Nr, Nz, zmin, dz, dt )

        # Open the file again, and get the field path
        f = self.open_file( fullpath )
        # (f is None if this processor does not participate in writing data)
        if f is not None:
            field_path = "/data/%d/fields/" %iteration
            field_grp = f[field_path]
        else:
            field_grp = None

        # Loop over the different quantities that should be written
        for fieldtype in self.fieldtypes:
            # Scalar field
            if fieldtype == "rho":
                self.write_dataset( field_grp, "rho", "rho" )
            # Vector field
            elif fieldtype in ["E", "B", "J"]:
                for coord in self.coords:
                    quantity = "%s%s" %(fieldtype, coord)
                    path = "%s/%s" %(fieldtype, coord)
                    self.write_dataset( field_grp, path, quantity )
            # PML field (save as scalar field)
            elif fieldtype.endswith("_pml"):
                self.write_dataset( field_grp, fieldtype, fieldtype )
            else:
                raise ValueError("Invalid string in fieldtypes: %s" %fieldtype)

        # Close the file (only the first proc does this)
        if f is not None:
            f.close()

        # Send data to the GPU if needed
        if self.fld.use_cuda :
            self.fld.send_fields_to_gpu()

    # Writing methods
    # ---------------
    def write_dataset( self, field_grp, path, quantity ) :
        """
        Write a given dataset

        Parameters
        ----------
        field_grp : an h5py.Group object
            The group that corresponds to the path indicated in meshesPath

        path : string
            The relative path where to write the dataset, in field_grp

        quantity : string
            Describes which field is being written.
            (Either rho, Er, Et, Ez, Br, Bz, Bt, Jr, Jt or Jz)
        """
        # Extract the correct dataset
        if field_grp is not None:
            dset = field_grp[path]
        else:
            dset = None

        # Write the mode 0 : only the real part is non-zero
        mode0 = self.get_dataset( quantity, 0 )
        if self.rank == 0:
            mode0 = mode0.T
            dset[0,:,:] = mode0[:,:].real
        # Write the higher modes
        # There is a factor 2 here so as to comply with the convention in
        # Lifschitz et al., which is also the convention adopted in Warp Circ
        for m in range(1,self.fld.Nm):
            mode = self.get_dataset( quantity, m )
            if self.rank == 0:
                mode = mode.T
                dset[2*m-1,:,:] = 2*mode[:,:].real
                dset[2*m,:,:] = 2*mode[:,:].imag

    def get_dataset( self, quantity, m ):
        """
        Get the field `quantity` in the mode `m`
        Gathers it on the first proc, in MPI mode

        Parameters
        ----------
        quantity: string
            Describes which field is being written.
            (Either rho, Er, Et, Ez, Br, Bz, Bt, Jr, Jt or Jz)

        m: int
            The index of the mode that is being written
        """
        # Get the data on each individual proc
        data_one_proc = getattr( self.fld.interp[m], quantity )

        # Gather the data
        if self.comm is not None:
            data_all_proc = self.comm.gather_grid_array( data_one_proc )
        else:
            data_all_proc = data_one_proc

        return( data_all_proc )

    # OpenPMD setup methods
    # ---------------------

    def create_file_empty_meshes( self, fullpath, iteration,
                                   time, Nr, Nz, zmin, dz, dt ):
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

        Nr, Nz: int
            The number of gridpoints along r and z in this diagnostics

        zmin: float (meters)
            The position of the left end of the box

        dz: float (meters)
            The resolution in z of this diagnostic

        dt: float (seconds)
            The timestep of the simulation
        """
        # Determine the shape of the datasets that will be written
        # First write real part mode 0, then imaginary part of higher modes
        data_shape = ( 2*self.fld.Nm - 1, Nr, Nz )

        # Create the file
        f = self.open_file( fullpath )

        # Setup the different layers of the openPMD file
        # (f is None if this processor does not participate is writing data)
        if f is not None:

            # Setup the attributes of the top level of the file
            self.setup_openpmd_file( f, iteration, time, dt )

            # Setup the meshes group (contains all the fields)
            field_path = "/data/%d/fields/" %iteration
            field_grp = f.require_group(field_path)
            self.setup_openpmd_meshes_group(field_grp)

            # Loop over the different quantities that should be written
            # and setup the corresponding datasets
            for fieldtype in self.fieldtypes:

                # Scalar field
                # e.g. 'rho', but also 'rho_electron' in the case of
                # the sub-class ParticleDensityDiagnostic
                # or PML component (saved as scalar field as well)
                if fieldtype.startswith("rho") or fieldtype.endswith("_pml"):
                    # Setup the dataset
                    dset = field_grp.require_dataset(
                        fieldtype, data_shape, dtype='f8')
                    self.setup_openpmd_mesh_component( dset, fieldtype )
                    # Setup the record to which it belongs
                    self.setup_openpmd_mesh_record( dset, fieldtype, dz, zmin )

                # Vector field
                elif fieldtype in ["E", "B", "J"]:
                    # Setup the datasets
                    for coord in self.coords:
                        quantity = "%s%s" %(fieldtype, coord)
                        path = "%s/%s" %(fieldtype, coord)
                        dset = field_grp.require_dataset(
                            path, data_shape, dtype='f8')
                        self.setup_openpmd_mesh_component( dset, quantity )
                    # Setup the record to which they belong
                    self.setup_openpmd_mesh_record(
                        field_grp[fieldtype], fieldtype, dz, zmin )

                # Unknown field
                else:
                    raise ValueError(
                        "Invalid string in fieldtypes: %s" %fieldtype)

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

    def setup_openpmd_mesh_record( self, dset, quantity, dz, zmin ) :
        """
        Sets the attributes that are specific to a mesh record

        Parameter
        ---------
        dset : an h5py.Dataset or h5py.Group object

        quantity : string
           The name of the record (e.g. "rho", "J", "E" or "B")

        dz: float (meters)
            The resolution in z of this diagnostic

        zmin: float (meters)
            The position of the left end of the grid
        """
        # Generic record attributes
        self.setup_openpmd_record( dset, quantity )

        # Geometry parameters
        dset.attrs['geometry'] = np.string_("thetaMode")
        dset.attrs['geometryParameters'] = \
            np.string_("m={:d};imag=+".format(self.fld.Nm))
        dset.attrs['gridSpacing'] = np.array([
                self.fld.interp[0].dr, dz ])
        dset.attrs["gridGlobalOffset"] = np.array([
            self.fld.interp[0].rmin, zmin ])
        dset.attrs['axisLabels'] = np.array([ b'r', b'z' ])

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
        dset.attrs["position"] = np.array([0.5, 0.5])
