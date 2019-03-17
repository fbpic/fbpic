# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file defines the generic class OpenPMDDiagnostic.

This class is a common class from which both ParticleDiagnostic
and FieldDiagnostic inherit
"""
import os
import datetime
from dateutil.tz import tzlocal
import numpy as np
import h5py
from fbpic import __version__ as fbpic_version

# Dictionaries of correspondance for openPMD
from .data_dict import unit_dimension_dict

class OpenPMDDiagnostic(object) :
    """
    Generic class that contains methods which are common
    to both FieldDiagnostic and ParticleDiagnostic
    """

    def __init__(self, period, comm, write_dir=None,
                iteration_min=0, iteration_max=np.inf,
                dt_period=None, dt_sim=None ):
        """
        General setup of the diagnostic

        Parameters
        ----------
        period : int, optional
            The period of the diagnostics, in number of timesteps.
            (i.e. the diagnostics are written whenever the number
            of iterations is divisible by `period`). Specify either this or
            `dt_period`.

        comm : an fbpic BoundaryCommunicator object or None
            If this is not None, the data is gathered on the first proc
            Otherwise, each proc writes its own data.
            (Make sure to use different write_dir in this case.)

        write_dir : string, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory

        iteration_min, iteration_max: ints
            The iterations between which data should be written
            (`iteration_min` is inclusive, `iteration_max` is exclusive)

        dt_period : float (in seconds), optional
            The period of the diagnostics, in physical time of the simulation.
            Specify either this or `period`

        dt_sim : float (in seconds), optional
            The timestep of the simulation.
            Only needed if `dt_period` is not None.
        """
        # Get the rank of this processor
        if comm is not None :
            self.rank = comm.rank
        else :
            self.rank = 0

        # Check period argument
        if ((period is None) and (dt_period is None)):
            raise ValueError("You need to pass either `period` or `dt_period`"
                "to the diagnostics.")
        if ((period is not None) and (dt_period is not None)):
            raise ValueError("You need to pass either `period` or `dt_period`"
                "to the diagnostics, \nbut do not pass both.")

        # Get the diagnostic period
        if period is None:
            period = dt_period/dt_sim  # Extract it from `dt_period`
        self.period = max(1, int(round(period))) # Impose non-zero integer

        # Register the arguments
        self.iteration_min = iteration_min
        self.iteration_max = iteration_max
        self.comm = comm

        # Get the directory in which to write the data
        if write_dir is None:
            self.write_dir = os.path.join( os.getcwd(), 'diags' )
        else:
            self.write_dir = os.path.abspath(write_dir)

        # Create a few addiditional directories within self.write_dir
        self.create_dir("")
        self.create_dir("hdf5")

    def open_file( self, fullpath ):
        """
        Open a file on either several processors or a single processor
        (For the moment, only single-processor is enabled, but this
        routine is a placeholder for future multi-proc implementation)

        If a processor does not participate in the opening of
        the file, this returns None, for that processor

        Parameter
        ---------
        fullpath: string
            The absolute path to the openPMD file

        Returns
        -------
        An h5py.File object, or None
        """
        # In gathering mode, only the first proc opens/creates the file.
        if self.rank == 0 :
            # Create the filename and open hdf5 file
            f = h5py.File( fullpath, mode="a" )
        else:
            f = None

        return(f)


    def write( self, iteration ) :
        """
        Check if the data should be written at this iteration
        (based on iteration) and if yes, write it.

        Parameter
        ---------
        iteration : int
            The current iteration number of the simulation.
        """
        # Check if the fields should be written at this iteration
        if iteration % self.period == 0 \
            and iteration >= self.iteration_min \
            and iteration < self.iteration_max:

            # Write the hdf5 file if needed
            self.write_hdf5( iteration )


    def create_dir( self, dir_path) :
        """
        Check whether the directory exists, and if not create it.

        Parameter
        ---------
        dir_path : string
           Relative path from the directory where the diagnostics
           are written
        """
        # The following operations are done only by the first processor.
        if self.rank == 0 :

            # Get the full path
            full_path = os.path.join( self.write_dir, dir_path )

            # Check wether it exists, and create it if needed
            if os.path.exists(full_path) == False :
                try:
                    os.makedirs(full_path)
                except OSError :
                    pass

    def setup_openpmd_file( self, f, iteration, time, dt ) :
        """
        Sets the attributes of the hdf5 file, that comply with OpenPMD

        Parameter
        ---------
        f : an h5py.File object

        iteration: int
            The iteration number of this diagnostic

        time: float (seconds)
            The physical time at this iteration

        dt: float (seconds)
            The timestep of the simulation
        """
        # Set the attributes of the HDF5 file

        # General attributes
        f.attrs["openPMD"] = np.string_("1.0.0")
        f.attrs["openPMDextension"] = np.uint32(1)
        f.attrs["software"] = np.string_("fbpic " + fbpic_version)
        f.attrs["date"] = np.string_(
            datetime.datetime.now(tzlocal()).strftime('%Y-%m-%d %H:%M:%S %z'))
        f.attrs["meshesPath"] = np.string_("fields/")
        f.attrs["particlesPath"] = np.string_("particles/")
        f.attrs["iterationEncoding"] = np.string_("fileBased")
        f.attrs["iterationFormat"] =  np.string_("data%T.h5")

        # Setup the basePath
        f.attrs["basePath"] = np.string_("/data/%T/")
        base_path = "/data/%d/" %iteration
        bp = f.require_group( base_path )
        bp.attrs["time"] = time
        bp.attrs["dt"] = dt
        bp.attrs["timeUnitSI"] = 1.

    def setup_openpmd_record( self, dset, quantity ) :
        """
        Sets the attributes of a record, that comply with OpenPMD

        Parameter
        ---------
        dset : an h5py.Dataset or h5py.Group object

        quantity : string
           The name of the record considered
        """
        if quantity.startswith('rho'): # particle density such as rho_electrons
            quantity = 'rho'

        dset.attrs["unitDimension"] = unit_dimension_dict[quantity]
        # No time offset (approximation)
        dset.attrs["timeOffset"] = 0.

    def setup_openpmd_component( self, dset ) :
        """
        Sets the attributes of a component, that comply with OpenPMD

        Parameter
        ---------
        dset : an h5py.Dataset or h5py.Group object
        """
        dset.attrs["unitSI"] = 1.
