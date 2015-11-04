"""
This file defines the generic class OpenPMDDiagnostic.

This class is a common class from which both ParticleDiagnostic
and FieldDiagnostic inherit
"""
import os
import datetime
from dateutil.tz import tzlocal
import numpy as np

# Dictionaries of correspondance for openPMD
from data_dict import unit_dimension_dict

class OpenPMDDiagnostic(object) :
    """
    Generic class that contains methods which are common
    to both FieldDiagnostic and ParticleDiagnostic
    """

    def __init__(self, period, write_dir=None ) :
        """
        General setup of the diagnostic

        Parameters
        ----------
        period : int
            The period of the diagnostics, in number of timesteps.
            (i.e. the diagnostics are written whenever the number
            of iterations is divisible by `period`)
            
        write_dir : string, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory
        """    
        # Register the arguments
        self.period = period
        self.rank = 0
            
        # Get the directory in which to write the data
        if write_dir is None :
            self.write_dir = os.getcwd()
        else :
            self.write_dir = os.path.abspath(write_dir)

        # Create a few addiditional directories within self.write_dir
        self.create_dir("")
        self.create_dir("diags")
        self.create_dir("diags/hdf5")

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
        if iteration % self.period == 0 :

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

    def setup_openpmd_file( self, f, dt, t, iteration ) :
        """
        Sets the attributes of the hdf5 file, that comply with OpenPMD
    
        Parameter
        ---------
        f : an h5py.File object
    
        t : float (seconds)
            The absolute time at this point in the simulation
        
        dt : float (seconds)
            The timestep of the simulation

        iteration : int
            The iteration corresponding to this timestep
        """
        # Set the attributes of the HDF5 file
    
        # General attributes
        f.attrs["openPMD"] = np.string_("1.0.0")
        f.attrs["openPMDextension"] = np.uint32(1)
        f.attrs["software"] = np.string_("fbpic")
        f.attrs["date"] = np.string_(
            datetime.datetime.now(tzlocal()).strftime('%Y-%m-%d %H:%M:%S %z'))
        f.attrs["meshesPath"] = np.string_("fields/")
        f.attrs["particlesPath"] = np.string_("particles/")
        f.attrs["iterationEncoding"] = np.string_("fileBased")
        f.attrs["iterationFormat"] =  np.string_("data%T.h5")

        # Setup the basePath
        base_path = "/data/%d/" %iteration
        f.attrs["basePath"] = np.string_(base_path)
        bp = f.require_group( base_path )
        bp.attrs["time"] = t
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

