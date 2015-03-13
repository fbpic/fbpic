"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of classes for the output of the code.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import datetime

class FieldDiagnostic(object) :
    """
    Class that defines the field diagnostics to be done.
    """

    def __init__(self, period, fldobject, write_dir=None,
                 output_pickle=True, output_png=True ) :
        """
        Initialize the field diagnostics.

        Parameters
        ----------
        period : int
            The period of the diagnostics in number of timesteps
            
        fieldtypes : list of strings
            The fields that should be 

            
        
        """
        # Register the arguments
        self.period = self.period
        self.dt = dt
        
        # Get the directory in which to write the data
        if type(write_dir) is not string :
            self.write_dir = os.getcwd()
        else :
            self.write_dir = write_dir
        
        # Create a few addiditional directories within this
        self.create_dir("diags")
        if output_pickle :
            self.create_dir("diags/pickle")
        if output_png : 
            self.create_dir("diags/png")

    def write( self, iteration ) :

        # Check if the fields should be written at this iteration
        if iteration % self.period == 0 :
            
            # Write the png files if needed
            if self.pickle == True :
                for fieldtype in fieldtypes :
                    self.write_png( fieldtype, iteration )
    
            if self.pickle == True :
                self.write_hdf5( iteration )
        
        # Copy the liveplots stuff

    def write_hdf5( self, iteration ) :
        """
        Write an HDF5 file that complies with the OpenPMD standard

        """
        # Create the filename and open hdf5 file
        filename = "fields%08d.h5" %iteration
        fullpath = os.path.join( self.write_dir, "hdf5", filename )
        f = h5py.File( fullpath, mode="w" )
        
        # 

            
        setup_openpmd( self, f )

        f.close()
        
    def setup_openpmd( f ) :
        """
        Sets the attributes of the hdf5 file that comply with OpenPMD,
        and creates its internal structure

        Parameter
        ---------
        f : an h5py.File object
        """
        # Set the attributes of the HDF5 file

        # General attributes
        f.attrs["software"] = "fbpic 1.0"
        today = datetime.datetime.now()
        f.attrs["date"] = today.strftime("%Y-%m-%d %H:%M:%S")
        f.attrs["version"] = "1.0.0"  # OpenPMD version
        f.attrs["basePath"] = "/"
        f.attrs["fieldsPath"] = "fields/"
        f.attrs["particlesPath"] = "particles/"
        # TimeSeries attributes
        f.attrs["timeStepEncoding"] = "fileBased"
        f.attrs["timeStepFormat"] = "fields%T.h5"
        f.attrs["timeStepUnitSI"] = self.period*self.fld.dt
        
        # Create the datasets
        dz = self.fld.interp[0].dz
        dr = self.fld.interp[0].dr
        datashape = ( self.fld.Nz, self.fld.Nr, 2*self.fld.Nm - 1 )
        for fieldtype in self.fieldtypes :
            if fieldtype == "rho" :
                dset = f.create_dataset("/fields/rho", datashape, dtype='f')
                setup_dataset( dset, dz, dr )

            if fieldtype in ["E", "B", "J"] :
                for coord in ["r", "t", "z"] :
                    dset = f.create_dataset("/fields/%s/%s" %(fieldtype,coord),
                                            datashape, dtype='f')
                    setup_dataset( dset, dz, dr )
                    


                    
def create_dir(dir_path) :
    """
    Check whether the directory exists, and if not create it.

    Parameter
    ---------
    dir_path : string
        Relative path from self.write_dir
    """
    # Get the full path
    full_path = os.path.join( self.write_dir, dir_path )
    
    # Check wether it exists, and create it if needed
    if os.path.exists(full_path) == False :
        os.makedirs(full_path)


def setup_dataset( dset, dz, dr ) :
    """
    Sets up the attribute of the dataset

    Parameters
    ----------
    dest : an h5py dataset object

    dz, dr : float (meters)
        The size of the steps on the grid
    """
    dset.attrs["unitSI"] = 1.
    dset.attrs["gridUnitSI"] = 1.
    dset.attrs["dx"] = dz
    dset.attrs["dy"] = dr
    dset.attrs["posX"] = 0.5  # All data is node-centered
    dset.attrs["posY"] = 0.5  # All data is node-centered
    dset.attrs["coordSystem"] = "right-handed"
    dset.attrs["dataOrder"] = "kji"
    dset.attrs["fieldSolver"] = "PSATD"
    dset.attrs["fieldSolverOrder"] = -1.0
    dset.attrs["fieldSmoothing"] = "None"
