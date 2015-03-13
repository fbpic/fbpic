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

    def __init__(self, period, fldobject, fieldtypes=["rho", "E", "B", "J"],
                 write_dir=None, output_hdf5=True, output_png=True ) :
        """
        Initialize the field diagnostics.

        Parameters
        ----------
        period : int
            The period of the diagnostics, in number of timesteps.
            (i.e. the diagnostics are written whenever the number
            of iterations is divisible by `period`)
            
        fldobject : a Fields object
            Points to the data that has to be written at each output

        fieldtypes : a list of strings, optional
            The strings are either "rho", "E", "B" or "J"
            and indicate which field should be written.
            Default : all fields are written
            
        write_dir : string, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory.

        output_hdf5 : bool, optional
            Whether to output an HDF5 file, every `period` timestep

        output_png : bool, optional
            Whether to output PNG images, every `period` timestep
        """
        # Register the arguments
        self.period = self.period
        self.fld = fldobject
        self.output_hdf5 = output_hdf5
        self.output_png = output_png
        
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
        """
        Check if the fields should be written at this iteration
        (based on self.period) and if yes, write them.

        Parameter
        ---------
        iteration : int
             The current iteration number of the simulation.
        """
        
        # Check if the fields should be written at this iteration
        if iteration % self.period == 0 :
            
            # Write the png files if needed
            if self.output_png == True :
                for fieldtype in fieldtypes :
                    self.write_png( fieldtype, iteration )

            # Write the hdf5 file if needed
            if self.output_hdf5 == True :
                self.write_hdf5( iteration )

    def write_hdf5( self, iteration ) :
        """
        Write an HDF5 file that complies with the OpenPMD standard

        Parameter
        ---------
        iteration : int
             The current iteration number of the simulation.
        """
        # Create the filename and open hdf5 file
        filename = "fields%08d.h5" %iteration
        fullpath = os.path.join( self.write_dir, "hdf5", filename )
        f = h5py.File( fullpath, mode="w" )
        
        # Set up its attributes            
        setup_openpmd_file( f, self.period*self.fld.dt )

        # Create the datasets
        # Loop over the different quantities that should be written
        for fieldtype in self.fieldtypes :
            # Scalar field
            if fieldtype == "rho" :
                write_dataset( f, "/fields/rho",
                    getattr( self.fld.interp[0], "rho"),
                    getattr( self.fld.interp[1], "rho" ),
                    self.fld.interp[0].dz, self.fld.interp[0].dr )
            # Vector field
            elif fieldtype in ["E", "B", "J"] :
                for coord in ["r", "t", "z"] :
                    quantity = "%s%s" %(fieldtype, coord)
                    path = "/fields/%s/%s" %(fieldtype, coord)
                    write_dataset( f, path,
                        getattr( self.fld.interp[0], quantity ),
                        getattr( self.fld.interp[1], quantity ),
                        self.fld.interp[0].dz, self.fld.interp[0].dr )
            else :
                raise ValueError("Invalid string in fieldtypes: %s" %fieldtype)
        
        # Close the file
        f.close()

# ------------------
# Utility functions
# ------------------

def write_dataset( f, path, mode0, mode1, dz, dr ) :
    """
    Write 

    Parameters
    ----------
    f : an h5py.File object

    path : string
        The path where to write the dataset, inside the file f

    mode0, mode1 : 2darray of complexs
        Represent the fields in the mode 0 and mode 1

    dz, dr : floats (meters)
        The size of the steps on the grid
    """
    # Shape of the data : first write the real part mode 0
    # and then the imaginary part of the mode 1
    datashape = (3, mode0.shape[0], mode0.shape[1])
        
    # Create the dataset and setup its attributes
    dset = f.create_dataset( path, datashape, dtype='f')
    setup_openpmd_dataset( dset, dz, dr )
    
    # Write the mode 0 : only the real part is non-zero
    dset[0,:,:] = mode0[:,:].real
    
    # Write the real and imaginary part of the mode 1
    # There is a factor 2 here so as to comply with the convention in
    # Lifschitz et al., which is also the convention adopted in Warp Circ
    dset[1,:,:] = 2*mode1[:,:].real
    dset[2,:,:] = 2*mode1[:,:].imag

                    
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


def setup_openpmd_file( f, time_interval ) :
    """
    Sets the attributes of the hdf5 file, that comply with OpenPMD
    
    Parameter
    ---------
    f : an h5py.File object

    time_interval : float (seconds)
        The interval between successive timesteps
    """
    # Set the attributes of the HDF5 file

    # General attributes
    f.attrs["software"] = "fbpic-1.0"
    today = datetime.datetime.now()
    f.attrs["date"] = today.strftime("%Y-%m-%d %H:%M:%S")
    f.attrs["version"] = "1.0.0"  # OpenPMD version
    f.attrs["basePath"] = "/"
    f.attrs["fieldsPath"] = "fields/"
    f.attrs["particlesPath"] = "particles/"
    # TimeSeries attributes
    f.attrs["timeStepEncoding"] = "fileBased"
    f.attrs["timeStepFormat"] = "fields%T.h5"
    f.attrs["timeStepUnitSI"] = time_interval


def setup_openpmd_dataset( dset, dz, dr ) :
    """
     Sets the attributes of the dataset, that comply with OpenPMD

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
    dset.attrs["dataOrder"] = "kji"  # column-major order due to numpy
    dset.attrs["fieldSolver"] = "PSATD"
    dset.attrs["fieldSolverOrder"] = -1.0
    dset.attrs["fieldSmoothing"] = "None"


    
