"""
This file defines the class FieldDiagnostic
"""
import os
import h5py
import numpy as np
from generic_diag import OpenPMDDiagnostic

class FieldDiagnostic(OpenPMDDiagnostic) :
    """
    Class that defines the field diagnostics to be done.

    Usage
    -----
    After initialization, the diagnostic is called by using the
    `write` method.
    """

    def __init__(self, period, fldobject,
                 fieldtypes=["rho", "E", "B", "J"], write_dir=None ) :
        """
        Initialize the field diagnostic.

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
        """
        # General setup
        OpenPMDDiagnostic.__init__(self, period, write_dir)
        
        # Register the arguments
        self.fld = fldobject
        self.fieldtypes = fieldtypes
        
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
           The name of the record (e.g. "rho", "J", "E" or "B")
        """
        # Generic record attributes
        self.setup_openpmd_record( dset, quantity )
        
        # Geometry parameters
        dset.attrs['geometry'] = np.string_("thetaMode")
        dset.attrs['geometryParameters'] = \
            np.string_("m={:d};imag=+".format(self.fld.Nm))
        dset.attrs['gridSpacing'] = np.array([
                self.fld.interp[0].dr, self.fld.interp[0].dz ])
        dset.attrs['axisLabels'] = np.array([ 'r', 'z' ])
            
        # Generic attributes
        dset.attrs["dataOrder"] = np.string_("C")
        dset.attrs["gridUnitSI"] = 1.
        dset.attrs["gridGlobalOffset"] = np.array([
            self.fld.interp[0].rmin, self.fld.interp[0].zmin ])
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


    def write_hdf5( self, iteration ) :
        """
        Write an HDF5 file that complies with the OpenPMD standard

        Parameter
        ---------
        iteration : int
             The current iteration number of the simulation.
        """
        # Create the file
        filename = "data%08d.h5" %iteration
        fullpath = os.path.join( self.write_dir, "diags/hdf5", filename )
        f = h5py.File( fullpath, mode="a" )
        
        # Set up its attributes
        self.setup_openpmd_file( f, self.fld.dt,
                                 iteration*self.fld.dt, iteration )

        # Setup the fields group
        field_path = "/data/%d/fields/" %iteration
        field_grp = f.require_group(field_path)
        self.setup_openpmd_meshes_group( field_grp )
            
        # Loop over the different quantities that should be written
        for fieldtype in self.fieldtypes :
            # Scalar field
            if fieldtype == "rho" :
                self.write_dataset( field_grp, "rho", "rho" )
                self.setup_openpmd_mesh_record( field_grp["rho"], "rho" )
            # Vector field
            elif fieldtype in ["E", "B", "J"] :
                for coord in ["r", "t", "z"] :
                    quantity = "%s%s" %(fieldtype, coord)
                    path = "%s/%s" %(fieldtype, coord)
                    self.write_dataset( field_grp, path, quantity )
                    self.setup_openpmd_mesh_record(
                        field_grp[fieldtype], fieldtype )
            else :
                raise ValueError("Invalid string in fieldtypes: %s" %fieldtype)
        
        # Close the file    
        f.close()

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
        # Shape of the data : first write the real part mode 0
        # and then the imaginary part of the mode 1
        datashape = ( 2*self.fld.Nm + 1, self.fld.Nr, self.fld.Nz )
        dset = field_grp.require_dataset( path, datashape, dtype='f' )
        self.setup_openpmd_mesh_component( dset, quantity )

        # Write the mode 0 : only the real part is non-zero
        mode0 = getattr( self.fld.interp[0], quantity ).T
        dset[0,:,:] = mode0[:,:].real
        # Write the higher modes
        # There is a factor 2 here so as to comply with the convention in
        # Lifschitz et al., which is also the convention adopted in Warp Circ
        for i in range(1,self.fld.Nm):
            mode = getattr( self.fld.interp[i], quantity ).T
            dset[2*i-1,:,:] = 2*mode[:,:].real
            dset[2*i,:,:] = 2*mode[:,:].imag
