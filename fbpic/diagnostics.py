"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of classes for the output of the code.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import h5py
import datetime

class OpenPMDDiagnostic(object) :
    """
    Generic class that contains methods which contains common
    to both FieldDiagnostic and ParticleDiagnostic
    """

    def __init__(self, write_dir ) :
        """
        General setup of the diagnostic

        Parameters
        ----------
        write_dir : string, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory.
        """ 
        # Get the directory in which to write the data
        if write_dir is None :
            self.write_dir = os.getcwd()
        else :
            self.write_dir = os.path.abspath(write_dir)
            
        # Create a few addiditional directories within self.write_dir
        self.create_dir("")
        self.create_dir("diags")
        self.create_dir("diags/hdf5")

    def create_dir( self, dir_path) :
        """
        Check whether the directory exists, and if not create it.
    
        Parameter
        ---------
        dir_path : string
           Relative path from the directory where the diagnostics
           are written
        """
        # Get the full path
        full_path = os.path.join( self.write_dir, dir_path )
        
        # Check wether it exists, and create it if needed
        if os.path.exists(full_path) == False :
            os.makedirs(full_path)

    @staticmethod
    def setup_openpmd_file( f, dt, t ) :
        """
        Sets the attributes of the hdf5 file, that comply with OpenPMD
        
        Parameter
        ---------
        f : an h5py.File object
    
        t : float (seconds)
        The absolute time at this point in the simulation
        
        dt : float (seconds)
        The timestep of the simulation
        """
        # General attributes
        f.attrs["software"] = "fbpic-1.0"
        today = datetime.datetime.now()
        f.attrs["date"] = today.strftime("%Y-%m-%d %H:%M:%S")
        f.attrs["version"] = "1.0.0"  # OpenPMD version
        f.attrs["basePath"] = "/"
        f.attrs["fieldsPath"] = "fields/"
        f.attrs["particlesPath"] = "particles/"
        # TimeSeries attributes
        f.attrs["iterationEncoding"] = "fileBased"
        f.attrs["iterationFormat"] = "data%T.h5"
        f.attrs["time"] = t
        f.attrs["timeStep"] = dt
        f.attrs["timeUnitSI"] = 1.
    
# ------------------------
# FieldDiagnostic class
# ------------------------
            
class FieldDiagnostic(OpenPMDDiagnostic) :
    """
    Class that defines the field diagnostics to be done.

    Usage
    -----
    After initialization, the diagnostic is called by using the write method.
    """

    def __init__(self, period, fldobject, fieldtypes=["rho", "E", "B", "J"],
                    write_dir=None, output_png_spectral=False,
                    output_png_spatial=False ) :
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

        output_png_spectral : bool, optional
            Whether to output PNG images of the spectral fields,
            every `period` timestep
            Default : deactivated, since it is very slow

        output_png_spatial : bool, optional
            Whether to output PNG images of the spatial fields,
            every `period` timestep
            Default : deactivated, since it is very slow
        """
        # General setup
        OpenPMDDiagnostic.__init__(self, write_dir)
        
        # Register the arguments
        self.period = period
        self.fld = fldobject
        self.fieldtypes = fieldtypes
        self.output_png_spatial = output_png_spatial
        self.output_png_spectral = output_png_spectral
        
        # Directory for PNG files
        if output_png_spatial or output_png_spectral :
            
            # Fields on the spatial grid
            if output_png_spatial :
                self.create_dir("diags/png_spatial")
                # Loop over fieldtypes
                for fieldtype in self.fieldtypes :
                    if fieldtype == "rho" :
                        self.create_dir("diags/png_spatial/rho")
                    elif fieldtype in ["E", "B", "J"] :
                        for coord in ["r", "t", "z"] :
                            path ="diags/png_spatial/%s%s" %(fieldtype, coord)
                            self.create_dir(path)

            # Fields on the spectral grid
            if output_png_spectral :
                self.create_dir("diags/png_spectral")
                # Loop over fieldtypes
                for fieldtype in self.fieldtypes :
                    if fieldtype == "rho" :
                        self.create_dir("diags/png_spectral/rho_next")
                    elif fieldtype in ["E", "B", "J"] :
                        for coord in ["p", "m", "z"] :
                            path ="diags/png_spectral/%s%s" %(fieldtype, coord)
                            self.create_dir(path)
        

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
            if self.output_png_spectral or self.output_png_spatial :
                for fieldtype in self.fieldtypes :
                    self.write_png( iteration )

            # Write the hdf5 files
            self.write_hdf5( iteration )
                

    def write_png( self, iteration ) :
        """
        Write the PNG files for the different required fields

        Parameter
        ---------
        iteration : int
             The current iteration number of the simulation.
        """
        # Spatial fields
        if self.output_png_spatial :
            fullpath = os.path.join( self.write_dir, "diags/png_spatial" )
            
            # Loop through the different fields
            for fieldtype in self.fieldtypes :
                # Scalar field
                if fieldtype == "rho" :
                    self.write_png_file( fullpath, iteration, "rho",
                                self.fld.interp[0], self.fld.interp[1] )
                # Vector field
                elif fieldtype in ["E", "B", "J"] :
                    for coord in ["r", "t", "z"] :
                        quantity = "%s%s" %(fieldtype, coord)
                        self.write_png_file( fullpath, iteration, quantity,
                            self.fld.interp[0], self.fld.interp[1] )

        # Spectral fields
        if self.output_png_spectral :
            fullpath = os.path.join( self.write_dir, "diags/png_spectral" )

            # Loop through the different fields
            for fieldtype in self.fieldtypes :
                if fieldtype == "rho" :                
                    write_png_file( fullpath, iteration, "rho_next",
                            self.fld.spect[0], self.fld.spect[1] )
                # Vector field
                elif fieldtype in ["E", "B", "J"] :
                    for coord in ["p", "m", "z"] :
                        quantity = "%s%s" %(fieldtype, coord)
                        write_png_file( fullpath, iteration, quantity,
                            self.fld.spect[0], self.fld.spect[1] )

    @staticmethod
    def write_png_file( path, iteration, quantity, mode0, mode1 ) :
        """
        Write the modes 0 and 1 of the field `quantity` to a PNG file.
    
        Parameters
        ----------
        path : string
        Path to the directory where the PNG files of the different
        fields are stored
    
        quantity : string
        The actual field that is to be plotted
    
        mode0, mode1 : InterpolationGrid objects or SpectralGrid objects
        Objects having the method `show`, that allows to plot the fields
        """
    
        # Plot the mode 0
        filename = os.path.join( path, quantity, "mode0_%08d.png" %iteration)
        plt.clf()
        mode0.show( quantity )
        plt.savefig( filename )
    
        # Plot the mode 1
        filename = os.path.join( path, quantity, "mode1_%08d.png" %iteration)
        plt.clf()
        mode1.show( quantity )
        plt.savefig( filename )
                        
    def write_hdf5( self, iteration ) :
        """
        Write an HDF5 file that complies with the OpenPMD standard

        Parameter
        ---------
        iteration : int
             The current iteration number of the simulation.
        """
        # Create the filename and open hdf5 file
        filename = "data%08d.h5" %iteration
        fullpath = os.path.join( self.write_dir, "diags/hdf5", filename )
        f = h5py.File( fullpath, mode="a" )
        
        # Set up its attributes            
        self.setup_openpmd_file( f, self.fld.dt, iteration*self.fld.dt )

        # Create the datasets
        # Loop over the different quantities that should be written
        for fieldtype in self.fieldtypes :
            # Scalar field
            if fieldtype == "rho" :
                self.write_dataset( f, "/fields/rho", "rho" )
            # Vector field
            elif fieldtype in ["E", "B", "J"] :
                for coord in ["r", "t", "z"] :
                    quantity = "%s%s" %(fieldtype, coord)
                    path = "/fields/%s/%s" %(fieldtype, coord)
                    self.write_dataset( f, path, quantity )
            else :
                raise ValueError("Invalid string in fieldtypes: %s" %fieldtype)
        
        # Close the file
        f.close()

    def write_dataset( self, f, path, quantity ) :
        """
        Write a given dataset
    
        Parameters
        ----------
        f : an h5py.File object
    
        path : string
            The path where to write the dataset, inside the file f
    
        quantity : string
            The name of the quantity to be written
        """
        # Extract the modes
        mode0 = getattr( self.fld.interp[0], quantity )
        mode1 = getattr( self.fld.interp[1], quantity )

        # Extract geometry of the box
        dz = self.fld.interp[0].dz
        dr = self.fld.interp[0].dr
        zmin = self.fld.interp[0].zmin

        # Shape of the data : first write the real part mode 0
        # and then the imaginary part of the mode 1
        datashape = (3, mode0.shape[0], mode0.shape[1])
        
        # Create the dataset and setup its attributes
        dset = f.require_dataset( path, datashape, dtype='f')
        self.setup_openpmd_dataset( dset, dz, dr, zmin )
        
        # Write the mode 0 : only the real part is non-zero
        dset[0,:,:] = mode0[:,:].real
        
        # Write the real and imaginary part of the mode 1
        # There is a factor 2 here so as to comply with the convention in
        # Lifschitz et al., which is also the convention adopted in Warp Circ
        dset[1,:,:] = 2*mode1[:,:].real
        dset[2,:,:] = 2*mode1[:,:].imag

    @staticmethod
    def setup_openpmd_dataset( dset, dz, dr, zmin ) :
        """
        Sets the attributes of the dataset, that comply with OpenPMD
    
        Parameters
        ----------
        dest : an h5py dataset object
    
        dz, dr : float (meters)
        The size of the steps on the grid
    
        zmin : float (meters)
        The position of the edge of the simulation box ablong the z direction
        """
        dset.attrs["unitSI"] = 1.
        dset.attrs["gridUnitSI"] = 1.
        dset.attrs["gridSpacing"] = np.array([dr, dz])
        dset.attrs["gridGlobalOffset"] = np.array([ 0., zmin])
        dset.attrs["position"] = np.array([ 0.5, 0.5])
        dset.attrs["dataOrder"] = "kji"  # column-major order due to numpy
        dset.attrs["fieldSolver"] = "PSATD"
        dset.attrs["fieldSolverOrder"] = -1.0
        dset.attrs["fieldSmoothing"] = "None"
        
# ------------------------
# ParticleDiagnostic class
# ------------------------

class ParticleDiagnostic(OpenPMDDiagnostic) :
    """
    Class that defines the particle diagnostics to be done.

    Usage
    -----
    After initialization, the diagnostic is called by using the
    `write` method.
    """

    def __init__(self, period, species,
                 particle_data=["position", "momentum", "weighting"],
                 select={"uz" : [0.5, None]}, write_dir=None ) :
        """
        Initialize the field diagnostics.

        Parameters
        ----------
        period : int
            The period of the diagnostics, in number of timesteps.
            (i.e. the diagnostics are written whenever the number
            of iterations is divisible by `period`)
        
        species : a dictionary of Particles objects
            The Species object that is written (e.g. elec)
            is assigned to the particleName of this species.
            (e.g. { "electrons" : )

        particle_data : a list of strings, optional 
            The particle properties are given by:
            ["position", "momentum", "weighting"]
            for the coordinates x,y,z.
            Default : electron particle data is written

        select : dict, optional
            A set of rules to select the particles, of the form
            'x' : [-4., 10.]   (Particles having x between -4 and 10 microns)
            'ux' : [-0.1, 0.1] (Particles having ux between -0.1 and 0.1 mc)
            'uz' : [5., None]  (Particles with uz above 5 mc)
            
        write_dir : a list of strings, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory.
        """
        # General setup
        OpenPMDDiagnostic.__init__(self, write_dir)
        
        # Register the arguments
        self.period = period
        self.particle_data = particle_data
        self.species = species
        self.select = select

        # Extract the timestep from a given species
        random_species = self.species.keys()[0]
        self.dt = self.species[random_species].dt
        
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
            
            # Check wether CUDA is used and receive particle data
            # from the GPU
            if self.species.use_cuda:
                self.species.receive_particles_from_gpu()

            # Write the hdf5 files
            self.write_hdf5( iteration )

            # Send the particle data back to GPU after the
            # diagnostic step
            if self.species.use_cuda:
                self.species.send_particles_to_gpu()

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
        
        # Create the filename and open hdf5 file
        f = h5py.File( fullpath, mode="a" )
        # Set up its attributes            
        self.setup_openpmd_file( f, self.dt, iteration*self.dt )
        
        # Loop over the different species and 
        # particle quantities that should be written
        for species_name in self.species.keys() :

            # Select the species and the particles
            # that will be written
            species = self.species[species_name]
            select_array = self.apply_selection( species )
            N = select_array.sum()  # Counts the number of 'True'

            for particle_var in self.particle_data :
            	# Write the datasets for each particle datatype
                if particle_var in [ "position", "momentum" ] :
                    for coord in ["x", "y", "z"] :
                        quantity = "%s/%s" %(particle_var, coord)
                        path = "/particles/%s/%s" %(species_name, quantity)
                        self.write_dataset( f, path, species, quantity,
                                            N, select_array )
                elif particle_var == "weighting" :
                    quantity = "weighting"
                    path = "/particles/%s/%s" % (species_name, quantity )
                    self.write_dataset( f, path, species, quantity,
                                        N, select_array )
                else :
                    raise ValueError("Invalid string in %s of species %s" 
                    				 %(particle_var, particle_name))
        
        # Close the file   
        f.close()

    def apply_selection( self, species ) :
        """
        Apply the rules of self.select to determine which
        particles should be written

        Parameters
        ----------
        species : a Particles object

        Returns
        -------
        A 1d array of the same shape as that particle array
        containing True for the particles that satify all
        the rules of self.select
        """
        # Initialize an array filled with True
        select_array = np.ones( species.Ntot, dtype='bool' )

        # Apply the rules successively
        if self.select is not None :
            # Go through the quantities on which a rule applies
            for quantity in self.select.keys() :
                
                quantity_array = getattr( species, quantity )
                # Lower bound
                if self.select[quantity][0] is not None :
                    select_array = np.logical_and(
                        quantity_array > self.select[quantity][0],
                        select_array )
                # Upper bound
                if self.select[quantity][1] is not None :
                    select_array = np.logical_and(
                        quantity_array < self.select[quantity][1],
                        select_array )

        return( select_array )

    def write_dataset( self, f, path, species, quantity, N, select_array ) :
        """
        Write a given dataset
    
        Parameters
        ----------
        f : an h5py.File object
    
        path : string
            The path where to write the dataset, inside the file f

        quantity : string
            The quantity to be written, in the openPMD convention
            (e.g. 'position/x', 'momentum/z', 'weighting')
            
        N : int
        	Contains the global number of particles
            
        species : a Particles object
        	The species object to get the particle data from 

        select_array : 1darray of bool
            An array of the same shape as that particle array
            containing True for the particles that satify all
            the rules of self.select
        """
        # Create the dataset and setup its attributes
        datashape = (N, )
        dset = f.require_dataset( path, datashape, dtype='f')
        #self.setup_openpmd_dataset( dset, dz, dr, zmin, quantity )

        # Extract the select particle quantity
        quantity_array = self.get_dataset( species, quantity, select_array )

        # Write it
        dset[:] = quantity_array

    def get_dataset( self, species, quantity, select_array ) :
        """
        Extract the array that satisfies select_array
        
        species : a Particles object
        	The species object to get the particle data from 

        quantity : string
            The quantity to be written, in the openPMD convention
            (e.g. 'position/x', 'momentum/z', 'weighting')
            
        select_array : 1darray of bool
            An array of the same shape as that particle array
            containing True for the particles that satify all
            the rules of self.select
        """
        # Find the right name of the quantity in fbpic
        fbpic_dict = { 'position/x' : 'x',
                       'position/y' : 'y',
                       'position/z' : 'z',
                       'momentum/x' : 'ux',
                       'momentum/y' : 'uy',
                       'momentum/z' : 'uz',
                       'weighting' : 'w'}

        # Extract the quantity
        quantity_array = getattr( species, fbpic_dict[quantity] )

        # Apply the selection
        quantity_array = quantity_array[ select_array ]

        # Apply a conversion factor for the momenta
        if quantity in [ 'momentum/x', 'momentum/y', 'momentum/z'] :
            quantity_array = species.m * constants.c * quantity_array
        
        return( quantity_array )
    
