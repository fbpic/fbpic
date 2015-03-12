"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of classes for the output of the code.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pickling.PWpickle import PW

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
        # Get the directory in which to write the data
        if type(write_dir) is not string :
            self.write_dir = os.getcwd()
        else :
            self.write_dir = write_dir
        
        # Create a few addiditional directories within this
        self.create_dir('diags')
        if output_pickle :
            self.create_dir('diags/pickle')
        if output_png : 
            self.create_dir('diags/png')

    def write( self, iteration ) :

        # Check if the fields should be written at this iteration
        if iteration % self.period == 0 :
            
            # Write the png files if needed
            if self.pickle == True :
                for fieldtype in fieldtypes :
                    self.write_png( fieldtype, iteration )
    
            if self.pickle == True :
                self.write_pickle( iteration )
        
        # Copy the liveplots stuff
        
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
    
