# Copyright 2020, FBPIC contributors
# Authors: Thomas Wilson, Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL

"""
This file defines the class CustomQuantity
"""
import numpy as np

class CustomQuantity(object):
    """
    Class that defines a custom particle quantity to be calculated
    """
    def __init__(self, name, function, arguments, dimensions=np.zeros(7) ):
        """
        Initialise a custom quantity to calculate.
        
        Parameters
        ----------
        name : str
            Name of the quantity, used for reference.
            
        function : func
            Function to compute the custom quantity.
            The function must accept a single argument, which may be assumed to
            take the form of a list of the relevant particle quantities for the
            computation. 
            For example, the formula for the angular momentum 
            component along z is (x*uy - y*ux), so a function of the form
            
            def lz_func(args):
            	return( args[0]*args[1] - args[2]*args[3] )            
            
            would be suitable.
            
        arguments : list of strings
            List of argument names to pass to the function.
            Following the above example, the corresponding argument list 
            would be ['x','uy','y','ux']
        
        dimensions : np.array of length 7, optional
            Dimensions of the final quantity. If omitted, the result is 
            assumed to be adimensional.
        """
        self.name = name
        self.function = function
        self.arguments = arguments
        self.dimensions = dimensions