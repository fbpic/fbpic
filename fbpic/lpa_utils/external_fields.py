# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
from numba import vectorize, float64
import numpy as np
# Check if CUDA is available, then import CUDA functions
from fbpic.cuda_utils import cuda_installed

class ExternalField( object ):

    def __init__(self, field_func, fieldtype, amplitude,
                 length_scale, species=None ):
        """
        Initialize an ExternalField object, so that the function
        `field_func` is called at each time step on the field `fieldtype`

        This object should be added to the list `external_fields`,
        which is an attribute of the Simulation object, so that the
        fields are applied at each timestep. (See the example below)

        The function `field_func` is automatically converted to a GPU
        function if needed, by using numba's ufunc feature.

        Parameters
        ----------
        field_func: callable
            Function of the form `field_func( F, x, y, z, t, amplitude,
            length_scale )` and which returns the modified field F'

            This function will be called at each timestep, with:

            - F: 1d array of shape (N_ptcl,), containing the field
              designated by fieldtype, gathered on the particles
            - x, y, z: 1d arrays of shape (N_ptcl), containing the
              positions of the particles
            - t: float, the time in the simulation
            - amplitude and length_scale: floats that can be used within
              the function expression

            **WARNING:** In the PIC loop, this function is called after
            the field gathering. Thus this function can potentially
            overwrite the fields that were gathered on the grid.
            To avoid this, use "return(F + external_field) " inside
            the definition of `field_func` instead of "return(external_field)"

            **WARNING:** Inside the definition of `field_func` please use
            the `math` module for mathematical functions, instead of numpy.
            This will allow the function to be compiled for GPU.

        fieldtype: string
            Specifies on which field `field_func` will be applied.
            Either 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'

        species: a Particles object, optionals
            The species on which the external field has to be applied.
            If no species is specified, the external field is applied
            to all particles.

        Example
        -------
        In order to define a magnetic undulator, polarized along y, with
        a field of 1 Tesla and a period of 1 cm :

        >>> def field_func( F, x, y, z, t , amplitude, length_scale ):
                return( F + amplitude * math.cos( 2*np.pi*z/length_scale ) )
        >>> sim.external_fields = [
                ExternalField( field_func, 'By', 1., 1.e-2 ) ]
        """
        # Check that fieldtype is a correct field
        if (fieldtype in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']) is False:
            raise ValueError()
        else:
            self.fieldtype = fieldtype

        # Register the arguments
        self.amplitude = amplitude
        self.length_scale = length_scale
        self.species = species

        # Compile the field_func for cpu and gpu
        signature = [ float64( float64, float64, float64,
                               float64, float64, float64, float64 ) ]
        cpu_compiler = vectorize( signature, target='cpu', nopython=True )
        self.cpu_func = cpu_compiler( field_func )
        if cuda_installed:
            gpu_compiler = vectorize( signature, target='cuda' )
            self.gpu_func = gpu_compiler( field_func )


    def apply_expression( self, ptcl, t ):
        """
        Apply the external field function to the particles

        This function is called at each timestep, after field gathering
        in the step function.

        Parameters
        ----------
        ptcl: a list a Particles objects
            The particles on which the external fields will be applied

        t: float (seconds)
            The time in the simulation
        """
        for species in ptcl:

            # If any species was specified at initialization,
            # apply the field only on this species
            if (self.species is None) or (species is self.species):

                # Only apply the field if there are macroparticles
                # in this species
                if species.Ntot > 0:

                    field = getattr( species, self.fieldtype )

                    if type( field ) is np.ndarray:
                        # Call the CPU function
                        self.cpu_func( field, species.x, species.y, species.z,
                              t, self.amplitude, self.length_scale, out=field )
                    else:
                        # Call the GPU function
                        self.gpu_func( field, species.x, species.y, species.z,
                              t, self.amplitude, self.length_scale, out=field )
