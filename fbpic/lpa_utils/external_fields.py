import numba
from numba import cuda

class ExternalField( object ):

    def __init__(self, field_func, fieldtype, amplitude,
                 length_scale, species=None ):
        """
        Initialize an ExternalField object, so that the function
        `field_func` is called at each time step on the field `fieldtype`

        The function field_func is automatically converted to a GPU
        function if needed, by using numba's ufunc feature.
        
        Parameters
        ----------
        field_func: callable
            Function of the form
                field_func( F, x, y, z, t, amplitude, length_scale )

            This function will be called at each timestep, with:
            - F: 1d array of shape (N_ptcl,), containing the field
              designated by fieldtype, gathered on the particles
            - x, y, z: 1d arrays of shape (N_ptcl), containing the
              positions of the particles
            - t: float, the time in the simulation
            - amplitude and length_scale: floats that can be used within
              the function expression

            **WARNING:** In the PIC loop, this function is called after
            the gathering operation. Thus this function can potentially
            overwrite the fields that were gathered on the grid.
            To avoid this, use "F += " inside the definition of the
            function instead of "F = "

        fieldtype: string
            Specifies on which field `field_func` will be applied
            Either 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'

        species: a Particles object, optionals
            The species on which the external field has to be applied
            If no species is specified, the external field is applied
            to all particles

        Example
        -------
        In order to define a magnetic undulator, polarized along y, with
        a field of 1 Tesla and a period of 1 cm :
        >>> def field_func( F, x, y, z, t , amplitude, length_scale ):
                F += amplitude * np.cos( 2*np.pi*z/length_scale )
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
    
        # Compile the field_func for cpu and gpu
        self.cpu_func = numba.vectorize( field_func, nopython=True )
        if cuda.is_available():
            self.gpu_func = numba.vectorize(
                field_func, nopython=True, target='cuda' )


    def apply_expression( self, ptcl, t ):
        """
        
        """

        for species in ptcl:      

            # If any species was specified at initialization,
            # apply the field only on this species
            if (self.species is None) or (species is self.species):

                field = getattr( species, self.fieldtype )

                if type( field ) is np.ndarray:
                    # Call the CPU function
                    self.cpu_func( field, species.x, species.y, species.z,
                                   t, self.amplitude, self.length_scale )
                else:
                    # Call the GPU function
                    self.gpu_func( field, species.x, species.y, species.z,
                                   t, self.amplitude, self.length_scale )
