import numpy as np

# Generic classes
# ---------------

class DensityProfile( object ):
    """
    Base class for all density profiles.
    Any new density profile should inherit from this class, and define its own
    __call__ method, using the same signature as the method below.
    Profiles that inherit from this base class can be summed,
    using the overloaded + operator.
    """
    def __init__( self ):
        """ 
        Initialize the density profile.
        (Each subclass should call this method at initialization.)

        Note:
        -----------
        Here we also register an empty argument string list `arg_str`. This list 
        should contain the strings representing the coordinates required for the 
        profile function evaluation, e.g. `["x", "y", "z"]` or `["r", "z"]`.

        In FBPIC, the available strings should be defined as a dictionary,
        ```agrDict = { "x": x, "y": y, "z": z, "th": theta, "r": r}```

        """
        self.arg_str = []


    def __call__( self, coords ):
        """ 
        Return the density value.

        Parameters
        -----------
        coords: tuple of ndarrays (meters)
            The positions at which to calculate the profile. Positins are in the 
            coordinate system defined by argument string variable.

        Returns:
        --------
        val: ndarray
            Array of the same shape as x, y, z, containing the density weighting
        """
        # The base class only defines dummy values
        # (This should be replaced by any class that inherits from this one.)
        return( np.zeros_like(coords[0]) )

    def __add__( self, other ):
        """
        Overload the + operations for density profiles
        """
        return( SummedDesnityProfile( self, other ) )


class SummedDesnityProfile( DensityProfile ):
    """ 
    Class that represents the sum of two instances of LaserProfile
    """
    def __init__( self, profile1, profile2 ):
        """
        Initialize the sum of two instances of DensityProfile
        Parameters
        -----------
        profile1, profile2: instances of DensityProfile
        """
        DensityProfile.__init__(self)

        # Register the profiles from which the sum should be calculated
        self.profile1 = profile1
        self.profile2 = profile2

    def __call__( self, coords ):
        """
        See the docstring of LaserProfile.E_field
        """
        val1 = self.profile1( coords )
        val2 = self.profile2( coords )
        return( val1+val2 )


class DensityProfileFromPolarGrid(DensityProfile):
    """ 
    Class that calculates the transvers density profile
    from the provided polar grid (2D array+axis).

    This method also applies the truncation of the azimuthal spectrum,
    to a given number `Nm_trunc`. Practically, and depending on the
    degree of asymmetry one try `2*Nm < Nm_trunc < 3*Nm`, and decrease
    in case of instability (for boosted-frame simulations).
    """
    def __init__(self, val, r_axis, Nm ):
        """ 
        Initialize the profile object.

        Parameters
        -----------
        val: ndarray
            2D map of the transverse profile of the shape (Nth, Nr)
        r: ndarray (meters)
            Radial axis coordinates (1D array)
        Nm: int
            Number of modes to retain in the interpolation

        """
        self.arg_str = ["z", "r", "th"]
        self._modes = np.arange(Nm)
        self._Nm = Nm
        self._fft_norm = 1./val.shape[0]
        self._rfft = np.fft.rfft(val, axis=0)
        self._r = r_axis

        self._harmonics = []
        for M in self._modes:
            self._harmonics.append( Interpolant1D(self._r,
              self._rfft[M]*self._fft_norm) )

    def __call__(self, z, r, th):

        """ 
        Return the density value

        Parameters
        -----------
        r, th, z: ndarrays (meters, radians, meters)
            The positions at which to calculate the profile

        Returns:
        --------
        vals: ndarray
            Array of the same shape as r, th, z, containing
            the weighting density
        """

        Np = r.size
        vals = np.zeros((Np,), dtype=np.double)

        exp_vals = np.exp( 1.j* self._modes[:,None] * th[None,:] )

        harmonics_p = np.zeros((self._Nm, Np), dtype=np.complex)
        for M in self._modes:
            harmonics_p[M,:] = self._harmonics[M](r)

        harmonics_p[1:] *= 2.0
        vals = (harmonics_p*exp_vals).sum(0).real

        return vals #*(vals>0)


class Interpolant1D:
    """Utility class to create lists of interpolants"""
    def __init__(self, coord, value):
        """Initialized with coordinates and values arrays"""
        self.coord = coord
        self.value = value
    def __call__(self, coord_p):
        """Called with the coordinate"""
        value_p = np.interp(coord_p, self.coord, self.value )
        return value_p
