# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure and methods associated with the fields.
"""
import numpy as np
from scipy.constants import c, mu_0, epsilon_0
from .numba_methods import numba_push_eb_standard, numba_push_eb_comoving, \
    numba_correct_currents_standard, numba_correct_currents_comoving
from .spectral_transform import SpectralTransformer

# Check if CUDA is available, then import CUDA functions
from fbpic.cuda_utils import cuda_installed
if cuda_installed:
    from fbpic.cuda_utils import cuda_tpb_bpg_2d
    from .cuda_methods import cuda, \
    cuda_correct_currents_standard, cuda_correct_currents_comoving, \
    cuda_divide_scalar_by_volume, cuda_divide_vector_by_volume, \
    cuda_erase_scalar, cuda_erase_vector, \
    cuda_filter_scalar, cuda_filter_vector, \
    cuda_push_eb_standard, cuda_push_eb_comoving, cuda_push_rho

class Fields(object) :
    """
    Class that contains the fields data of the simulation

    Methods
    -------
    - push : Advances the fields over one timestep
    - interp2spect : Transforms the fields from the
           interpolation grid to the spectral grid
    - spect2interp : Transforms the fields from the
           spectral grid to the interpolation grid
    - correct_currents : Corrects the currents so that
           they satisfy the conservation equation
    - erase : sets the fields to zero on the interpolation grid
    - divide_by_volume : divide the fields by the cell volume

    Main attributes
    ----------
    All the following attributes are lists,
    with one element per azimuthal mode
    - interp : a list of InterpolationGrid objects
        Contains the field data on the interpolation grid
    - spect : a list of SpectralGrid objects
        Contains the field data on the spectral grid
    - trans : a list of SpectralTransformer objects
        Allows to transform back and forth between the
        interpolation and spectral grid
    - psatd : a list of PsatdCoeffs
        Contains the coefficients to solve the Maxwell equations
    """
    def __init__( self, Nz, zmax, Nr, rmax, Nm, dt, zmin=0.,
                  n_order=-1, v_comoving=None,
                  use_galilean=True, use_cuda=False ) :
        """
        Initialize the components of the Fields object

        Parameters
        ----------
        Nz : int
            The number of gridpoints in z

        zmin, zmax : float (zmin, optional)
            The initial position of the left and right
            edge of the box along z

        Nr : int
            The number of gridpoints in r

        rmax : float
            The position of the edge of the box along r

        Nm : int
            The number of azimuthal modes

        dt : float
            The timestep of the simulation, required for the
            coefficients of the psatd scheme

        v_comoving: float or None, optional
            If this variable is None, the standard PSATD is used (default).
            Otherwise, the current is assumed to be "comoving",
            i.e. constant with respect to (z - v_comoving * t).
            This can be done in two ways: either by
            - Using a PSATD scheme that takes this hypothesis into account
            - Solving the PSATD scheme in a Galilean frame

        use_galilean: bool, optional
            Determines which one of the two above schemes is used
            When use_galilean is true, the whole grid moves
            with a speed v_comoving

        n_order : int, optional
           The order of the stencil for the z derivatives
           Use -1 for infinite order
           Otherwise use a positive, even number. In this case
           the stencil extends up to n_order/2 cells on each side.

        use_cuda : bool, optional
            Wether to use the GPU or not
        """
        # Register the arguments inside the object
        self.Nz = Nz
        self.Nr = Nr
        self.rmax = rmax
        self.Nm = Nm
        self.dt = dt
        self.n_order = n_order
        self.v_comoving = v_comoving
        self.use_galilean = use_galilean

        # Define wether or not to use the GPU
        self.use_cuda = use_cuda
        if (self.use_cuda==True) and (cuda_installed==False) :
            print('*** Cuda not available for the fields.')
            print('*** Performing the field operations on the CPU.')
            self.use_cuda = False

        # Infer the values of the z and kz grid
        dz = (zmax-zmin)/Nz
        z = dz * ( np.arange( 0, Nz ) + 0.5 ) + zmin

        # Create the list of the transformers, which convert the fields
        # back and forth between the spatial and spectral grid
        # (one object per azimuthal mode)
        self.trans = []
        for m in range(Nm) :
            self.trans.append( SpectralTransformer(
                Nz, Nr, m, rmax, use_cuda=self.use_cuda ) )

        # Create the interpolation grid for each modes
        # (one grid per azimuthal mode)
        self.interp = [ ]
        for m in range(Nm) :
            # Extract the radial grid for mode m
            r = self.trans[m].dht0.get_r()
            # Create the object
            self.interp.append( InterpolationGrid( z, r, m,
                                        use_cuda=self.use_cuda ) )

        # Get the kz and (finite-order) modified kz arrays
        # (According to FFT conventions, the kz array starts with
        # positive frequencies and ends with negative frequency.)
        kz_true = 2*np.pi* np.fft.fftfreq( Nz, dz )
        kz_modified = get_modified_k( kz_true, n_order, dz )

        # Create the spectral grid for each mode, as well as
        # the psatd coefficients
        # (one grid per azimuthal mode)
        self.spect = [ ]
        self.psatd = [ ]
        for m in range(Nm) :
            # Extract the inhomogeneous spectral grid for mode m
            kr = 2*np.pi * self.trans[m].dht0.get_nu()
            # Create the object
            self.spect.append( SpectralGrid( kz_modified, kr, m,
                kz_true, self.interp[m].dz, self.interp[m].dr,
                use_cuda=self.use_cuda ) )
            self.psatd.append( PsatdCoeffs( self.spect[m].kz,
                                self.spect[m].kr, m, dt, Nz, Nr,
                                V=self.v_comoving,
                                use_galilean=self.use_galilean,
                                use_cuda=self.use_cuda ) )

        # Initialize the needed prefix sum array for sorting
        if self.use_cuda:
            # Shift in the indices, induced by the moving window
            self.prefix_sum_shift = 0

    def send_fields_to_gpu( self ):
        """
        Copy the fields to the GPU.

        After this function is called, the array attributes of the
        interpolation and spectral grids point to GPU arrays
        """
        if self.use_cuda:
            for m in range(self.Nm) :
                self.interp[m].send_fields_to_gpu()
                self.spect[m].send_fields_to_gpu()

    def receive_fields_from_gpu( self ):
        """
        Receive the fields from the GPU.

        After this function is called, the array attributes of the
        interpolation and spectral grids are accessible by the CPU again.
        """
        if self.use_cuda:
            for m in range(self.Nm) :
                self.interp[m].receive_fields_from_gpu()
                self.spect[m].receive_fields_from_gpu()

    def push(self, use_true_rho=False) :
        """
        Push the different azimuthal modes over one timestep,
        in spectral space.

        use_true_rho : bool, optional
            Whether to use the rho projected on the grid.
            If set to False, this will use div(E) and div(J)
            to evaluate rho and its time evolution.
            In the case use_true_rho==False, the rho projected
            on the grid is used only to correct the currents, and
            the simulation can be run without the neutralizing ions.
        """
        # Push each azimuthal grid individually, by passing the
        # corresponding psatd coefficients
        for m in range(self.Nm) :
            self.spect[m].push_eb_with( self.psatd[m], use_true_rho )
            self.spect[m].push_rho()

    def correct_currents(self) :
        """
        Correct the currents so that they satisfy the
        charge conservation equation
        """
        # Correct each azimuthal grid individually
        for m in range(self.Nm) :
            self.spect[m].correct_currents( self.dt, self.psatd[m] )

    def correct_divE(self) :
        """
        Correct the currents so that they satisfy the
        charge conservation equation
        """
        # Correct each azimuthal grid individually
        for m in range(self.Nm) :
            self.spect[m].correct_divE()

    def interp2spect(self, fieldtype) :
        """
        Transform the fields `fieldtype` from the interpolation
        grid to the spectral grid

        Parameter
        ---------
        fieldtype :
            A string which represents the kind of field to transform
            (either 'E', 'B', 'J', 'rho_next', 'rho_prev')
        """
        # Use the appropriate transformation depending on the fieldtype.
        if fieldtype == 'E' :
            for m in range(self.Nm) :
            # Transform each azimuthal grid individually
                self.trans[m].interp2spect_scal(
                    self.interp[m].Ez, self.spect[m].Ez )
                self.trans[m].interp2spect_vect(
                    self.interp[m].Er, self.interp[m].Et,
                    self.spect[m].Ep, self.spect[m].Em )
        elif fieldtype == 'B' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].interp2spect_scal(
                    self.interp[m].Bz, self.spect[m].Bz )
                self.trans[m].interp2spect_vect(
                    self.interp[m].Br, self.interp[m].Bt,
                    self.spect[m].Bp, self.spect[m].Bm )
        elif fieldtype == 'J' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].interp2spect_scal(
                    self.interp[m].Jz, self.spect[m].Jz )
                self.trans[m].interp2spect_vect(
                    self.interp[m].Jr, self.interp[m].Jt,
                    self.spect[m].Jp, self.spect[m].Jm )
        elif fieldtype == 'rho_next' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].interp2spect_scal(
                    self.interp[m].rho, self.spect[m].rho_next )
        elif fieldtype == 'rho_prev' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].interp2spect_scal(
                    self.interp[m].rho, self.spect[m].rho_prev )
        else :
            raise ValueError( 'Invalid string for fieldtype: %s' %fieldtype )

    def spect2interp(self, fieldtype) :
        """
        Transform the fields `fieldtype` from the spectral grid
        to the spectral grid

        Parameter
        ---------
        fieldtype :
            A string which represents the kind of field to transform
            (either 'E', 'B', 'J', 'rho')
        """
        # Use the appropriate transformation depending on the fieldtype.
        if fieldtype == 'E' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].spect2interp_scal(
                    self.spect[m].Ez, self.interp[m].Ez )
                self.trans[m].spect2interp_vect(
                    self.spect[m].Ep,  self.spect[m].Em,
                    self.interp[m].Er, self.interp[m].Et )
        elif fieldtype == 'B' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].spect2interp_scal(
                    self.spect[m].Bz, self.interp[m].Bz )
                self.trans[m].spect2interp_vect(
                    self.spect[m].Bp, self.spect[m].Bm,
                    self.interp[m].Br, self.interp[m].Bt )
        elif fieldtype == 'J' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].spect2interp_scal(
                    self.spect[m].Jz, self.interp[m].Jz )
                self.trans[m].spect2interp_vect(
                    self.spect[m].Jp,  self.spect[m].Jm,
                    self.interp[m].Jr, self.interp[m].Jt )
        elif fieldtype == 'rho' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].spect2interp_scal(
                    self.spect[m].rho_next, self.interp[m].rho )
        else :
            raise ValueError( 'Invalid string for fieldtype: %s' %fieldtype )

    def erase(self, fieldtype ) :
        """
        Sets the field `fieldtype` to zero on the interpolation grid

        Parameter
        ---------
        fieldtype : string
            A string which represents the kind of field to be erased
            (either 'E', 'B', 'J', 'rho')
        """
        if self.use_cuda :
            # Obtain the cuda grid
            dim_grid, dim_block = cuda_tpb_bpg_2d( self.Nz, self.Nr )

            # Erase the arrays on the GPU
            if fieldtype == 'rho' :
                cuda_erase_scalar[dim_grid, dim_block](
                    self.interp[0].rho, self.interp[1].rho )
            elif fieldtype == 'J' :
                cuda_erase_vector[dim_grid, dim_block](
                    self.interp[0].Jr, self.interp[1].Jr,
                    self.interp[0].Jt, self.interp[1].Jt,
                    self.interp[0].Jz, self.interp[1].Jz )
            elif fieldtype == 'E' :
                cuda_erase_vector[dim_grid, dim_block](
                    self.interp[0].Er, self.interp[1].Er,
                    self.interp[0].Et, self.interp[1].Et,
                    self.interp[0].Ez, self.interp[1].Ez )
            elif fieldtype == 'B' :
                cuda_erase_vector[dim_grid, dim_block](
                    self.interp[0].Br, self.interp[1].Br,
                    self.interp[0].Bt, self.interp[1].Bt,
                    self.interp[0].Bz, self.interp[1].Bz )
            else :
                raise ValueError('Invalid string for fieldtype: %s'%fieldtype)
        else :
            # Erase the arrays on the CPU
            if fieldtype == 'rho' :
                for m in range(self.Nm) :
                    self.interp[m].rho[:,:] = 0.
            elif fieldtype == 'J' :
                for m in range(self.Nm) :
                    self.interp[m].Jr[:,:] = 0.
                    self.interp[m].Jt[:,:] = 0.
                    self.interp[m].Jz[:,:] = 0.
            elif fieldtype == 'E' :
                for m in range(self.Nm) :
                    self.interp[m].Er[:,:] = 0.
                    self.interp[m].Et[:,:] = 0.
                    self.interp[m].Ez[:,:] = 0.
            elif fieldtype == 'B' :
                for m in range(self.Nm) :
                    self.interp[m].Br[:,:] = 0.
                    self.interp[m].Bt[:,:] = 0.
                    self.interp[m].Bz[:,:] = 0.
            else :
                raise ValueError('Invalid string for fieldtype: %s'%fieldtype)

    def filter_spect( self, fieldtype ) :
        """
        Filter the field `fieldtype` on the spectral grid

        Parameter
        ---------
        fieldtype : string
            A string which represents the kind of field to be filtered
            (either 'E', 'B', 'J', 'rho_next' or 'rho_prev')
        """
        for m in range(self.Nm) :
            self.spect[m].filter( fieldtype )

    def divide_by_volume( self, fieldtype ) :
        """
        Divide the field `fieldtype` in each cell by the cell volume,
        on the interpolation grid.

        This is typically done for rho and J, after the charge and
        current deposition.

        Parameter
        ---------
        fieldtype :
            A string which represents the kind of field to be erased
            (either 'rho' or 'J')
        """
        if self.use_cuda :
            # Perform division on the GPU
            dim_grid, dim_block = cuda_tpb_bpg_2d( self.Nz, self.Nr )

            if fieldtype == 'rho' :
                cuda_divide_scalar_by_volume[dim_grid, dim_block](
                    self.interp[0].rho, self.interp[1].rho,
                    self.interp[0].d_invvol, self.interp[1].d_invvol )
            elif fieldtype == 'J' :
                cuda_divide_vector_by_volume[dim_grid, dim_block](
                    self.interp[0].Jr, self.interp[1].Jr,
                    self.interp[0].Jt, self.interp[1].Jt,
                    self.interp[0].Jz, self.interp[1].Jz,
                    self.interp[0].d_invvol, self.interp[1].d_invvol )
            else :
                raise ValueError('Invalid string for fieldtype: %s'%fieldtype)
        else :
            # Perform division on the CPU
            if fieldtype == 'rho' :
                for m in range(self.Nm) :
                    self.interp[m].rho = \
                    self.interp[m].rho * self.interp[m].invvol[np.newaxis,:]
            elif fieldtype == 'J' :
                for m in range(self.Nm) :
                    self.interp[m].Jr = \
                    self.interp[m].Jr * self.interp[m].invvol[np.newaxis,:]
                    self.interp[m].Jt = \
                    self.interp[m].Jt * self.interp[m].invvol[np.newaxis,:]
                    self.interp[m].Jz = \
                    self.interp[m].Jz * self.interp[m].invvol[np.newaxis,:]
            else :
                raise ValueError('Invalid string for fieldtype: %s'%fieldtype)


class InterpolationGrid(object) :
    """
    Contains the fields and coordinates of the spatial grid.

    Main attributes :
    - z,r : 1darrays containing the positions of the grid
    - Er, Et, Ez, Br, Bt, Bz, Jr, Jt, Jz, rho :
      2darrays containing the fields.
    """

    def __init__(self, z, r, m, use_cuda=False ) :
        """
        Allocates the matrices corresponding to the spatial grid

        Parameters
        ----------
        z : 1darray of float
            The positions of the longitudinal, spatial grid

        r : 1darray of float
            The positions of the radial, spatial grid

        m : int
            The index of the mode

        use_cuda : bool, optional
            Wether to use the GPU or not
        """

        # Register the arrays and their length
        Nz = len(z)
        Nr = len(r)
        self.Nz = Nz
        self.z = z.copy()
        self.Nr = Nr
        self.r = r.copy()
        self.m = m

        # Register a few grid properties
        dr = r[1] - r[0]
        dz = z[1] - z[0]
        self.dr = dr
        self.dz = dz
        self.invdr = 1./dr
        self.invdz = 1./dz
        # rmin, rmax, zmin, zmax correspond to the edge of cells
        self.rmin = self.r.min() - 0.5*dr
        self.rmax = self.r.max() + 0.5*dr
        self.zmin = self.z.min() - 0.5*dz
        self.zmax = self.z.max() + 0.5*dz
        # Cell volume (assuming an evenly-spaced grid)
        vol = np.pi*dz*( (r+0.5*dr)**2 - (r-0.5*dr)**2 )
        # NB : No Verboncoeur-type correction required
        self.invvol = 1./vol

        # Allocate the fields arrays
        self.Er = np.zeros( (Nz, Nr), dtype='complex' )
        self.Et = np.zeros( (Nz, Nr), dtype='complex' )
        self.Ez = np.zeros( (Nz, Nr), dtype='complex' )
        self.Br = np.zeros( (Nz, Nr), dtype='complex' )
        self.Bt = np.zeros( (Nz, Nr), dtype='complex' )
        self.Bz = np.zeros( (Nz, Nr), dtype='complex' )
        self.Jr = np.zeros( (Nz, Nr), dtype='complex' )
        self.Jt = np.zeros( (Nz, Nr), dtype='complex' )
        self.Jz = np.zeros( (Nz, Nr), dtype='complex' )
        self.rho = np.zeros( (Nz, Nr), dtype='complex' )

        # Check whether the GPU should be used
        self.use_cuda = use_cuda

        # Replace the invvol array by an array on the GPU, when using cuda
        if self.use_cuda :
            self.d_invvol = cuda.to_device( self.invvol )

    def send_fields_to_gpu( self ):
        """
        Copy the fields to the GPU.

        After this function is called, the array attributes
        point to GPU arrays.
        """
        self.Er = cuda.to_device( self.Er )
        self.Et = cuda.to_device( self.Et )
        self.Ez = cuda.to_device( self.Ez )
        self.Br = cuda.to_device( self.Br )
        self.Bt = cuda.to_device( self.Bt )
        self.Bz = cuda.to_device( self.Bz )
        self.Jr = cuda.to_device( self.Jr )
        self.Jt = cuda.to_device( self.Jt )
        self.Jz = cuda.to_device( self.Jz )
        self.rho = cuda.to_device( self.rho )

    def receive_fields_from_gpu( self ):
        """
        Receive the fields from the GPU.

        After this function is called, the array attributes
        are accessible by the CPU again.
        """
        self.Er = self.Er.copy_to_host()
        self.Et = self.Et.copy_to_host()
        self.Ez = self.Ez.copy_to_host()
        self.Br = self.Br.copy_to_host()
        self.Bt = self.Bt.copy_to_host()
        self.Bz = self.Bz.copy_to_host()
        self.Jr = self.Jr.copy_to_host()
        self.Jt = self.Jt.copy_to_host()
        self.Jz = self.Jz.copy_to_host()
        self.rho = self.rho.copy_to_host()

    def show(self, fieldtype, below_axis=True, scale=1,
             gridscale=1.e-6, **kw) :
        """
        Show the field `fieldtype` on the interpolation grid

        Parameters
        ----------
        fieldtype : string
            Name of the field to be plotted.
            (either 'Er', 'Et', 'Ez', 'Br', 'Bt', 'Bz',
            'Jr', 'Jt', 'Jz', 'rho')

        scale : float, optional
            Value by which the field should be divided before plotting

        gridscale : float, optional
            Value by which to scale the z and r axis
            Default : scale it in microns

        kw : dictionary
            Options to be passed to matplotlib's imshow
        """
        # matplotlib only needs to be imported if this function is called
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
        'The package `matplotlib` is required to show the fields.'
        '\nPlease install it: `conda install matplotlib`')

        # Select the field to plot
        plotted_field = getattr( self, fieldtype)
        # Show the field also below the axis for a more realistic picture
        if below_axis == True :
            plotted_field = np.hstack( (plotted_field[:,::-1],plotted_field) )
            extent = np.array([self.zmin, self.zmax, -self.rmax, self.rmax])
        else :
            extent = np.array([self.zmin, self.zmax, self.rmin, self.rmax])
        extent = extent/gridscale
        # Title
        plt.suptitle(
            '%s on the interpolation grid, for mode %d' %(fieldtype, self.m) )

        # Plot the real part
        plt.subplot(211)
        plt.imshow( plotted_field.real.T[::-1]/scale, aspect='auto',
                    interpolation='nearest', extent = extent, **kw )
        plt.xlabel('z')
        plt.ylabel('r')
        cb = plt.colorbar()
        cb.set_label('Real part')

        # Plot the imaginary part
        plt.subplot(212)
        plt.imshow( plotted_field.imag.T[::-1]/scale, aspect='auto',
                    interpolation='nearest', extent = extent, **kw )
        plt.xlabel('z')
        plt.ylabel('r')
        cb = plt.colorbar()
        cb.set_label('Imaginary part')

class SpectralGrid(object) :
    """
    Contains the fields and coordinates of the spectral grid.
    """

    def __init__(self, kz_modified, kr, m, kz_true, dz, dr, use_cuda=False ) :
        """
        Allocates the matrices corresponding to the spectral grid

        Parameters
        ----------
        kz_modified : 1darray of float
            The modified wavevectors of the longitudinal, spectral grid
            (Different then kz_true in the case of a finite-stencil)

        kr : 1darray of float
            The wavevectors of the radial, spectral grid

        m : int
            The index of the mode

        kz_true : 1darray of float
            The true wavevector of the longitudinal, spectral grid
            (The actual kz that a Fourier transform would give)

        dz, dr: float
            The grid spacings (needed to calculate
            precisely the filtering function in spectral space)

        use_cuda : bool, optional
            Wether to use the GPU or not
        """
        # Register the arrays and their length
        Nz = len(kz_modified)
        Nr = len(kr)
        self.Nr = Nr
        self.Nz = Nz
        self.m = m

        # Find the limits of the grid (useful for plotting ; use the true kz)
        self.kzmin = kz_true.min()
        self.kzmax = kz_true.max()
        self.krmin = kr.min()
        self.krmax = kr.max()

        # Allocate the fields arrays
        self.Ep = np.zeros( (Nz, Nr), dtype='complex' )
        self.Em = np.zeros( (Nz, Nr), dtype='complex' )
        self.Ez = np.zeros( (Nz, Nr), dtype='complex' )
        self.Bp = np.zeros( (Nz, Nr), dtype='complex' )
        self.Bm = np.zeros( (Nz, Nr), dtype='complex' )
        self.Bz = np.zeros( (Nz, Nr), dtype='complex' )
        self.Jp = np.zeros( (Nz, Nr), dtype='complex' )
        self.Jm = np.zeros( (Nz, Nr), dtype='complex' )
        self.Jz = np.zeros( (Nz, Nr), dtype='complex' )
        self.rho_prev = np.zeros( (Nz, Nr), dtype='complex' )
        self.rho_next = np.zeros( (Nz, Nr), dtype='complex' )

        # Auxiliary arrays
        # - for the field solve
        #   (use the modified kz, since this corresponds to the stencil)
        self.kz, self.kr = np.meshgrid( kz_modified, kr, indexing='ij' )
        # - for filtering
        #   (use the true kz, so as to effectively filter the high k's)
        self.filter_array = get_filter_array( kz_true, kr, dz, dr )
        # - for current correction
        self.F = np.zeros( (Nz, Nr), dtype='complex' )
        self.inv_k2 = 1./np.where( ( self.kz == 0 ) & (self.kr == 0),
                                   1., self.kz**2 + self.kr**2 )
        self.inv_k2[ ( self.kz == 0 ) & (self.kr == 0) ] = 0.

        # Check whether to use the GPU
        self.use_cuda = use_cuda

        # Transfer the auxiliary arrays on the GPU
        if self.use_cuda :
            self.d_filter_array = cuda.to_device( self.filter_array )
            self.d_inv_k2 = cuda.to_device( self.inv_k2 )
            self.d_kz = cuda.to_device( self.kz )
            self.d_kr = cuda.to_device( self.kr )
            # NB: F is not needed on the GPU (on-the-fly variables)

    def send_fields_to_gpu( self ):
        """
        Copy the fields to the GPU.

        After this function is called, the array attributes
        point to GPU arrays.
        """
        self.Ep = cuda.to_device( self.Ep )
        self.Em = cuda.to_device( self.Em )
        self.Ez = cuda.to_device( self.Ez )
        self.Bp = cuda.to_device( self.Bp )
        self.Bm = cuda.to_device( self.Bm )
        self.Bz = cuda.to_device( self.Bz )
        self.Jp = cuda.to_device( self.Jp )
        self.Jm = cuda.to_device( self.Jm )
        self.Jz = cuda.to_device( self.Jz )
        self.rho_prev = cuda.to_device( self.rho_prev )
        self.rho_next = cuda.to_device( self.rho_next )

    def receive_fields_from_gpu( self ):
        """
        Receive the fields from the GPU.

        After this function is called, the array attributes
        are accessible by the CPU again.
        """
        self.Ep = self.Ep.copy_to_host()
        self.Em = self.Em.copy_to_host()
        self.Ez = self.Ez.copy_to_host()
        self.Bp = self.Bp.copy_to_host()
        self.Bm = self.Bm.copy_to_host()
        self.Bz = self.Bz.copy_to_host()
        self.Jp = self.Jp.copy_to_host()
        self.Jm = self.Jm.copy_to_host()
        self.Jz = self.Jz.copy_to_host()
        self.rho_prev = self.rho_prev.copy_to_host()
        self.rho_next = self.rho_next.copy_to_host()

    def correct_currents (self, dt, ps) :
        """
        Correct the currents so that they satisfy the
        charge conservation equation

        Parameters
        ----------
        dt : float
            Timestep of the simulation
        """
        # Precalculate useful coefficient
        inv_dt = 1./dt

        if self.use_cuda :
            # Obtain the cuda grid
            dim_grid, dim_block = cuda_tpb_bpg_2d( self.Nz, self.Nr)
            # Correct the currents on the GPU
            if ps.V is None:
                # With standard PSATD algorithm
                cuda_correct_currents_standard[dim_grid, dim_block](
                    self.rho_prev, self.rho_next, self.Jp, self.Jm, self.Jz,
                    self.d_kz, self.d_kr, self.d_inv_k2,
                    inv_dt, self.Nz, self.Nr )
            else:
                # With Galilean/comoving algorithm
                cuda_correct_currents_comoving[dim_grid, dim_block](
                    self.rho_prev, self.rho_next, self.Jp, self.Jm, self.Jz,
                    self.d_kz, self.d_kr, self.d_inv_k2,
                    ps.d_j_corr_coef, ps.d_T_eb, ps.d_T_cc,
                    inv_dt, self.Nz, self.Nr)
        else :
            # Correct the currents on the CPU
            if ps.V is None:
                # With standard PSATD algorithm
                numba_correct_currents_standard(
                    self.rho_prev, self.rho_next, self.Jp, self.Jm, self.Jz,
                    self.kz, self.kr, self.inv_k2, inv_dt, self.Nz, self.Nr )
            else:
                # With Galilean/comoving algorithm
                numba_correct_currents_comoving(
                    self.rho_prev, self.rho_next, self.Jp, self.Jm, self.Jz,
                    self.kz, self.kr, self.inv_k2,
                    ps.j_corr_coef, ps.T_eb, ps.T_cc,
                    inv_dt, self.Nz, self.Nr)

    def correct_divE(self) :
        """
        Correct the electric field, so that it satisfies the equation
        div(E) - rho/epsilon_0 = 0
        """
        # Correct div(E) on the CPU

        # Calculate the intermediate variable F
        self.F[:,:] = - self.inv_k2 * (
            - self.rho_prev/epsilon_0 \
            + 1.j*self.kz*self.Ez + self.kr*( self.Ep - self.Em ) )

        # Correct the current accordingly
        self.Ep += 0.5*self.kr*self.F
        self.Em += -0.5*self.kr*self.F
        self.Ez += -1.j*self.kz*self.F


    def push_eb_with(self, ps, use_true_rho=False ) :
        """
        Push the fields over one timestep, using the psatd coefficients.

        Parameters
        ----------
        ps : PsatdCoeffs object
            psatd object corresponding to the same m mode

        use_true_rho : bool, optional
            Whether to use the rho projected on the grid.
            If set to False, this will use div(E) and div(J)
            to evaluate rho and its time evolution.
            In the case use_true_rho==False, the rho projected
            on the grid is used only to correct the currents, and
            the simulation can be run without the neutralizing ions.
        """
        # Check that psatd object passed as argument is the right one
        # (i.e. corresponds to the right mode)
        assert( self.m == ps.m )

        if self.use_cuda :
            # Obtain the cuda grid
            dim_grid, dim_block = cuda_tpb_bpg_2d( self.Nz, self.Nr)
            # Push the fields on the GPU
            if ps.V is None:
                # With the standard PSATD algorithm
                cuda_push_eb_standard[dim_grid, dim_block](
                    self.Ep, self.Em, self.Ez, self.Bp, self.Bm, self.Bz,
                    self.Jp, self.Jm, self.Jz, self.rho_prev, self.rho_next,
                    ps.d_rho_prev_coef, ps.d_rho_next_coef, ps.d_j_coef,
                    ps.d_C, ps.d_S_w, self.d_kr, self.d_kz, ps.dt,
                    use_true_rho, self.Nz, self.Nr )
            else:
                # With the Galilean/comoving algorithm
                cuda_push_eb_comoving[dim_grid, dim_block](
                    self.Ep, self.Em, self.Ez, self.Bp, self.Bm, self.Bz,
                    self.Jp, self.Jm, self.Jz, self.rho_prev, self.rho_next,
                    ps.d_rho_prev_coef, ps.d_rho_next_coef, ps.d_j_coef,
                    ps.d_C, ps.d_S_w, ps.d_T_eb, ps.d_T_cc, ps.d_T_rho,
                    self.d_kr, self.d_kz, ps.dt, ps.V,
                    use_true_rho, self.Nz, self.Nr )
        else :
            # Push the fields on the CPU
            if ps.V is None:
                # With the standard PSATD algorithm
                numba_push_eb_standard(
                    self.Ep, self.Em, self.Ez, self.Bp, self.Bm, self.Bz,
                    self.Jp, self.Jm, self.Jz, self.rho_prev, self.rho_next,
                    ps.rho_prev_coef, ps.rho_next_coef, ps.j_coef,
                    ps.C, ps.S_w, self.kr, self.kz, ps.dt,
                    use_true_rho, self.Nz, self.Nr )
            else:
                # With the Galilean/comoving algorithm
                numba_push_eb_comoving(
                    self.Ep, self.Em, self.Ez, self.Bp, self.Bm, self.Bz,
                    self.Jp, self.Jm, self.Jz, self.rho_prev, self.rho_next,
                    ps.rho_prev_coef, ps.rho_next_coef, ps.j_coef,
                    ps.C, ps.S_w, ps.T_eb, ps.T_cc, ps.T_rho,
                    self.kr, self.kz, ps.dt, ps.V,
                    use_true_rho, self.Nz, self.Nr )

    def push_rho(self) :
        """
        Transfer the values of rho_next to rho_prev,
        and set rho_next to zero
        """
        if self.use_cuda :
            # Obtain the cuda grid
            dim_grid, dim_block = cuda_tpb_bpg_2d( self.Nz, self.Nr)
            # Push the fields on the GPU
            cuda_push_rho[dim_grid, dim_block](
                self.rho_prev, self.rho_next, self.Nz, self.Nr )
        else :
            # Push the fields on the CPU
            self.rho_prev[:,:] = self.rho_next[:,:]
            self.rho_next[:,:] = 0.

    def filter(self, fieldtype) :
        """
        Filter the field `fieldtype`

        Parameter
        ---------
        fieldtype : string
            A string which represents the kind of field to be filtered
            (either 'E', 'B', 'J', 'rho_next' or 'rho_prev')
        """
        if self.use_cuda :
            # Obtain the cuda grid
            dim_grid, dim_block = cuda_tpb_bpg_2d( self.Nz, self.Nr)
            # Filter fields on the GPU
            if fieldtype == 'rho_prev' :
                cuda_filter_scalar[dim_grid, dim_block](
                    self.rho_prev, self.d_filter_array, self.Nz, self.Nr )
            elif fieldtype == 'rho_next' :
                cuda_filter_scalar[dim_grid, dim_block](
                    self.rho_next, self.d_filter_array, self.Nz, self.Nr )
            elif fieldtype == 'J' :
                cuda_filter_vector[dim_grid, dim_block]( self.Jp, self.Jm,
                        self.Jz, self.d_filter_array, self.Nz, self.Nr)
            elif fieldtype == 'E' :
                cuda_filter_vector[dim_grid, dim_block]( self.Ep, self.Em,
                        self.Ez, self.d_filter_array, self.Nz, self.Nr)
            elif fieldtype == 'B' :
                cuda_filter_vector[dim_grid, dim_block]( self.Bp, self.Bm,
                        self.Bz, self.d_filter_array, self.Nz, self.Nr)
            else :
                raise ValueError('Invalid string for fieldtype: %s'%fieldtype)
        else :
            # Filter fields on the CPU

            if fieldtype == 'rho_prev' :
                self.rho_prev = self.rho_prev * self.filter_array
            elif fieldtype == 'rho_next' :
                self.rho_next = self.rho_next * self.filter_array
            elif fieldtype == 'J' :
                self.Jp = self.Jp * self.filter_array
                self.Jm = self.Jm * self.filter_array
                self.Jz = self.Jz * self.filter_array
            elif fieldtype == 'E' :
                self.Ep = self.Ep * self.filter_array
                self.Em = self.Em * self.filter_array
                self.Ez = self.Ez * self.filter_array
            elif fieldtype == 'B' :
                self.Bp = self.Bp * self.filter_array
                self.Bm = self.Bm * self.filter_array
                self.Bz = self.Bz * self.filter_array
            else :
                raise ValueError('Invalid string for fieldtype: %s'%fieldtype)

    def show(self, fieldtype, below_axis=True, scale=1, **kw) :
        """
        Show the field `fieldtype` on the spectral grid

        Parameters
        ----------
        fieldtype : string
            Name of the field to be plotted.
            (either 'Ep', 'Em', 'Ez', 'Bp', 'Bm', 'Bz',
            'Jp', 'Jm', 'Jz', 'rho_prev', 'rho_next')

        scale : float
            Value by which the field should be divide before plotting

        kw : dictionary
            Options to be passed to matplotlib's imshow
        """
        # matplotlib only needs to be imported if this function is called
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
        'The package `matplotlib` is required to show the fields.'
        '\nPlease install it: `conda install matplotlib`')

        # Select the field to plot
        plotted_field = getattr( self, fieldtype)
        # Fold it so as to center the 0 frequency
        plotted_field = np.fft.fftshift( plotted_field, axes=0 )
        if below_axis == True :
            plotted_field = np.hstack((plotted_field[:,::-1], plotted_field))
            extent = [ self.kzmin, self.kzmax, -self.krmax, self.krmax ]
        else :
            extent = [ self.kzmin, self.kzmax, self.krmin, self.krmax ]
        # Title
        plt.suptitle(
            '%s on the spectral grid, for mode %d' %(fieldtype, self.m) )

        # Plot the real part
        plt.subplot(211)
        plt.imshow( plotted_field.real.T[::-1]/scale, aspect='auto',
                    interpolation='nearest', extent = extent, **kw )
        plt.xlabel('kz')
        plt.ylabel('kr')
        cb = plt.colorbar()
        cb.set_label('Real part')

        # Plot the imaginary part
        plt.subplot(212)
        plt.imshow( plotted_field.imag.T[::-1]/scale, aspect='auto',
                    interpolation='nearest', extent = extent, **kw )
        plt.xlabel('kz')
        plt.ylabel('kr')
        cb = plt.colorbar()
        cb.set_label('Imaginary part')

class PsatdCoeffs(object) :
    """
    Contains the coefficients of the PSATD scheme for a given mode.
    """

    def __init__( self, kz, kr, m, dt, Nz, Nr, V=None,
                  use_galilean=False, use_cuda=False ) :
        """
        Allocates the coefficients matrices for the psatd scheme.

        Parameters
        ----------
        kz : 2darray of float
            The positions of the longitudinal, spectral grid

        kr : 2darray of float
            The positions of the radial, spectral grid

        m : int
            Index of the mode

        dt : float
            The timestep of the simulation

        V: float or None, optional
            If this variable is None, the standard PSATD is used (default).
            Otherwise, the current is assumed to be "comoving",
            i.e. constant with respect to (z - v_comoving * t).
            This can be done in two ways: either by
            - Using a PSATD scheme that takes this hypothesis into account
            - Solving the PSATD scheme in a Galilean frame

        use_galilean: bool, optional
            Determines which one of the two above schemes is used
            When use_galilean is true, the whole grid moves
            with a speed v_comoving

        use_cuda : bool, optional
            Wether to use the GPU or not
        """
        # Shortcuts
        i = 1.j

        # Register m and dt
        self.m = m
        self.dt = dt
        inv_dt = 1./dt
        # Register velocity of galilean/comoving frame
        self.V = V

        # Construct the omega and inverse omega array
        w = c*np.sqrt( kz**2 + kr**2 )
        inv_w = 1./np.where( w == 0, 1., w ) # Avoid division by 0

        # Construct the C coefficient arrays
        self.C = np.cos( w*dt )
        # Construct the S/w coefficient arrays
        self.S_w = np.sin( w*dt )*inv_w
        # Enforce the right value for w==0
        self.S_w[ w==0 ] = dt

        # Calculate coefficients that are specific to galilean/comoving scheme
        if self.V is not None:

            # Theta coefficients due to galilean/comoving scheme
            T2 = np.exp(i*kz*V*dt)
            if use_galilean is False:
                T = np.exp(i*0.5*kz*V*dt)
            # The coefficients T_cc and T_eb abstract the modification
            # of the comoving current or galilean frame, so that the Maxwell
            # equations can be written in the same form
            if use_galilean:
                self.T_eb = T2
                self.T_cc = np.ones_like(T2)
            else:
                self.T_cc = T
                self.T_eb = np.ones_like(T2)

            # Theta-like coefficient for calculation of rho_diff
            if V != 0.:
                i_kz_V = i*kz*self.V
                i_kz_V[ kz==0 ] = 1.
                self.T_rho = np.where(
                    kz == 0., -self.dt, (1.-T2)/(self.T_cc*i_kz_V) )
            else:
                self.T_rho = -self.dt*np.ones_like(kz)

            # Precalculate some coefficients
            if V != 0.:
                # Calculate pre-factor
                inv_w_kzV = 1./np.where(
                                (w**2 - kz**2 * V**2)==0,
                                1.,
                                (w**2 - kz**2 * V**2) )
                # Calculate factor involving 1/T2
                inv_1_T2 = 1./np.where(T2 == 1, 1., 1-T2)
                # Calculate Xi 1 coefficient
                xi_1 = 1./self.T_cc * inv_w_kzV \
                       * (1. - T2*self.C + i*kz*V*T2*self.S_w)
                # Calculate Xi 2 coefficient
                xi_2 = np.where(
                        kz!=0,
                        inv_w_kzV * ( 1. \
                            + i*kz*V * T2 * self.S_w * inv_1_T2 \
                            + kz**2*V**2 * inv_w**2 * T2 * \
                            inv_1_T2*(1-self.C) ),
                        1.*inv_w**2 * (1.-self.S_w*inv_dt) )
                # Calculate Xi 3 coefficient
                xi_3 = np.where(
                        kz!=0,
                        self.T_eb * inv_w_kzV * ( self.C \
                            + i*kz*V * T2 *self.S_w * inv_1_T2 \
                            + kz**2*V**2 * inv_w**2 * \
                            inv_1_T2 * (1-self.C) ),
                        1.*inv_w**2 * (self.C-self.S_w*inv_dt) )

            # Calculate correction coefficient for j
            if V !=0:
                self.j_corr_coef = np.where( kz != 0,
                            (-i*kz*V)*inv_1_T2,
                            inv_dt )
            else:
                self.j_corr_coef = inv_dt*np.ones_like(kz)

        # Construct j_coef array (for use in the Maxwell equations)
        if V is None or V == 0:
            self.j_coef = mu_0*c**2*(1.-self.C)*inv_w**2
        else:
            self.j_coef = mu_0*c**2*(xi_1)
        # Enforce the right value for w==0
        self.j_coef[ w==0 ] = mu_0*c**2*(0.5*dt**2)

        # Calculate rho_prev coefficient array
        if V is None or V == 0:
            self.rho_prev_coef = c**2/epsilon_0*(
                self.C - inv_dt*self.S_w )*inv_w**2
        else:
            self.rho_prev_coef = c**2/epsilon_0*(xi_3)
        # Enforce the right value for w==0
        self.rho_prev_coef[ w==0 ] = c**2/epsilon_0*(-1./3*dt**2)

        # Calculate rho_next coefficient array
        if V is None or V == 0:
            self.rho_next_coef = c**2/epsilon_0*(
                1 - inv_dt*self.S_w )*inv_w**2
        else:
            self.rho_next_coef = c**2/epsilon_0*(xi_2)
        # Enforce the right value for w==0
        self.rho_next_coef[ w==0 ] = c**2/epsilon_0*(1./6*dt**2)

        # Replace these array by arrays on the GPU, when using cuda
        if use_cuda:
            self.d_C = cuda.to_device(self.C)
            self.d_S_w = cuda.to_device(self.S_w)
            self.d_j_coef = cuda.to_device(self.j_coef)
            self.d_rho_prev_coef = cuda.to_device(self.rho_prev_coef)
            self.d_rho_next_coef = cuda.to_device(self.rho_next_coef)
            if self.V is not None:
                # Variables which are specific to the Galilean/comoving scheme
                self.d_T_eb = cuda.to_device(self.T_eb)
                self.d_T_cc = cuda.to_device(self.T_cc)
                self.d_T_rho = cuda.to_device(self.T_rho)
                self.d_j_corr_coef = cuda.to_device(self.j_corr_coef)

# -----------------
# Utility function
# -----------------

def get_filter_array( kz, kr, dz, dr ) :
    """
    Return the array that multiplies the fields in k space

    The filtering function is 1-sin( k/kmax * pi/2 )**2.
    (equivalent to a one-pass binomial filter in real space,
    for the longitudinal direction)

    Parameters
    ----------
    kz: 1darray
        The true wavevectors of the longitudinal, spectral grid
        (i.e. not the kz modified by finite order)

    kr: 1darray
        The transverse wavevectors on the spectral grid

    dz, dr: float
        The grid spacings (needed to calculate
        precisely the filtering function in spectral space)

    Returns
    -------
    A 2darray of shape ( len(kz), len(kr) )
    """
    # Find the 1D filter in z
    filt_z = 1. - np.sin( 0.5 * kz * dz )**2

    # Find the 1D filter in r
    filt_r = 1. - np.sin( 0.5 * kr * dr )**2

    # Build the 2D filter by takin the product
    filter_array = filt_z[:, np.newaxis] * filt_r[np.newaxis, :]

    return( filter_array )


def get_modified_k(k, n_order, dz):
    """
    Calculate the modified k that corresponds to a finite-order stencil

    The modified k are given by the formula
    $$ [k] = \sum_{n=1}^{m} a_n \,\frac{\sin(nk\Delta z)}{n\Delta z}$$
    with
    $$a_{n} = - \left(\frac{m+1-n}{m+n}\right) a_{n-1}$$

    Parameter:
    ----------
    k: 1darray
       Values of the real k at which to calculate the modified k

    n_order: int
       The order of the stencil
           Use -1 for infinite order
           Otherwise use a positive, even number. In this case
           the stencil extends up to n_order/2 cells on each side.

    dz: double
       The spacing of the grid in z

    Returns:
    --------
    A 1d array of the same length as k, which contains the modified ks
    """
    # Check the order
    # - For n_order = -1, do not go through the function.
    if n_order==-1:
        return( k )
    # - Else do additional checks
    elif n_order%2==1 or n_order<=0 :
        raise ValueError('Invalid n_order: %d' %n_order)
    else:
        m = int(n_order/2)

    # Calculate the stencil coefficients a_n by recurrence
    # (See definition of the a_n in the docstring)
    # $$ a_{n} = - \left(\frac{m+1-n}{m+n}\right) a_{n-1} $$
    stencil_coef = np.zeros(m+1)
    stencil_coef[0] = -2.
    for n in range(1,m+1):
        stencil_coef[n] = - (m+1-n)*1./(m+n) * stencil_coef[n-1]

    # Array of values for n: from 1 to m
    n_array = np.arange(1,m+1)
    # Array of values of sin:
    # first axis corresponds to k and second axis to n (from 1 to m)
    sin_array = np.sin( k[:,np.newaxis] * n_array[np.newaxis,:] * dz ) / \
                ( n_array[np.newaxis,:] * dz )

    # Modified k
    k_array = np.tensordot( sin_array, stencil_coef[1:], axes=(-1,-1))

    return( k_array )
