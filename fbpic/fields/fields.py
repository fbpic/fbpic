# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the high-level Fields class.
"""
import warnings
import numpy as np
from fbpic.utils.threading import nthreads
from .numba_methods import sum_reduce_2d_array
from .cuda_methods import cuda_copy_arrays
from .utility_methods import get_modified_k
from .spectral_transform import SpectralTransformer
from .interpolation_grid import FieldInterpolationGrid, \
                         EnvelopeInterpolationGrid
from .spectral_grid import FieldSpectralGrid, \
                         EnvelopeSpectralGrid
from .psatd_coefs import PsatdCoeffs
from fbpic.utils.cuda import cuda_installed

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
                  n_order=-1, v_comoving=None, use_galilean=True,
                  current_correction='cross-deposition', use_cuda=False,
                  create_threading_buffers=False, use_envelope=False,
                  lambda_envelope=0.8e-6 ):
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

        current_correction: string, optional
            The method used in order to ensure that the continuity equation
            is satisfied. Either `curl-free` or `cross-deposition`.

        use_cuda : bool, optional
            Wether to use the GPU or not

        create_threading_buffers: bool, optional
            Whether to create the buffers used in order to perform
            charge/current deposition with threading on CPU
            (buffers are duplicated with the number of threads)

        use_envelope: bool, optional
            Whether to use the envelope approximation of the fields or not
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
        self.zmin = zmin
        self.zmax = zmax

        # Define wether or not to use the GPU
        self.use_cuda = use_cuda
        if (self.use_cuda==True) and (cuda_installed==False) :
            warnings.warn(
                'Cuda not available for the fields.\n'
                'Performing the field operations on the CPU.' )
            self.use_cuda = False

        # Register the current correction type
        if current_correction in ['curl-free', 'cross-deposition']:
            self.current_correction = current_correction
        else:
            raise ValueError('Unkown current correction:%s'%current_correction)

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
            # Create the object
            self.interp.append( FieldInterpolationGrid(
                Nz, Nr, m, zmin, zmax, rmax, use_cuda=self.use_cuda ) )

        # Get the kz and (finite-order) modified kz arrays
        # (According to FFT conventions, the kz array starts with
        # positive frequencies and ends with negative frequency.)
        dz = (zmax-zmin)/Nz
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
            self.spect.append( FieldSpectralGrid( kz_modified, kr, m,
                kz_true, self.interp[m].dz, self.interp[m].dr,
                current_correction, use_cuda=self.use_cuda ) )
            self.psatd.append( PsatdCoeffs( self.spect[m].kz,
                                self.spect[m].kr, m, dt, Nz, Nr,
                                V=self.v_comoving,
                                use_galilean=self.use_galilean,
                                use_cuda=self.use_cuda ) )

        # Record flags that indicates whether, for the sources *in
        # spectral space*, the guard cells have been exchanged via MPI
        self.exchanged_source = \
            {'J': False, 'rho_prev': False, 'rho_new': False,
                'rho_next_xy': False, 'rho_next_z': False }

        # Generate duplicated deposition arrays, when using threading
        # (One copy per thread ; 2 guard cells on each side in z and r,
        # in order to store contributions from, at most, cubic shape factors ;
        # these deposition guard cells are folded into the regular box
        # inside `sum_reduce_2d_array`)
        if create_threading_buffers:
            self.rho_global = np.zeros( dtype=np.complex128,
                shape=(nthreads, self.Nm, self.Nz+4, self.Nr+4) )
            self.Jr_global = np.zeros( dtype=np.complex128,
                    shape=(nthreads, self.Nm, self.Nz+4, self.Nr+4) )
            self.Jt_global = np.zeros( dtype=np.complex128,
                    shape=(nthreads, self.Nm, self.Nz+4, self.Nr+4) )
            self.Jz_global = np.zeros( dtype=np.complex128,
                    shape=(nthreads, self.Nm, self.Nz+4, self.Nr+4) )

        # By default will not use the envelope model
        self.use_envelope = use_envelope
        if self.use_envelope:
            self.lambda_envelope = lambda_envelope
            #Create the envelope interpolation grids for each modes
            #The envelope modes range from -Nm + 1 to Nm - 1
            self.envelope_interp = []
            self.envelope_mode_numbers = [ m for m in range(self.Nm) ] + \
                                         [ m for m in range(-self.Nm+1, 0)]
            for m in self.envelope_mode_numbers[self.Nm:]:
                self.trans.append( SpectralTransformer(
                    Nz, Nr, m, rmax, use_cuda=self.use_cuda ) )
            for m in self.envelope_mode_numbers:
                #Modes are listed in order: 0, 1, ..., Nm - 1, -Nm + 1, ..., -1
                self.envelope_interp.append(
                    EnvelopeInterpolationGrid(
                        self.Nz, self.Nr, m, self.zmin, self.zmax,
                        self.rmax, use_cuda=self.use_cuda ) )

            #Create the envelope spectral grids for each modes
            self.envelope_spect = []
            for m in self.envelope_mode_numbers:
                #Modes are listed in order: 0, 1, ..., Nm - 1, -Nm + 1, ..., -1
                kr = 2*np.pi * self.trans[m].dht0.get_nu()
                self.envelope_spect.append(
                    EnvelopeSpectralGrid( kz_modified, kr,
                        m, kz_true, self.envelope_interp[m].dz,
                        self.envelope_interp[m].dr, use_cuda=self.use_cuda ) )

            # Create the psatd coefficients relevant only
            # to the envelope model for each positive mode
            for m in range(self.Nm):
                self.psatd[m].compute_envelope_coefs(self.spect[m].kz,
                            self.spect[m].kr, m, self.dt, self.Nz,
                            self.Nr, 2*np.pi/lambda_envelope)

            self.a_global = np.zeros(dtype=np.complex128,
                shape=(2*Nm-1, Nz, Nr))
            self.grad_a_r_global = np.zeros(dtype=np.complex128,
                shape=(2*Nm-1, Nz, Nr))
            self.grad_a_t_global = np.zeros(dtype=np.complex128,
                shape=(2*Nm-1, Nz, Nr))
            self.grad_a_z_global  = np.zeros(dtype=np.complex128,
                shape=(2*Nm-1, Nz, Nr))

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
            if self.use_envelope:
                for m in self.envelope_mode_numbers:
                    self.envelope_interp[m].send_fields_to_gpu()
                    self.envelope_spect[m].send_fields_to_gpu()

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
            if self.use_envelope:
                for m in self.envelope_mode_numbers:
                    self.envelope_interp[m].receive_fields_from_gpu()
                    self.envelope_spect[m].receive_fields_from_gpu()

    def push(self, use_true_rho=False, check_exchanges=False):
        """
        Push the different azimuthal modes over one timestep,
        in spectral space.

        Parameters
        ----------
        use_true_rho : bool, optional
            Whether to use the rho projected on the grid.
            If set to False, this will use div(E) and div(J)
            to evaluate rho and its time evolution.
            In the case use_true_rho==False, the rho projected
            on the grid is used only to correct the currents, and
            the simulation can be run without the neutralizing ions.
        check_exchanges: bool, optional
            Check whether the guard cells of the fields rho and J
            have been properly exchanged via MPI
        """
        if check_exchanges:
            # Ensure consistency: fields should be exchanged
            assert self.exchanged_source['J'] == True
            if use_true_rho:
                assert self.exchanged_source['rho_prev'] == True
                assert self.exchanged_source['rho_next'] == True

        # Push each azimuthal grid individually, by passing the
        # corresponding psatd coefficients
        for m in range(self.Nm) :
            self.spect[m].push_eb_with( self.psatd[m], use_true_rho )
        for m in range(self.Nm) :
            self.spect[m].push_rho()

    def push_envelope(self):
        """
        Push the different azimuthal modes over one timestep,
        in spectral space.
        """
        # push each azimuthal mode individually
        for m in self.envelope_mode_numbers :
            self.envelope_spect[m].push_envelope_with(self.psatd[abs(m)])

    def correct_currents(self, check_exchanges=False) :
        """
        Correct the currents so that they satisfy the
        charge conservation equation

        Parameter
        ---------
        check_exchanges: bool
            Check whether the guard cells of the fields rho and J
            have been properly exchanged via MPI
        """
        if check_exchanges:
            # Ensure consistency (charge and current should
            # not be exchanged via MPI before correction)
            assert self.exchanged_source['rho_prev'] == False
            assert self.exchanged_source['rho_next'] == False
            assert self.exchanged_source['J'] == False
            if self.current_correction == 'cross-deposition':
                assert self.exchanged_source['rho_next_xy'] == False
                assert self.exchanged_source['rho_next_z'] == False

        # Correct each azimuthal grid individually
        for m in range(self.Nm) :
            self.spect[m].correct_currents(
                self.dt, self.psatd[m], self.current_correction )

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
            (either 'E', 'B', 'J', 'rho_next', 'rho_prev', 'a', 'grad_a')
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
        elif fieldtype in ['rho_prev', 'rho_next', 'rho_next_z', 'rho_next_xy']:
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                spectral_rho = getattr( self.spect[m], fieldtype )
                self.trans[m].interp2spect_scal(
                    self.interp[m].rho, spectral_rho )
        elif fieldtype == 'a' and self.use_envelope:
            # Transform each azimuthal grid individually
            for m in self.envelope_mode_numbers:
                self.trans[m].interp2spect_scal(
                    self.envelope_interp[m].a, self.envelope_spect[m].a )
        elif fieldtype == 'a_old' and self.use_envelope:
            # Transform each azimuthal grid individually
            for m in self.envelope_mode_numbers:
                self.trans[m].interp2spect_scal(
                    self.envelope_interp[m].a_old, self.envelope_spect[m].a_old)
        elif fieldtype == 'grad_a' and self.use_envelope:
            # Transform each azimuthal grid individually
            for m in self.envelope_mode_numbers :
                self.trans[m].interp2spect_scal(
                    self.envelope_interp[m].grad_a_z,
                     self.envelope_spect[m].grad_a_z )
                self.trans[m].interp2spect_vect(
                    self.envelope_interp[m].grad_a_r,
                    self.envelope_interp[m].grad_a_t,
                    self.envelope_spect[m].grad_a_p,
                    self.envelope_spect[m].grad_a_m )
        else:
            raise ValueError( 'Invalid string for fieldtype: %s' %fieldtype )

    def spect2interp(self, fieldtype) :
        """
        Transform the fields `fieldtype` from the spectral grid
        to the interpolation grid

        Parameter
        ---------
        fieldtype :
            A string which represents the kind of field to transform
            (either 'E', 'B', 'J', 'rho_next', 'rho_prev', 'a', 'grad_a')
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
        elif fieldtype == 'rho_next' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].spect2interp_scal(
                    self.spect[m].rho_next, self.interp[m].rho )
        elif fieldtype == 'rho_prev' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].spect2interp_scal(
                    self.spect[m].rho_prev, self.interp[m].rho )
        elif fieldtype == 'a' and self.use_envelope:
            # Transform each azimuthal grid individually
            for m in self.envelope_mode_numbers :
                self.trans[m].spect2interp_scal(
                    self.envelope_spect[m].a, self.envelope_interp[m].a )
        elif fieldtype == 'a_old' and self.use_envelope:
            # Transform each azimuthal grid individually
            for m in self.envelope_mode_numbers:
                self.trans[m].spect2interp_scal(
                    self.envelope_spect[m].a_old, self.envelope_interp[m].a_old )
        elif fieldtype == 'grad_a' and self.use_envelope :
            # Transform each azimuthal grid individually
            for m in self.envelope_mode_numbers :
                self.trans[m].spect2interp_scal(
                    self.envelope_spect[m].grad_a_z,
                    self.envelope_interp[m].grad_a_z )
                self.trans[m].spect2interp_vect(
                    self.envelope_spect[m].grad_a_p,
                    self.envelope_spect[m].grad_a_m,
                    self.envelope_interp[m].grad_a_r,
                    self.envelope_interp[m].grad_a_t )
        else :
            raise ValueError( 'Invalid string for fieldtype: %s' %fieldtype )

    def spect2partial_interp(self, fieldtype) :
        """
        Transform the fields `fieldtype` from the spectral grid,
        by only performing an inverse FFT in z (but no Hankel transform)

        This is typically done before exchanging guard cells in z
        (a full FFT+Hankel transform is not necessary in this case)

        The result is stored in the interpolation grid (for economy of memory),
        but one should be aware that these fields are not actually the
        interpolation fields. These "incorrect" fields would however be
        overwritten by subsequent calls to `spect2interp` (see `step` function)

        Parameter
        ---------
        fieldtype :
            A string which represents the kind of field to transform
            (either 'E', 'B', 'J', 'rho_next', 'rho_prev', 'a', 'grad_a')
        """
        # Use the appropriate transformation depending on the fieldtype.
        if fieldtype == 'E' :
            for m in range(self.Nm) :
                self.trans[m].fft.inverse_transform(
                    self.spect[m].Ez, self.interp[m].Ez )
                self.trans[m].fft.inverse_transform(
                    self.spect[m].Ep, self.interp[m].Er )
                self.trans[m].fft.inverse_transform(
                    self.spect[m].Em, self.interp[m].Et )
        elif fieldtype == 'B' :
            for m in range(self.Nm) :
                self.trans[m].fft.inverse_transform(
                    self.spect[m].Bz, self.interp[m].Bz )
                self.trans[m].fft.inverse_transform(
                    self.spect[m].Bp, self.interp[m].Br )
                self.trans[m].fft.inverse_transform(
                    self.spect[m].Bm, self.interp[m].Bt )
        elif fieldtype == 'J' :
            for m in range(self.Nm) :
                self.trans[m].fft.inverse_transform(
                    self.spect[m].Jz, self.interp[m].Jz )
                self.trans[m].fft.inverse_transform(
                    self.spect[m].Jp, self.interp[m].Jr )
                self.trans[m].fft.inverse_transform(
                    self.spect[m].Jm, self.interp[m].Jt )
        elif fieldtype == 'rho_next' :
            for m in range(self.Nm) :
                self.trans[m].fft.inverse_transform(
                    self.spect[m].rho_next, self.interp[m].rho )
        elif fieldtype == 'rho_prev' :
            for m in range(self.Nm) :
                self.trans[m].fft.inverse_transform(
                    self.spect[m].rho_prev, self.interp[m].rho )
        elif fieldtype == 'a' and self.use_envelope:
            for m in self.envelope_mode_numbers :
                self.trans[m].fft.inverse_transform(
                    self.envelope_spect[m].a, self.envelope_interp[m].a )
        elif fieldtype == 'a_old' and self.use_envelope:
            # Transform each azimuthal grid individually
            for m in self.envelope_mode_numbers:
                self.trans[m].fft.inverse_transform(
                    self.envelope_spect[m].a_old, self.envelope_interp[m].a_old )
        elif fieldtype == 'grad_a' :
            for m in self.envelope_mode_numbers  :
                self.trans[m].fft.inverse_transform(
                    self.envelope_spect[m].grad_a_z,
                    self.envelope_interp[m].grad_a_z )
                self.trans[m].fft.inverse_transform(
                    self.envelope_spect[m].grad_a_p,
                    self.envelope_interp[m].grad_a_r )
                self.trans[m].fft.inverse_transform(
                    self.envelope_spect[m].grad_a_m,
                    self.envelope_interp[m].grad_a_t )

        else :
            raise ValueError( 'Invalid string for fieldtype: %s' %fieldtype )


    def partial_interp2spect(self, fieldtype) :
        """
        Transform the fields `fieldtype` from the partial representation
        in interpolation space (obtained from `spect2partial_interp`)
        to the spectral grid.

        This is typically done after exchanging guard cells in z
        (a full FFT+Hankel transform is not necessary in this case)

        Parameter
        ---------
        fieldtype :
            A string which represents the kind of field to transform
            (either 'E', 'B', 'J', 'rho_next', 'rho_prev', 'a', 'grad_a')
        """
        # Use the appropriate transformation depending on the fieldtype.
        if fieldtype == 'E' :
            for m in range(self.Nm) :
                self.trans[m].fft.transform(
                    self.interp[m].Ez, self.spect[m].Ez )
                self.trans[m].fft.transform(
                    self.interp[m].Er, self.spect[m].Ep )
                self.trans[m].fft.transform(
                    self.interp[m].Et, self.spect[m].Em )
        elif fieldtype == 'B' :
            for m in range(self.Nm) :
                self.trans[m].fft.transform(
                    self.interp[m].Bz, self.spect[m].Bz )
                self.trans[m].fft.transform(
                    self.interp[m].Br, self.spect[m].Bp )
                self.trans[m].fft.transform(
                    self.interp[m].Bt, self.spect[m].Bm )
        elif fieldtype == 'J' :
            for m in range(self.Nm) :
                self.trans[m].fft.transform(
                    self.interp[m].Jz, self.spect[m].Jz )
                self.trans[m].fft.transform(
                    self.interp[m].Jr, self.spect[m].Jp )
                self.trans[m].fft.transform(
                    self.interp[m].Jt, self.spect[m].Jm )
        elif fieldtype == 'rho_next' :
            for m in range(self.Nm) :
                self.trans[m].fft.transform(
                    self.interp[m].rho, self.spect[m].rho_next )
        elif fieldtype == 'rho_prev' :
            for m in range(self.Nm) :
                self.trans[m].fft.transform(
                    self.interp[m].rho, self.spect[m].rho_prev )
        elif fieldtype == 'a' and self.use_envelope:
            for m in self.envelope_mode_numbers :
                self.trans[m].fft.transform(
                    self.envelope_interp[m].a, self.envelope_spect[m].a )
        elif fieldtype == 'a_old' and self.use_envelope:
            # Transform each azimuthal grid individually
            for m in self.envelope_mode_numbers:
                self.trans[m].fft.transform(
                    self.envelope_interp[m].a_old, self.envelope_spect[m].a_old)
        elif fieldtype == 'grad_a' :
            for m in self.envelope_mode_numbers :
                self.trans[m].fft.transform(
                    self.envelope_interp[m].grad_a_z,
                    self.envelope_spect[m].grad_a_z )
                self.trans[m].fft.transform(
                    self.envelope_interp[m].grad_a_r,
                    self.envelope_spect[m].grad_a_p )
                self.trans[m].fft.transform(
                    self.envelope_interp[m].grad_a_t,
                    self.envelope_spect[m].grad_a_m )

        else :
            raise ValueError( 'Invalid string for fieldtype: %s' %fieldtype )


    def erase(self, fieldtype ) :
        """
        Sets the field `fieldtype` to zero on the interpolation grid

        (For 'rho' and 'J', on CPU, this also erases the duplicated
        deposition buffer, with one copy per thread)

        Parameter
        ---------
        fieldtype : string
            A string which represents the kind of field to be erased
            (either 'E', 'B', 'J', 'rho')
        """
        # Erase the fields in the interpolation grid
        for m in range(self.Nm):
            self.interp[m].erase(fieldtype)
        # Erase the duplicated deposition buffer
        if not self.use_cuda:
            if fieldtype == 'rho':
                self.rho_global[:,:,:,:] = 0.
            elif fieldtype == 'J':
                self.Jr_global[:,:,:,:] = 0.
                self.Jt_global[:,:,:,:] = 0.
                self.Jz_global[:,:,:,:] = 0.

    def sum_reduce_deposition_array(self, fieldtype):
        """
        Sum the duplicated array for rho and J deposition on CPU
        into a single array.

        This function does nothing when running on GPU

        Parameters
        ----------
        fieldtype : string
            A string which represents the kind of field to be erased
            (either 'J' or 'rho')
        """
        # Skip this function when running on GPU
        if self.use_cuda:
            return

        # Sum thread-local results to main field array
        if fieldtype == 'rho':
            for m in range(self.Nm):
                sum_reduce_2d_array( self.rho_global, self.interp[m].rho, m )
        elif fieldtype == 'J':
            for m in range(self.Nm):
                sum_reduce_2d_array( self.Jr_global, self.interp[m].Jr, m )
                sum_reduce_2d_array( self.Jt_global, self.interp[m].Jt, m )
                sum_reduce_2d_array( self.Jz_global, self.interp[m].Jz, m )
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
            A string which represents the kind of field to be divided by
            the volume (either 'rho' or 'J')
        """
        for m in range(self.Nm):
            self.interp[m].divide_by_volume( fieldtype )

    def compute_grad_a(self):
        """
        Compute the diffent grad_a scalars in spectral space from the current
        'a' field, then converts them to interpolation space.

        After this function is called, grad_a fields in both spectral and
        interpolation space are up to date.
        """
        for m in self.envelope_mode_numbers:
            self.envelope_spect[m].compute_grad_a()
        self.spect2interp('grad_a')

    def globalize_arrays(self):
        """
        Copies the data on the a and grad_a fields that we have scattered in the
        different EnvelopeInterpolationGrid in 4 global arrays that are used for
        the gathering on the particles.
        """
        if self.use_cuda:
            for m in self.envelope_mode_numbers:
                cuda_copy_arrays(self.a_global, m, self.envelope_interp[m].a)
                cuda_copy_arrays(self.grad_a_r_global, m, self.envelope_interp[m].grad_a_r)
                cuda_copy_arrays(self.grad_a_t_global, m, self.envelope_interp[m].grad_a_t)
                cuda_copy_arrays(self.grad_a_z_global, m, self.envelope_interp[m].grad_a_z)
        else:
            for m in self.envelope_mode_numbers:
                self.a_global[m,:,:] = self.envelope_interp[m].a
                self.grad_a_r_global[m,:,:] = self.envelope_interp[m].grad_a_r
                self.grad_a_t_global[m,:,:] = self.envelope_interp[m].grad_a_t
                self.grad_a_z_global[m,:,:] = self.envelope_interp[m].grad_a_z
