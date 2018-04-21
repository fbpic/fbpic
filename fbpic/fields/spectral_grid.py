# Copyright 2018, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the SpectralGrid class.
"""
import numpy as np
from scipy.constants import epsilon_0
from .utility_methods import get_filter_array
from .numba_methods import numba_push_eb_standard, numba_push_eb_comoving, \
    numba_correct_currents_curlfree_standard, \
    numba_correct_currents_crossdeposition_standard, \
    numba_correct_currents_curlfree_comoving, \
    numba_correct_currents_crossdeposition_comoving
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    from fbpic.utils.cuda import cuda_tpb_bpg_2d
    from .cuda_methods import cuda, \
    cuda_correct_currents_curlfree_standard, \
    cuda_correct_currents_crossdeposition_standard, \
    cuda_correct_currents_curlfree_comoving, \
    cuda_correct_currents_crossdeposition_comoving, \
    cuda_filter_scalar, cuda_filter_vector, \
    cuda_push_eb_standard, cuda_push_eb_comoving, cuda_push_rho


class SpectralGrid(object) :
    """
    Contains the fields and coordinates of the spectral grid.
    """

    def __init__(self, kz_modified, kr, m, kz_true, dz, dr,
                        current_correction, use_cuda=False ) :
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

        current_correction: string, optional
            The method used in order to ensure that the continuity equation
            is satisfied. Either `curl-free` or `cross-deposition`.

        use_cuda : bool, optional
            Wether to use the GPU or not
        """
        # Register the arrays and their length
        Nz = len(kz_modified)
        Nr = len(kr)
        self.Nr = Nr
        self.Nz = Nz
        self.m = m

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
        if current_correction == 'cross-deposition':
            self.rho_next_z = np.zeros( (Nz, Nr), dtype='complex' )
            self.rho_next_xy = np.zeros( (Nz, Nr), dtype='complex' )

        # Auxiliary arrays
        # - for the field solve
        #   (use the modified kz, since this corresponds to the stencil)
        self.kz, self.kr = np.meshgrid( kz_modified, kr, indexing='ij' )
        # - for filtering
        #   (use the true kz, so as to effectively filter the high k's)
        self.filter_array = get_filter_array( kz_true, kr, dz, dr )
        # - for curl-free current correction
        if current_correction == 'curl-free':
            self.inv_k2 = 1./np.where( ( self.kz == 0 ) & (self.kr == 0),
                                       1., self.kz**2 + self.kr**2 )
            self.inv_k2[ ( self.kz == 0 ) & (self.kr == 0) ] = 0.

        # Register shift factor used for shifting the fields
        # in the spectral domain when using a moving window
        self.field_shift = np.exp(1.j*kz_true*dz)

        # Check whether to use the GPU
        self.use_cuda = use_cuda

        # Transfer the auxiliary arrays on the GPU
        if self.use_cuda :
            self.d_filter_array = cuda.to_device( self.filter_array )
            self.d_kz = cuda.to_device( self.kz )
            self.d_kr = cuda.to_device( self.kr )
            self.d_field_shift = cuda.to_device( self.field_shift )
            if current_correction == 'curl-free':
                self.d_inv_k2 = cuda.to_device( self.inv_k2 )


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
        # Only when using the cross-deposition
        if hasattr( self, 'rho_next_z' ):
            self.rho_next_z = cuda.to_device( self.rho_next_z )
            self.rho_next_xy = cuda.to_device( self.rho_next_xy )


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
        # Only when using the cross-deposition
        if hasattr( self, 'rho_next_z' ):
            self.rho_next_z = self.rho_next_z.copy_to_host()
            self.rho_next_xy = self.rho_next_xy.copy_to_host()


    def correct_currents (self, dt, ps, current_correction ):
        """
        Correct the currents so that they satisfy the
        charge conservation equation

        Parameters
        ----------
        dt: float
            Timestep of the simulation

        ps: a PSATDCoefs object
            Contains coefficients that are used in the current correction

        current_correction: string
            The type of current correction performed
        """
        # Precalculate useful coefficient
        inv_dt = 1./dt

        if self.use_cuda :
            # Obtain the cuda grid
            dim_grid, dim_block = cuda_tpb_bpg_2d( self.Nz, self.Nr)
            # Correct the currents on the GPU
            if ps.V is None:
                # With standard PSATD algorithm
                # Method: curl-free
                if current_correction == 'curl-free':
                    cuda_correct_currents_curlfree_standard \
                        [dim_grid, dim_block](
                            self.rho_prev, self.rho_next,
                            self.Jp, self.Jm, self.Jz,
                            self.d_kz, self.d_kr, self.d_inv_k2,
                            inv_dt, self.Nz, self.Nr )
                # Method: cross-deposition
                elif current_correction == 'cross-deposition':
                    cuda_correct_currents_crossdeposition_standard \
                        [dim_grid, dim_block](
                            self.rho_prev, self.rho_next,
                            self.rho_next_z, self.rho_next_xy,
                            self.Jp, self.Jm, self.Jz,
                            self.d_kz, self.d_kr, inv_dt, self.Nz, self.Nr)
            else:
                # With Galilean/comoving algorithm
                # Method: curl-free
                if current_correction == 'curl-free':
                    cuda_correct_currents_curlfree_comoving \
                        [dim_grid, dim_block](
                            self.rho_prev, self.rho_next,
                            self.Jp, self.Jm, self.Jz,
                            self.d_kz, self.d_kr, self.d_inv_k2,
                            ps.d_j_corr_coef, ps.d_T_eb, ps.d_T_cc,
                            inv_dt, self.Nz, self.Nr)
                # Method: cross-deposition
                elif current_correction == 'cross-deposition':
                    cuda_correct_currents_crossdeposition_comoving \
                        [dim_grid, dim_block](
                            self.rho_prev, self.rho_next,
                            self.rho_next_z, self.rho_next_xy,
                            self.Jp, self.Jm, self.Jz,
                            self.d_kz, self.d_kr,
                            ps.d_j_corr_coef, ps.d_T_eb, ps.d_T_cc,
                            inv_dt, self.Nz, self.Nr)
        else :
            # Correct the currents on the CPU
            if ps.V is None:
                # With standard PSATD algorithm
                # Method: curl-free
                if current_correction == 'curl-free':
                    numba_correct_currents_curlfree_standard(
                        self.rho_prev, self.rho_next,
                        self.Jp, self.Jm, self.Jz,
                        self.kz, self.kr, self.inv_k2,
                        inv_dt, self.Nz, self.Nr)
                # Method: cross-deposition
                elif current_correction == 'cross-deposition':
                    numba_correct_currents_crossdeposition_standard(
                        self.rho_prev, self.rho_next,
                        self.rho_next_z, self.rho_next_xy,
                        self.Jp, self.Jm, self.Jz,
                        self.kz, self.kr, inv_dt, self.Nz, self.Nr)
            else:
                # With Galilean/comoving algorithm
                # Method: curl-free
                if current_correction == 'curl-free':
                    numba_correct_currents_curlfree_comoving(
                        self.rho_prev, self.rho_next,
                        self.Jp, self.Jm, self.Jz,
                        self.kz, self.kr, self.inv_k2,
                        ps.j_corr_coef, ps.T_eb, ps.T_cc,
                        inv_dt, self.Nz, self.Nr)
                # Method: cross-deposition
                elif current_correction == 'cross-deposition':
                    numba_correct_currents_crossdeposition_comoving(
                        self.rho_prev, self.rho_next,
                        self.rho_next_z, self.rho_next_xy,
                        self.Jp, self.Jm, self.Jz,
                        self.kz, self.kr,
                        ps.j_corr_coef, ps.T_eb, ps.T_cc,
                        inv_dt, self.Nz, self.Nr)


    def correct_divE(self) :
        """
        Correct the electric field, so that it satisfies the equation
        div(E) - rho/epsilon_0 = 0
        """
        # Correct div(E) on the CPU

        # Calculate the intermediate variable F
        F = - self.inv_k2 * (
            - self.rho_prev/epsilon_0 \
            + 1.j*self.kz*self.Ez + self.kr*( self.Ep - self.Em ) )

        # Correct the current accordingly
        self.Ep += 0.5*self.kr*F
        self.Em += -0.5*self.kr*F
        self.Ez += -1.j*self.kz*F

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
            if fieldtype == 'J' :
                cuda_filter_vector[dim_grid, dim_block]( self.Jp, self.Jm,
                        self.Jz, self.d_filter_array, self.Nz, self.Nr)
            elif fieldtype == 'E' :
                cuda_filter_vector[dim_grid, dim_block]( self.Ep, self.Em,
                        self.Ez, self.d_filter_array, self.Nz, self.Nr)
            elif fieldtype == 'B' :
                cuda_filter_vector[dim_grid, dim_block]( self.Bp, self.Bm,
                        self.Bz, self.d_filter_array, self.Nz, self.Nr)
            elif fieldtype in ['rho_prev', 'rho_next',
                                'rho_next_z', 'rho_next_xy']:
                spectral_rho = getattr( self, fieldtype )
                cuda_filter_scalar[dim_grid, dim_block](
                    spectral_rho, self.d_filter_array, self.Nz, self.Nr )
            else :
                raise ValueError('Invalid string for fieldtype: %s'%fieldtype)
        else :
            # Filter fields on the CPU
            if fieldtype == 'J':
                self.Jp = self.Jp * self.filter_array
                self.Jm = self.Jm * self.filter_array
                self.Jz = self.Jz * self.filter_array
            elif fieldtype == 'E':
                self.Ep = self.Ep * self.filter_array
                self.Em = self.Em * self.filter_array
                self.Ez = self.Ez * self.filter_array
            elif fieldtype == 'B':
                self.Bp = self.Bp * self.filter_array
                self.Bm = self.Bm * self.filter_array
                self.Bz = self.Bz * self.filter_array
            elif fieldtype in ['rho_prev', 'rho_next',
                                'rho_next_z', 'rho_next_xy']:
                spectral_rho = getattr( self, fieldtype )
                spectral_rho *= self.filter_array
            else :
                raise ValueError('Invalid string for fieldtype: %s'%fieldtype)
