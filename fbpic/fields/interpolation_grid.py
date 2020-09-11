# Copyright 2018, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the InterpolationGrid class.
"""
import numpy as np
from fbpic.fields.spectral_transform.hankel import DHT
from scipy.special import j1, jn_zeros
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    import cupy
    from fbpic.utils.cuda import cuda_tpb_bpg_2d
    from .cuda_methods import \
        cuda_erase_scalar, cuda_erase_vector, \
        cuda_divide_scalar_by_volume, cuda_divide_vector_by_volume

class InterpolationGrid(object) :
    """
    Contains the fields and coordinates of the spatial grid.

    Main attributes :
    - z,r : 1darrays containing the positions of the grid
    - Er, Et, Ez, Br, Bt, Bz, Jr, Jt, Jz, rho :
      2darrays containing the fields.
    """

    def __init__(self, Nz, Nr, m, zmin, zmax, rmax,
                    use_pml=False, use_cuda=False,
                    use_ruyten_shapes=True,
                    use_modified_volume=True ):
        """
        Allocates the matrices corresponding to the spatial grid

        Parameters
        ----------
        Nz, Nr : int
            The number of gridpoints in z and r

        m : int
            The index of the mode

        zmin, zmax : float (zmin, optional)
            The initial position of the left and right
            edge of the box along z

        rmax : float
            The position of the edge of the box along r

        use_pml: bool, optional
            Whether to allocate and use Perfectly-Matched-Layers split fields

        use_cuda : bool, optional
            Wether to use the GPU or not

        use_ruyten_shapes: bool, optional
            Whether to use Ruyten shape factors

        use_modified_volume: bool, optional
            Whether to use the modified cell volume (only used for m=0)
        """
        # Register the size of the arrays
        self.Nz = Nz
        self.Nr = Nr
        self.m = m
        self.use_pml = use_pml

        # Register a few grid properties
        dr = rmax/Nr
        dz = (zmax-zmin)/Nz
        self.dr = dr
        self.dz = dz
        self.invdr = 1./dr
        self.invdz = 1./dz
        # rmin, rmax, zmin, zmax correspond to the edge of cells
        self.rmin = 0.
        self.rmax = rmax
        self.zmin = zmin
        self.zmax = zmax

        nr_vals = np.arange(Nr)

        # If enabled, modify the cell volume in mode 0 to ensure that all charge
        # is correctly represented in spectral space. This is done by modifying
        # the cell radius.
        if use_modified_volume and m == 0:
            # Modified cell volume (assuming Hankel-transform corrected volumes)
            alphas = jn_zeros(0,Nr)
            d = DHT( 0, 0, Nr, Nz, rmax, use_cuda=False )
            vol = dz*np.array([( d.M[nr,:]*2./(alphas*j1(alphas)) ).sum()
                                for nr in nr_vals ])
        else:
            # Standard cell volumes
            r = (0.5 + np.arange(Nr))*dr
            vol = np.pi*dz*( (r+0.5*dr)**2 - (r-0.5*dr)**2 )

        # Inverse of cell volume
        self.invvol = 1./vol

        # If enabled, use Ruyten-corrected shape factors
        # (Ruyten JCP 105 (1993) https://doi.org/10.1006/jcph.1993.1070).
        # This ensures that a uniform distribution of macroparticles
        # results in a uniform deposited charge density on the grid,
        # in the limit of many macroparticles in r.
        if use_ruyten_shapes:
            # The Ruyten shape factor coefficients are precalculated and stored
            norm_vol = vol/(2*np.pi*self.dr**2*self.dz)
            self.ruyten_linear_coef = 6./(nr_vals+1)*( \
                                np.cumsum(norm_vol) - 0.5*(nr_vals+1.)**2 - 1./24 )
            self.ruyten_cubic_coef = 6./(nr_vals+1)*( \
                                np.cumsum(norm_vol) - 0.5*(nr_vals+1.)**2 - 1./8 )
            # Correct first value for cubic coeff
            self.ruyten_cubic_coef[0] = 6.*( norm_vol[0] - 0.5 - 239./(15*2**7) )
        else:
            # For standard shapes, the Ruyten coefficients are simply set to zero
            self.ruyten_linear_coef = np.zeros(Nr)
            self.ruyten_cubic_coef = np.zeros(Nr)
        # Add one element (whose value is 0) at the beginning of the coefficient
        # arrays. This element corresponds to the coefficient used by particles
        # located in the first half of the first cell, radially.
        # Note that, the Ruyten correction generally only affects the 2 nearest
        # gridpoints - even with cubic shape - and that this correction has
        # equal value but opposite sign at those 2 points.
        # For the particles located in the first half of the first cell,
        # only the first grid point is affected, and thus the Ruyten
        # corrections (with equal value but opposite sign) is either summed
        # or substracted at this point.
        # When summed (e.g. for rho in mode m=0), the correction cancels out
        # and so the actual value of the coefficient does not matter.
        # When subtracted (e.g. for rho in mode m=1), setting the coefficient
        # to 0 ensure that the correction does not modify the deposition of
        # particles close to the axis.
        self.ruyten_linear_coef = np.concatenate(
                                    (np.array([0.]), self.ruyten_linear_coef) )
        self.ruyten_cubic_coef = np.concatenate(
                                    (np.array([0.]), self.ruyten_cubic_coef) )

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
        # Allocate the PML fields if needed
        if self.use_pml:
            self.Er_pml = np.zeros( (Nz, Nr), dtype='complex' )
            self.Et_pml = np.zeros( (Nz, Nr), dtype='complex' )
            self.Br_pml = np.zeros( (Nz, Nr), dtype='complex' )
            self.Bt_pml = np.zeros( (Nz, Nr), dtype='complex' )

        # Check whether the GPU should be used
        self.use_cuda = use_cuda

        # Replace the invvol array as well as the Ruyten coefficients by an array
        # on the GPU, when using cuda
        if self.use_cuda :
            self.d_invvol = cupy.asarray( self.invvol )
            self.d_ruyten_linear_coef = cupy.asarray( self.ruyten_linear_coef )
            self.d_ruyten_cubic_coef = cupy.asarray( self.ruyten_cubic_coef )

    @property
    def z(self):
        """Returns the 1d array of z, when the user queries self.z"""
        return( self.zmin + (0.5+np.arange(self.Nz))*self.dz )

    @property
    def r(self):
        """Returns the 1d array of r, when the user queries self.r"""
        return( self.rmin + (0.5+np.arange(self.Nr))*self.dr )

    def send_fields_to_gpu( self ):
        """
        Copy the fields to the GPU.

        After this function is called, the array attributes
        point to GPU arrays.
        """
        self.Er = cupy.asarray( self.Er )
        self.Et = cupy.asarray( self.Et )
        self.Ez = cupy.asarray( self.Ez )
        self.Br = cupy.asarray( self.Br )
        self.Bt = cupy.asarray( self.Bt )
        self.Bz = cupy.asarray( self.Bz )
        self.Jr = cupy.asarray( self.Jr )
        self.Jt = cupy.asarray( self.Jt )
        self.Jz = cupy.asarray( self.Jz )
        self.rho = cupy.asarray( self.rho )
        if self.use_pml:
            self.Er_pml = cupy.asarray( self.Er_pml )
            self.Et_pml = cupy.asarray( self.Et_pml )
            self.Br_pml = cupy.asarray( self.Br_pml )
            self.Bt_pml = cupy.asarray( self.Bt_pml )

    def receive_fields_from_gpu( self ):
        """
        Receive the fields from the GPU.

        After this function is called, the array attributes
        are accessible by the CPU again.
        """
        self.Er = self.Er.get()
        self.Et = self.Et.get()
        self.Ez = self.Ez.get()
        self.Br = self.Br.get()
        self.Bt = self.Bt.get()
        self.Bz = self.Bz.get()
        self.Jr = self.Jr.get()
        self.Jt = self.Jt.get()
        self.Jz = self.Jz.get()
        self.rho = self.rho.get()
        if self.use_pml:
            self.Er_pml = self.Er_pml.get()
            self.Et_pml = self.Et_pml.get()
            self.Br_pml = self.Br_pml.get()
            self.Bt_pml = self.Bt_pml.get()

    def erase( self, fieldtype ):
        """
        Sets the field `fieldtype` to zero on the interpolation grid

        Parameter
        ---------
        fieldtype : string
            A string which represents the kind of field to be erased
            (either 'E', 'B', 'J', 'rho')
        """
        if self.use_cuda:
            # Obtain the cuda grid
            dim_grid, dim_block = cuda_tpb_bpg_2d( self.Nz, self.Nr )

            # Erase the arrays on the GPU
            if fieldtype == 'rho':
                cuda_erase_scalar[dim_grid, dim_block](self.rho)
            elif fieldtype == 'J':
                cuda_erase_vector[dim_grid, dim_block](
                      self.Jr, self.Jt, self.Jz)
            elif fieldtype == 'E':
                cuda_erase_vector[dim_grid, dim_block](
                      self.Er, self.Et, self.Ez)
            elif fieldtype == 'B':
                cuda_erase_vector[dim_grid, dim_block](
                      self.Br, self.Bt, self.Bz)
            else:
                raise ValueError('Invalid string for fieldtype: %s'%fieldtype)
        else :
            # Erase the arrays on the CPU
            if fieldtype == 'rho':
                self.rho[:,:] = 0.
            elif fieldtype == 'J':
                self.Jr[:,:] = 0.
                self.Jt[:,:] = 0.
                self.Jz[:,:] = 0.
            elif fieldtype == 'E' :
                self.Er[:,:] = 0.
                self.Et[:,:] = 0.
                self.Ez[:,:] = 0.
            elif fieldtype == 'B' :
                self.Br[:,:] = 0.
                self.Bt[:,:] = 0.
                self.Bz[:,:] = 0.
            else :
                raise ValueError('Invalid string for fieldtype: %s'%fieldtype)

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
        if self.use_cuda :
            # Perform division on the GPU
            dim_grid, dim_block = cuda_tpb_bpg_2d( self.Nz, self.Nr )

            if fieldtype == 'rho':
                cuda_divide_scalar_by_volume[dim_grid, dim_block](
                        self.rho, self.d_invvol )
            elif fieldtype == 'J':
                cuda_divide_vector_by_volume[dim_grid, dim_block](
                        self.Jr, self.Jt, self.Jz, self.d_invvol )
            else:
                raise ValueError('Invalid string for fieldtype: %s'%fieldtype)
        else :
            # Perform division on the CPU
            if fieldtype == 'rho':
                self.rho *= self.invvol[np.newaxis,:]
            elif fieldtype == 'J':
                self.Jr *= self.invvol[np.newaxis,:]
                self.Jt *= self.invvol[np.newaxis,:]
                self.Jz *= self.invvol[np.newaxis,:]
            else:
                raise ValueError('Invalid string for fieldtype: %s'%fieldtype)
