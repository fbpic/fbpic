# Copyright 2018, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the InterpolationGrid class.
"""
import numpy as np
from numba import cuda
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
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

    def __init__(self, Nz, Nr, m, zmin, zmax, rmax, use_cuda=False ) :
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

        use_cuda : bool, optional
            Wether to use the GPU or not
        """
        # Register the size of the arrays
        self.Nz = Nz
        self.Nr = Nr
        self.m = m

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
        # Cell volume (assuming an evenly-spaced grid)
        r = (0.5 + np.arange(Nr))*dr
        vol = np.pi*dz*( (r+0.5*dr)**2 - (r-0.5*dr)**2 )
        # Note: No Verboncoeur-type correction required
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
