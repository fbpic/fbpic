"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to implement the moving window.
"""
import numpy as np
from scipy.constants import c
from particles import Particles

class MovingWindow(object) :
    """
    Class that contains the moving window's variables and methods

    One major problem of the moving window in a spectral code is that \
    the fields `wrap around` the moving window, .i.e the fields that
    disappear at the left end reappear at the right end, as a consequence
    of the periodicity of the Fourier transform.

    Attributes
    ----------
    - v : speed of the moving window
    - ncells_zero : number of cells in which the fields are set to zero
    - ncells_damp : number of cells in which the currents are damped
    - damp_array : 1darray by which the density and currents get multiplied

    Methods
    -------
    - move : shift the moving window by v*dt
    - damp : set the density and current progressively to zero at the
             left end of the box
    """
    
    def __init__( self, zmin=0, v=c, ncells_zero=1,
                 ncells_damp=1, damp_shape='None' ) :
        """
        Initializes a moving window object.

        Parameters
        ----------
        zmin : float (meters), optional
            The starting position of the moving window
        
        v : float (meters per seconds), optional
            The speed of the moving window
        
        ncells_zero : int, optional
            Number of cells in which the fields are set to zero,
            at the right end of the box

        ncells_damp : int, optional
            Number of cells over which the currents and density are
            progressively set to 0, at the left end of the box

        damp_shape : string, optional
            How to damp the fields
            Either 'None', 'linear', 'sin'
        """
        # Attach position and speed
        self.zmin = zmin
        self.v = v

        # Attach numerical parameters
        self.ncells_zero = ncells_zero
        self.ncells_damp = ncells_damp
        self.damp_shape = damp_shape

        # Create the damping array for the density and currents
        if damp_shape == 'None' :
            self.damp_array = np.ones(ncells_damp)
        elif damp_shape == 'linear' :
            self.damp_array = np.linspace(0, 1, ncells_damp)
        elif damp_shape == 'sin' :
            self.damp_array = np.sin( np.linspace(0, np.pi/2, ncells_damp) )
        else :
            raise ValueError("Invalid string for damp_shape : %s" %damp_shape)
        
    def move( self, fld, ptcl, p_nz, dt ) :
        """
        Check whether the grid should be moved.
        If yes shift the fields, and add new particles
    
        Parameters
        ----------
        fld : a Fields object
            Contains the fields data of the simulation
    
        ptcl : a list of Particles objects
            Contains the particles data for each species
    
        p_nz : int
            Number of macroparticles per cell along the z direction
    
        dt : float (in seconds)
            Timestep of the simulation
        """
    
        # Move the position of the moving window object
        self.zmin = self.zmin + self.v*dt
        
        # As long as the window is ahead of the grid,
        # shift the grid and the particles
        while fld.interp[0].zmin < self.zmin :
            
            # Shift the fields
            shift_fields( fld, self.ncells_zero )
    
            # Extract a few quantities of the new (shifted) grid
            zmin = fld.interp[0].zmin
            zmax = fld.interp[0].zmax
            dz = fld.interp[0].dz
    
            # Now that the grid has moved, remove the particles that are
            # outside of it, and add new particles in the next-to-last cell
            for species in ptcl :
                clean_outside_particles( species, zmin-0.5*dz )
                add_particles( species, zmax-1.5*dz, zmax-0.5*dz, p_nz )

    def damp( self, grid, fieldtype ) :
        """
        Set the fields progressively to zero, at the right
        end of the moving window.

        This is done by multiplying the self.ncells_damp first cells
        of the field array (along z) by self.damp_array

        Parameters
        ----------
        grid : a list of InterpolationGrid objects
            (one element per azimuthal mode)
            Contains the field data on the interpolation grid

        fieldtype : string
            A string indicating which field to smooth
        """
        # Extract the length of the grid
        Nm = len(grid)

        # Loop over the azimuthal modes
        for m in range( Nm ) :
            # Choose the right field to damp
            if fieldtype == 'J' :
                damp_field( grid[m].Jr, self.damp_array, self.ncells_damp )
                damp_field( grid[m].Jt, self.damp_array, self.ncells_damp )
                damp_field( grid[m].Jz, self.damp_array, self.ncells_damp )
            elif fieldtype == 'rho' :
                damp_field( grid[m].rho, self.damp_array, self.ncells_damp )


# ---------------------------------------
# Utility functions for the moving window
# ---------------------------------------

def damp_field( field_array, damp_array, n ) :
    """
    Multiply the n first cells of field_array by damp_array,
    along the first axis.

    Parameters
    ----------
    field_array : 2darray of complexs
    damp_array : 2darray of reals
    n : int
    """
    field_array[:n,:] = damp_array[:,np.newaxis] * field_array[:n,:] 

                
def shift_fields(fld, ncells_zero ) :
    """
    Shift all the fields in the object 'fld'
    
    The fields on the interpolation grid are shifted by one cell in z
    The corresponding fields on the spectral grid are calculated through FFTs

    Parameter
    ---------
    fld : a Fields object
        Contains the fields to be shifted

    ncells_zero : int
        The number of cells to set to zero at the right end of the box    
    """
    # Shift the fields on the interpolation grid
    for m in range(fld.Nm) :
        shift_interp_grid( fld.interp[m], ncells_zero )

    # Shift the fields on the spectral grid
    for m in range(fld.Nm) :
        shift_spect_grid( fld.spect[m], fld.trans[m], ncells_zero )    

def shift_interp_grid( grid, ncells_zero, shift_currents=False ) :
    """
    Shift the interpolation grid by one cell

    shift_currents : bool, optional
        Whether to also shift the currents
        Default : False, since the currents are recalculated from
        scratch at each PIC cycle 
    
    Parameters
    ----------
    grid : an InterpolationGrid corresponding to one given azimuthal mode 
        Contains the values of the fields on the interpolation grid,
        and is modified by this function.
        
    ncells_zero : int
        The number of cells to set to zero at the right end of the box
    """
    # Modify the values of the corresponding z's 
    grid.z += grid.dz
    grid.zmin += grid.dz
    grid.zmax += grid.dz

    # Shift all the fields
    shift_interp_field( grid.Er, ncells_zero )
    shift_interp_field( grid.Et, ncells_zero )
    shift_interp_field( grid.Ez, ncells_zero )
    shift_interp_field( grid.Br, ncells_zero )
    shift_interp_field( grid.Bt, ncells_zero )
    shift_interp_field( grid.Bz, ncells_zero )
    if shift_currents :
        shift_interp_field( grid.Jr, ncells_zero )
        shift_interp_field( grid.Jt, ncells_zero )
        shift_interp_field( grid.Jz, ncells_zero )
        shift_interp_field( grid.rho, ncells_zero )

def shift_spect_grid( grid, trans, ncells_zero, shift_currents=False ) :
    """
    Calculate the spectral grid corresponding to a shifted
    interpolation grid

    Parameters
    ----------
    grid : a SpectralGrid object corresponding to one given azimuthal mode
        Contains the values of the fields on the spectral grid,
        and is modified by this function.

    trans : a SpectralTransform object
        Needed to perform the FFT transforms

    ncells_zero : int
        The number of cells to set to zero at the right end of the box
        
    shift_currents : bool, optional
        Whether to also shift the currents
        Default : False, since the currents are recalculated from
        scratch at each PIC cycle 
    """
    # Shift all the fields
    shift_spect_field( grid.Ep, trans, ncells_zero )
    shift_spect_field( grid.Em, trans, ncells_zero )
    shift_spect_field( grid.Ez, trans, ncells_zero )
    shift_spect_field( grid.Bp, trans, ncells_zero )
    shift_spect_field( grid.Bm, trans, ncells_zero )
    shift_spect_field( grid.Bz, trans, ncells_zero )
    # Also shift rho_prev since it is not recalculated at each PIC cycle
    shift_spect_field( grid.rho_prev, trans, ncells_zero )
    if shift_currents :
        shift_spect_field( grid.rho_next, trans, ncells_zero ) 
        shift_spect_field( grid.Jp, trans, ncells_zero )
        shift_spect_field( grid.Jm, trans, ncells_zero )
        shift_spect_field( grid.Jz, trans, ncells_zero )
    
        
def shift_spect_field( field_array, trans, ncells_zero=1 ) :
    """
    Calculate the field in spectral space that corresponds to
    a shifted field on the interpolation grid
    
    This is done through the succession of an IFFT,
    a shift along z and an FFT
    (no Hankel transform needed since only the z direction
    is concerned by the moving window )
    
    Parameters
    ----------
    field_array : 2darray of complexs
        Contains the value of the fields, and is modified by this function 

    trans : a SpectralTransform object
        Needed to perform the FFT transforms
        
    n_cells_zero : int, optional
        The number of cells to set to zero at the right end of the box
    """
    # Copy the array into the FFTW buffer
    trans.spect_buffer_r[:,:] = field_array[:,:]
    # Perform the inverse FFT
    trans.ifft_r()

    # Shift the the values in the buffer
    shift_interp_field( trans.interp_buffer_r, ncells_zero )

    # Perform the FFT (back to spectral space)
    trans.fft_r()
    # Copy the buffer into the fields
    field_array[:,:] = trans.spect_buffer_r[:,:]
     
    
def shift_interp_field( field_array, n_cells_zero=1 ) :
    """
    Shift the field 'field_array' by one cell (backwards)
    
    Parameters
    ----------
    field_array : 2darray of complexs
        Contains the value of the fields, and is modified by this function
        
    n_cells_zero : int, optional
        The number of cells to set to zero at the right end of the box
    """
    # Transfer the values to one cell before
    field_array[:-1,:] = field_array[1:,:]
    # Zero out the new fields
    field_array[-n_cells_zero:,:] = 0.


def clean_outside_particles( species, zmin ) :
    """
    Removes the particles that are below `zmin`.

    Parameters
    ----------
    species : a Particles object
        Contains the data of this species

    zmin : float
        The lower bound under which particles are removed
    """

    # Select the particles that are still inside the box
    selec = ( species.z > zmin )

    # Keep only this selection, in the different arrays that contains the
    # particle properties (x, y, z, ux, uy, uz, etc...)
    # Instead of hard-coding x = x[selec], y=y[selec], etc... here we loop
    # over the particles attributes, and resize the attributes that are
    # arrays with one element per particles.
    # The advantage is that nothing needs to be added to this piece of code,
    # if a new particle attribute is later added in particles.py.

    # Loop over the attributes
    for key, attribute in vars(species).items() :
        # Detect if it is an array
        if type(attribute) is np.ndarray :
            # Detect if it has one element per particle
            if attribute.shape == ( species.Ntot ,) :
                # Affect the resized array to the object
                setattr( species, key, attribute[selec] )

    # Adapt the number of particles accordingly
    species.Ntot = len( species.w )

def add_particles( species, zmin, zmax, Npz ) :
    """
    Create new particles between zmin and zmax, and add them to `species`

    Parameters
    ----------
    species : a Particles object
       Contains the particle data of that species

    zmin, zmax : floats (meters)
       The positions between which the new particles are created

    Npz : int
       The total number of particles to be added along the z axis
       (The number of particles along r and theta is the same as that of
       `species`)
    """

    # Take the angle of the last particle as a global shift in theta,
    # in order to prevent the successively-added particles from being aligned
    global_theta = np.angle( species.x[-1] + 1.j*species.y[-1] )
    # Create the particles that will be added
    new_ptcl = Particles( species.q, species.m, species.n,
        Npz, zmin, zmax, species.Npr, species.rmin, species.rmax,
        species.Nptheta, species.dt, global_theta )

    # Add the properties of these new particles to species object
    # Loop over the attributes of the species
    for key, attribute in vars(species).items() :
        # Detect if it is an array
        if type(attribute) is np.ndarray :
            # Detect if it has one element per particle
            if attribute.shape == ( species.Ntot ,) :
                # Concatenate the attribute of species and of new_ptcl
                new_attribute = np.hstack(
                    ( getattr(species, key), getattr(new_ptcl, key) )  )
                # Affect the resized array to the species object
                setattr( species, key, new_attribute )

    # Add the number of new particles to the global count of particles
    species.Ntot += new_ptcl.Ntot
    
