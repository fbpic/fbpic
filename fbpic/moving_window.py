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
    - ncells_damp : number of cells in which the fields are damped

    Methods
    -------
    - move : shift the moving window by v*dt
    - damp : set the fields progressively to zero at the left end of the box
    """
    
    def __init__( self, zmin=0, v=c, ncells_zero=1,
                 ncells_damp=1, damp_shape='cos', gradual_damp_EB=True ) :
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
            Either 'None', 'linear', 'sin', 'cos'

        gradual_damp_EB : bool, optional
            Whether to gradually damp the fields EB
            If False, no damping at all will be applied to the fields E and B
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
            self.damp_array_J = np.ones(ncells_damp)
        elif damp_shape == 'linear' :
            self.damp_array_J = np.linspace(0, 1, ncells_damp)
        elif damp_shape == 'sin' :
            self.damp_array_J = np.sin( np.linspace(0, np.pi/2, ncells_damp) )
        elif damp_shape == 'cos' :
            self.damp_array_J = 0.5-0.5*np.cos(
                np.linspace(0, np.pi, ncells_damp) )
        else :
            raise ValueError("Invalid string for damp_shape : %s" %damp_shape)

        # Create the damping array for the E and B fields
        self.damp_array_EB = np.ones(ncells_damp)
        if gradual_damp_EB :
            # Contrary to the fields rho and J which are recalculated
            # at each timestep, the fields E and B accumulate damping
            # over the successive timesteps. Therefore, the damping on
            # E and B should be lighter. The following formula ensures
            # (for a static field) that the successive applications of
            # damping result in the same damping shape as for J.
            self.damp_array_EB[:-1] = \
              self.damp_array_J[:-1]/self.damp_array_J[1:]
        
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
            self.shift_fields( fld )
    
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
        Set the currents progressively to zero, at the right
        end of the moving window.

        This is done by multiplying the self.ncells_damp first cells
        of the field array (along z) by self.damp_array_J

        NB : The fields E and B are not damped with this function but
        are damped by shift_interp_field in the function move.

        Parameters
        ----------
        grid : a list of InterpolationGrid objects
            (one element per azimuthal mode)
            Contains the field data on the interpolation grid

        fieldtype : string
            A string indicating which field to damp
        """
        # Extract the length of the grid
        Nm = len(grid)

        # Loop over the azimuthal modes
        for m in range( Nm ) :
            # Choose the right field to damp
            if fieldtype == 'J' :
                damp_field( grid[m].Jr, self.damp_array_J, self.ncells_damp )
                damp_field( grid[m].Jt, self.damp_array_J, self.ncells_damp )
                damp_field( grid[m].Jz, self.damp_array_J, self.ncells_damp )
            elif fieldtype == 'rho' :
                damp_field( grid[m].rho, self.damp_array_J, self.ncells_damp )
            else :
                raise ValueError("Invalid string for fieldtype : %s" %fieldtype)

            
    def shift_fields( self, fld ) :
        """
        Shift all the fields in the object 'fld'
        
        The fields on the interpolation grid are shifted by one cell in z
        The corresponding fields on the spectral grid are calculated through FFTs
    
        Parameter
        ---------
        fld : a Fields object
            Contains the fields to be shifted 
        """
        # Shift the fields on the interpolation grid
        for m in range(fld.Nm) :
            self.shift_interp_grid( fld.interp[m] )
    
        # Shift the fields on the spectral grid
        for m in range(fld.Nm) :
            self.shift_spect_grid( fld.spect[m], fld.trans[m] )

            
    def shift_interp_grid( self, grid, shift_currents=False ) :
        """
        Shift the interpolation grid by one cell

        Parameter
        ---------
        
        shift_currents : bool, optional
            Whether to also shift the currents
            Default : False, since the currents are recalculated from
            scratch at each PIC cycle 
    
        Parameters
        ----------
        grid : an InterpolationGrid corresponding to one given azimuthal mode 
            Contains the values of the fields on the interpolation grid,
            and is modified by this function.

        shift_currents : bool, optional
            Whether to also shift the currents
            Default : False, since the currents are recalculated from
            scratch at each PIC cycle
        """
        # Modify the values of the corresponding z's 
        grid.z += grid.dz
        grid.zmin += grid.dz
        grid.zmax += grid.dz
    
        # Shift all the fields
        self.shift_interp_field( grid.Er )
        self.shift_interp_field( grid.Et )
        self.shift_interp_field( grid.Ez )
        self.shift_interp_field( grid.Br )
        self.shift_interp_field( grid.Bt )
        self.shift_interp_field( grid.Bz )
        if shift_currents :
            self.shift_interp_field( grid.Jr )
            self.shift_interp_field( grid.Jt )
            self.shift_interp_field( grid.Jz )
            self.shift_interp_field( grid.rho )


    def shift_spect_grid( self, grid, trans, shift_currents=False ) :
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
        
        shift_currents : bool, optional
            Whether to also shift the currents
            Default : False, since the currents are recalculated from
            scratch at each PIC cycle 
        """
        # Shift all the fields
        self.shift_spect_field( grid.Ep, trans )
        self.shift_spect_field( grid.Em, trans )
        self.shift_spect_field( grid.Ez, trans )
        self.shift_spect_field( grid.Bp, trans )
        self.shift_spect_field( grid.Bm, trans )
        self.shift_spect_field( grid.Bz, trans )
        # Also shift rho_prev since it is not recalculated at each PIC cycle
        self.shift_spect_field( grid.rho_prev, trans )
        if shift_currents :
            self.shift_spect_field( grid.rho_next, trans ) 
            self.shift_spect_field( grid.Jp, trans )
            self.shift_spect_field( grid.Jm, trans )
            self.shift_spect_field( grid.Jz, trans )

        
    def shift_spect_field( self, field_array, trans ) :
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
        """
        # Copy the array into the FFTW buffer
        trans.spect_buffer_r[:,:] = field_array[:,:]
        # Perform the inverse FFT
        trans.ifft_r()
    
        # Shift the the values in the buffer
        self.shift_interp_field( trans.interp_buffer_r )
    
        # Perform the FFT (back to spectral space)
        trans.fft_r()
        # Copy the buffer into the fields
        field_array[:,:] = trans.spect_buffer_r[:,:]

    def shift_interp_field( self, field_array ) :
        """
        Shift the field 'field_array' by one cell (backwards)
        
        Parameters
        ----------
        field_array : 2darray of complexs
            Contains the value of the fields, and is modified by this function
        """
        # Transfer the values to one cell before
        field_array[:-1,:] = field_array[1:,:]
        # Apply damping, using the EB array
        damp_field( field_array, self.damp_array_EB, self.ncells_damp )
        # Zero out the new fields
        field_array[-self.ncells_zero:,:] = 0.
        
        
# ---------------------------------------
# Utility functions for the moving window
# ---------------------------------------

def damp_field( field_array, damp_array, n ) :
    """
    Multiply the n first cells and last n cells of field_array
    by damp_array, along the first axis.

    Parameters
    ----------
    field_array : 2darray of complexs
    damp_array : 2darray of reals
    n : int
    """
    field_array[:n,:] = damp_array[:,np.newaxis] * field_array[:n,:]
    field_array[-n:,:] = damp_array[::-1,np.newaxis] * field_array[-n:,:]


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
    
