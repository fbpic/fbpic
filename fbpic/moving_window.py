"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to implement the moving window.
"""
import numpy as np
from scipy.constants import c
from particles import Particles

def move_window( fld, ptcl, p_nz, time ) :
    """
    Check if the window should be moved, and if yes shift the fields,
    and add new particles

    Parameters
    ----------
    fld : a Fields object
        Contains the fields data of the simulation

    ptcl : a list of Particles objects
        Contains the particles data for each species

    p_nz : int
        Number of macroparticles per cell along the z direction

    time : float (in seconds)
        The global time of the simulation
    """
    
    # Move the window as long as zmin < ct
    while fld.interp[0].zmin < c*time :
        
        # Shift the fields
        shift_fields( fld )

        # Extract a few quantities of the new (shifted) grid
        zmin = fld.interp[0].zmin
        zmax = fld.interp[0].zmax
        dz = fld.interp[0].dz
    
        # Now that the grid has moved, remove the particles that are
        # outside of it, and add new particles in the next-to-last cell
        for species in ptcl :
            clean_outside_particles( species, zmin-0.5*dz )
            add_particles( species, zmax-1.5*dz, zmax-0.5*dz, p_nz )

def shift_fields(fld) :
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
        shift_interp_grid( fld.interp[m] )

    # Shift the fields on the spectral grid
    for m in range(fld.Nm) :
        shift_spect_grid( fld.spect[m], fld.trans[m] )
    

def shift_interp_grid( grid, shift_currents=False ) :
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
    """
    # Modify the values of the corresponding z's 
    grid.z += grid.dz
    grid.zmin += grid.dz
    grid.zmax += grid.dz

    # Shift all the fields
    shift_interp_field( grid.Er )
    shift_interp_field( grid.Et )
    shift_interp_field( grid.Ez )
    shift_interp_field( grid.Br )
    shift_interp_field( grid.Bt )
    shift_interp_field( grid.Bz )
    if shift_currents :
        shift_interp_field( grid.Jr )
        shift_interp_field( grid.Jt )
        shift_interp_field( grid.Jz )
        shift_interp_field( grid.rho )

def shift_spect_grid( grid, trans, shift_currents=False ) :
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
    shift_spect_field( grid.Ep, trans )
    shift_spect_field( grid.Em, trans )
    shift_spect_field( grid.Ez, trans )
    shift_spect_field( grid.Bp, trans )
    shift_spect_field( grid.Bm, trans )
    shift_spect_field( grid.Bz, trans )
    # Also shift rho_prev since it is not recalculated at each PIC cycle
    shift_spect_field( grid.rho_prev, trans )
    if shift_currents :
        shift_spect_field( grid.rho_next, trans ) 
        shift_spect_field( grid.Jp, trans )
        shift_spect_field( grid.Jm, trans )
        shift_spect_field( grid.Jz, trans )
    
        
def shift_spect_field( field_array, trans ) :
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
    shift_interp_field( trans.interp_buffer_r )

    # Perform the FFT (back to spectral space)
    trans.fft_r()
    # Copy the buffer into the fields
    field_array[:,:] = trans.spect_buffer_r[:,:]
     
    
def shift_interp_field( field_array, n_cells_zero=10 ) :
    """
    Shift the field 'field_array' by one cell (backwards)
    
    Parameters
    ----------
    field_array : 2darray of complexs
        Contains the value of the fields, and is modified by this function
        
    n_cells_zero : int
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
    
