"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to implement the moving window.
"""

def shift_window(sim) :
    # Check if it is the right time to shift the window

    # Shift the fields

    # Shift the particles

    
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
        shift_interp_grid( fld.spect[m] )
    

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
     
    
def shift_interp_field( field_array, n_cells_zero=1 ) :
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
    field_array[-n_cells_zero,:] = 0.
