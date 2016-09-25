# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the optimized particles methods that use numba on a CPU
"""
import numpy as np

# -----------------------
# Particle shapes utility
# -----------------------

def linear_weights(x, invdx, offset, Nx, direction) :
    """
    Return the matrix indices and the shape factors, for linear shapes.

    The boundary conditions are determined by direction :
    - direction='z' : periodic conditions
    - direction='r' : absorbing at the upper bound,
                      using guard cells at the lower bounds
    NB : the guard cells are not technically part of the field arrays 
    The weight deposited in the guard cells are added positively or
    negatively to the lower cell of the field array, depending on the
    exact field considered.

    Parameters
    ----------
    x : 1darray of floats (in meters)
        Array of particle positions along a given direction
        (one element per macroparticle)

    invdx : float (in meters^-1)
        Inverse of the grid step along the considered direction

    offset : float (in meters)
        Position of the edge of the simulation box,
        along the direction considered

    Nx : int
        Number of gridpoints along the considered direction

    direction : string
        Determines the boundary conditions. Either 'r' or 'z'

    Returns
    -------
    A tuple containing :
    
    i_lower, i_upper : 1darray of integers
        (one element per macroparticle)
        Contains the index of the cell immediately below each
        macroparticle, along the considered axis
    i_upper : 1darray of integers
        (one element per macroparticle)
        Contains the index of the cell immediately above each
        macroparticle, along the considered axis
    S_lower : 1darray of floats
        (one element per macroparticle)
        Contains the weight for the lower cell, for each macroparticle.
        The weight for the upper cell is just 1-S_lower.
    S_upper : 1darray of floats
    """

    # Positions of the particles, in the cell unit
    x_cell =  invdx*(x - offset) - 0.5
    
    # Index of the uppper and lower cell
    i_lower = np.floor( x_cell ).astype('int')  
    i_upper = i_lower + 1

    # Linear weight
    S_lower = i_upper - x_cell
    S_upper = x_cell - i_lower
    
    # Treat the boundary conditions
    if direction=='r' :   # Radial boundary condition
        # Lower bound : place the weight in the guard cells
        out_of_bounds =  (i_lower < 0)
        S_guard = np.where( out_of_bounds, S_lower, 0. )
        S_lower = np.where( out_of_bounds, 0., S_lower )
        i_lower = np.where( out_of_bounds, 0, i_lower )
        # Upper bound : absorbing
        i_lower = np.where( i_lower > Nx-1, Nx-1, i_lower )
        i_upper = np.where( i_upper > Nx-1, Nx-1, i_upper )
        # Return the result
        return( i_lower, i_upper, S_lower, S_upper, S_guard )
        
    elif direction=='z' :  # Longitudinal boundary condition
        # Lower bound : periodic
        i_lower = np.where( i_lower < 0, i_lower+Nx, i_lower )
        i_upper = np.where( i_upper < 0, i_upper+Nx, i_upper )
        # Upper bound : periodic
        i_lower = np.where( i_lower > Nx-1, i_lower-Nx, i_lower )
        i_upper = np.where( i_upper > Nx-1, i_upper-Nx, i_upper )
        # Return the result
        return( i_lower, i_upper, S_lower, S_upper )

    else :
        raise ValueError("Unrecognized `direction` : %s" %direction)

# ----------------------------
# Angle initialization utility
# ----------------------------

def unalign_angles( thetap, Npz, Npr, method='irrational' ) :
    """
    Shift the angles so that the particles are
    not all aligned along the arms of a star transversely

    The fact that the particles are all aligned can produce
    numerical artefacts, especially if the polarization of the laser
    is aligned with this direction.

    Here, for each position in r and z, we add the *same*
    shift for all the Nptheta particles that are at this position.
    (This preserves the fact that certain modes are 0 initially.)
    How this shift varies from one position to another depends on
    the method chosen.

    Parameters
    ----------
    thetap : 3darray of floats
        An array of shape (Npr, Npz, Nptheta) containing the angular
        positions of the particles, and which is modified by this function.

    Npz, Npr : ints
        The number of macroparticles along the z and r directions
    
    method : string
        Either 'random' or 'irrational'
    """
    # Determine the angle shift
    if method == 'random' :
        angle_shift = 2*np.pi*np.random.rand(Npz, Npr)
    elif method == 'irrational' :
        # Subrandom sequence, by adding irrational number (sqrt(2) and sqrt(3))
        # This ensures that the sequence does not wrap around and induce
        # correlations
        shiftr = np.sqrt(2)*np.arange(Npr)
        shiftz = np.sqrt(3)*np.arange(Npz)
        angle_shift = 2*np.pi*( shiftz[:,np.newaxis] + shiftr[np.newaxis,:] )
        angle_shift = np.mod( angle_shift, 2*np.pi )
    else :
        raise ValueError(
      "method must be either 'random' or 'irrational' but is %s" %method )

    # Add the angle shift to thetap
    # np.newaxis ensures that the angles that are at the same positions
    # in r and z have the same shift
    thetap[:,:,:] = thetap[:,:,:] + angle_shift[:,:, np.newaxis]
