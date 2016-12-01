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

def weights(x, invdx, offset, Nx, direction, shape_order):
    """
    Return the matrix indices and the shape factors for a given direction
    and a given shape order.

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

    shape_order : int
        Order of the shape factor.
        Either 1, 2 or 3


    -------
    A tuple containing :

    i: 1D array of 1D arrays
        This array contains the indicies for each particle
        the first index is the respective cell index,
        the second for the respective particle

    S: 1D array of 1D arrays
        This array contains the shape factors for each particle
        the first index is the respective cell index,
        the second for the respective particle

    """

    # Positions of the particles, in the cell unit
    x_cell = invdx*(x - offset) - 0.5

    # Initialize empty arrays of the correct size
    i = []
    S = []

    # Indices and shapes
    if shape_order == 1:
        i.append(np.floor(x_cell).astype('int'))
        i.append(i[0] + 1)
        # Linear weight
        S.append(i[1] - x_cell)
        S.append(x_cell - i[0])
    elif shape_order == 3:
        i.append(np.floor(x_cell).astype('int') - 1)
        i.append(i[0] + 1)
        i.append(i[1] + 1)
        i.append(i[2] + 1)
        # Qubic Weights
        S.append(-1./6. * ((x_cell-i[0])-2)**3)
        S.append(1./6. * (3*((x_cell-i[1])**3) - 6*((x_cell-i[1])**2)+4))
        S.append(1./6. * (3*((i[2]-x_cell)**3) - 6*((i[2]-x_cell)**2)+4))
        S.append(-1./6. * ((i[3]-x_cell)-2)**3)
    else:
        raise ValueError("shapes other than linear and cubic are not supported yet.")

    # Periodic boundary conditions
    # Counter to go through the indices
    # Note: We cycle through the indices for the cells.
    # in i[0] we have the indicies of the cells for all the particles that
    # lie the most to the left. In i[1] the cell indices next to it for each
    # particle.
    counter = 0
    if direction == 'z':
        for index in i:
            # Lower Bound Periodic
            i[counter] = np.where(i[counter] < 0, i[counter]+Nx, i[counter])
            # Upper Bound Periodic
            i[counter] = np.where(i[counter] > Nx-1, i[counter]-Nx, i[counter])
            counter += 1

    elif direction == 'r':
        for index in i:
            # Upper bound : absorbing
            i[counter] = np.where(index > Nx-1, Nx-1, i[counter])
            counter += 1
            # Note: The lower bound index shift for r is done in the gather
            # and deposit methods because the sign changes .
            # This avoids using specific guard cells.
    else:
        raise ValueError("Unrecognized `direction` : %s" % direction)
    # Return the result
    return(np.asarray(i, dtype=np.int64), np.asarray(S, dtype=np.float64))

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
