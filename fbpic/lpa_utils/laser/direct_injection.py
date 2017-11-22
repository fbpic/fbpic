# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines methods to directly inject the laser in the Simulation box
"""

def add_laser_direct( fld, profile, fw_propagating, boost ):
    """
    Add a linearly-polarized laser pulse in the simulation

    Parameters:
    -----------
    TODO
    """
    # (Should mirror the particle space charge calculation)

    # Initialize a set of values to try in r and theta

    # Calculate the Ex and Ey

    # Transform them in azimuthally-decomposed Er and Etheta

    # Gather value onto a single grid
    # Go to spectral space

    # Calculate Ez, so that the field is divergence-free

    # Calculate B, from d_t B = curl(E)

    # Come back to real space
    # Scatter the values to all procs
    # Add the values to the local grid
    # Transform back to spectral space
    pass
