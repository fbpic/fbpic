# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines methods to directly inject the laser in the Simulation box
"""
import numpy as np

def add_laser_direct_envelope( sim, laser_profile, boost ):
    """
    Add a laser pulse envelope in the simulation, by directly replacing any other A field.

    Parameters:
    -----------
    sim: a Simulation object
        The structure that contains the simulation.

    laser_profile: a valid gaussian laser profile object
        Laser profiles can be imported from fbpic.lpa_utils.laser

    boost: a BoostConverter object or None
       Contains the information about the boost to be applied
    """
    print("Initializing laser pulse envelope on the mesh...")
    
    # Check that no other laser simulation has been already added
    if sim.fld.use_envelope:
        raise ValueError("Another laser profile has already been added")
    sim.fld.activate_envelope_model(laser_profile.k0)

    # Get the local azimuthally-decomposed laser fields A and dtA on each proc
    laser_A, laser_dtA = get_laser_A_dtA( sim, laser_profile, boost )
    for m in sim.fld.envelope_mode_numbers:
        sim.fld.envelope_interp[m].A[:,:] = laser_A[:,:,m]
        sim.fld.envelope_interp[m].dtA[:,:] = laser_dtA[:,:,m]

    print("Done.\n")


def get_laser_A_dtA( sim, laser_profile, boost ):
    """
    Calculate the laser A and dtA envelope fields on the points of the interpolation
    grid of sim, and decompose into azimuthal modes.

    Parameters:
    -----------
    sim: a Simulation object
        The structure that contains the simulation.

    boost: a BoostConverter object or None
       Contains the information about the boost to be applied

    Returns:
    --------
    A_m, dtA_m: 3d_arrays of complexs
        Arrays of size (Nz, Nr, 2*Nm), that represent the
        azimuthally-decomposed envelope fields of the laser. The first Nm points
        along the last axis correspond to the values in the modes m>= 0.
    """
    # Initialize a grid on which the laser amplitude should be calculated
    # - Get the 1d arrays of the grid
    z = sim.fld.interp[0].z
    r = sim.fld.interp[0].r
    # - Sample the field at 2*Nm values of theta, in order to
    #   perform the azimuthal decomposition of the fields
    ntheta = 2*sim.fld.Nm
    theta = (2*np.pi/ntheta) * np.arange( ntheta )
    # - Get corresponding 3d arrays
    z_3d, r_3d, theta_3d = np.meshgrid( z, r, theta, indexing='ij' )
    cos_theta_3d = np.cos(theta_3d)
    sin_theta_3d = np.sin(theta_3d)
    x_3d = r_3d * cos_theta_3d
    y_3d = r_3d * sin_theta_3d
    

    # For boosted-frame: convert time and position to the lab-frame
    if boost is not None:
        raise NotImplementedError("The envelope model is not implemented \
                            for boosted-frame simulations.")
    else:
        zlab_3d = z_3d
        tlab = sim.time

    # Evaluate the scalar A and dtA fields in these positions
    A_3d, dtA_3d = laser_profile.A_field( x_3d, y_3d, zlab_3d, tlab )

    # For boosted-frame: scale the lab-frame value of the fields to
    # the corresponding boosted-frame value
    if boost is not None:
        scale_factor = 1./( boost.gamma0*(1+boost.beta0) )
        A_3d *= scale_factor
        dtA_3d *= scale_factor

    # Perform the azimuthal decomposition of the Er and Et fields
    # and add them to the mesh
    A_m_3d = np.fft.ifft(A_3d, axis=-1)
    dtA_m_3d = np.fft.ifft(dtA_3d, axis=-1)

    return( A_m_3d, dtA_m_3d )
