# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines methods to directly inject the laser in the Simulation box
"""
import numpy as np
from scipy.constants import c
from fbpic.fields import Fields

def add_laser_direct_envelope( sim, laser_profile, boost ):
    """
    Add a laser pulse envelope in the simulation, by directly replacing any other A field.

    Note:
    -----
    Currently only Gaussian profile can be used
    (which must provide the *transverse electric field*)

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

    # Get the local azimuthally-decomposed laser fields A and dtA on each proc
    laser_A, laser_dtA = get_laser_A_dtA( sim, laser_profile, boost )
    for m in range(sim.fld.Nm):
        sim.fld.interp[m].A[:,:] = laser_A[:,:,m]
        sim.fld.interp[m].dtA[:,:] = laser_dtA[:,:,m]

    # Create a global field object across all subdomains, and copy the fields
    # (Calculating the self-consistent Ez and B is a global operation)
    global_Nz, _ = sim.comm.get_Nz_and_iz(
                    local=False, with_damp=True, with_guard=False )
    global_zmin, global_zmax = sim.comm.get_zmin_zmax(
                    local=False, with_damp=True, with_guard=False )
    global_fld = Fields( global_Nz, global_zmax,
            sim.fld.Nr, sim.fld.rmax, sim.fld.Nm, sim.fld.dt,
            zmin=global_zmin, n_order=sim.fld.n_order, use_cuda=False)
    # Gather the fields of the interpolation grid
    for m in range(sim.fld.Nm):
        for field in ['A', 'dtA']:
            local_array = getattr( sim.fld.interp[m], field )
            gathered_array = sim.comm.gather_grid_array(
                                local_array, with_damp=True)
            setattr( global_fld.interp[m], field, gathered_array )


    # Communicate the results from proc 0 to the other procs
    # and add it to the interpolation grid of sim.fld.
    # - First find the indices at which the fields should be added
    Nz_local, iz_start_local_domain = sim.comm.get_Nz_and_iz(
        local=True, with_damp=True, with_guard=False, rank=sim.comm.rank )
    _, iz_start_local_array = sim.comm.get_Nz_and_iz(
        local=True, with_damp=True, with_guard=True, rank=sim.comm.rank )
    iz_in_array = iz_start_local_domain - iz_start_local_array
    # - Then loop over modes and fields
    for m in range(sim.fld.Nm):
        for field in ['A', 'dtA']:
            # Get the local result from proc 0
            global_array = getattr( global_fld.interp[m], field )
            local_array = sim.comm.scatter_grid_array(
                                    global_array, with_damp=True)
            # Add it to the fields of sim.fld
            local_field = getattr( sim.fld.interp[m], field )
            local_field[ iz_in_array:iz_in_array+Nz_local, : ] += local_array

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
        zlab_3d = boost.gamma0*( z_3d + boost.beta0 * c * sim.time )
        tlab = boost.gamma0*( sim.time + (boost.beta0 * 1./c) * z_3d )
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
