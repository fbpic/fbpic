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

def add_laser_direct( sim, laser_profile, boost ):
    """
    Add a laser pulse in the simulation, by directly adding it to the mesh

    Note:
    -----
    Arbitrary laser profiles can be passed through `laser_profile`
    (which must provide the *transverse electric field*)
    For any profile:
    - The field is automatically decomposed into azimuthal modes
    - The Ez field is automatically calculated so as to ensure that div(E)=0
    - The B field is automatically calculated so as to ensure propagation
    in the right direction.

    Parameters:
    -----------
    sim: a Simulation object
        The structure that contains the simulation.

    laser_profile: a valid laser profile object
        Laser profiles can be imported from fbpic.lpa_utils.laser

    boost: a BoostConverter object or None
       Contains the information about the boost to be applied
    """
    if sim.comm.rank == 0:
        print("Initializing laser pulse on the mesh...")

    # Get the local azimuthally-decomposed laser fields Er and Et on each proc
    laser_Er, laser_Et = get_laser_Er_Et( sim, laser_profile, boost )
    # Save previous values on the grid, and replace them with the laser fields
    # (This is done in preparation for gathering among procs)
    saved_Er = []
    saved_Et = []
    for m in range(sim.fld.Nm):
        saved_Er.append( sim.fld.interp[m].Er.copy() )
        sim.fld.interp[m].Er[:,:] = laser_Er[:,:,m]
        saved_Et.append( sim.fld.interp[m].Et.copy() )
        sim.fld.interp[m].Et[:,:] = laser_Et[:,:,m]

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
        for field in ['Er', 'Et']:
            local_array = getattr( sim.fld.interp[m], field )
            gathered_array = sim.comm.gather_grid_array(
                                local_array, with_damp=True)
            setattr( global_fld.interp[m], field, gathered_array )

    # Now that the (gathered) laser fields are stored in global_fld,
    # copy the saved field back into the local grid
    for m in range(sim.fld.Nm):
        sim.fld.interp[m].Er[:,:] = saved_Er[m]
        sim.fld.interp[m].Et[:,:] = saved_Et[m]

    # Calculate the Ez and B fields on the global grid
    # (For a multi-proc simulation: only performed by the first proc)
    if sim.comm.rank == 0:
        calculate_laser_fields( global_fld, laser_profile.propag_direction )

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
        for field in ['Er', 'Et', 'Ez', 'Br', 'Bt', 'Bz']:
            # Get the local result from proc 0
            global_array = getattr( global_fld.interp[m], field )
            local_array = sim.comm.scatter_grid_array(
                                    global_array, with_damp=True)
            # Add it to the fields of sim.fld
            local_field = getattr( sim.fld.interp[m], field )
            local_field[ iz_in_array:iz_in_array+Nz_local, : ] += local_array

    if sim.comm.rank == 0:
        print("Done.\n")


def get_laser_Er_Et( sim, laser_profile, boost ):
    """
    Calculate the laser Er and Et fields on the points of the interpolation
    grid of sim, and decompose into azimuthal modes.

    Parameters:
    -----------
    sim: a Simulation object
        The structure that contains the simulation.

    boost: a BoostConverter object or None
       Contains the information about the boost to be applied

    Returns:
    --------
    Er_m, Er_t: 3d_arrays of complexs
        Arrays of size (Nz, Nr, 2*Nm), that represent the
        azimuthally-decomposed fields of the laser. The first Nm points
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

    # Evaluate the transverse Er and Et field at these position
    Ex_3d, Ey_3d = laser_profile.E_field( x_3d, y_3d, zlab_3d, tlab )
    Er_3d = cos_theta_3d * Ex_3d + sin_theta_3d * Ey_3d
    Et_3d = - sin_theta_3d * Ex_3d + cos_theta_3d * Ey_3d

    # For boosted-frame: scale the lab-frame value of the fields to
    # the corresponding boosted-frame value
    if boost is not None:
        scale_factor = 1./( boost.gamma0*(1+boost.beta0) )
        Er_3d *= scale_factor
        Et_3d *= scale_factor

    # Perform the azimuthal decomposition of the Er and Et fields
    # and add them to the mesh
    Er_m_3d = np.fft.ifft(Er_3d, axis=-1)
    Et_m_3d = np.fft.ifft(Et_3d, axis=-1)

    return( Er_m_3d, Et_m_3d )


def calculate_laser_fields( fld, propag_direction ):
    """
    Given the fields Er and Et of the laser (in `fld`),
    calculate the fields Ez and B in a self-consistent manner,
    i.e. so that div(E) = 0 and B satisfies d_t B = -curl(E)

    Parameters:
    -----------
    fld: a Fields object
        Contains the fields of the global domain
        (with the correct Er and Et of the laser on the interpolation grid)

    propag_direction: float (either +1. or -1.)
        Whether the laser is propagating in the forward or backward direction.
    """
    # Get the (filtered) E field in spectral space
    fld.interp2spect('E')
    spect = fld.spect
    # Filter the fields in spectral space (with smoother+compensator, otherwise
    # the amplitude of the laser can significantly reduced for low resolution)
    dz = fld.interp[0].dz
    kz_true = 2*np.pi* np.fft.fftfreq( fld.Nz, dz )
    filter_array = (1. - np.sin(0.5*kz_true*dz)**2) * \
                   (1. + np.sin(0.5*kz_true*dz)**2)
    for m in range(fld.Nm):
        spect[m].Ep *= filter_array[:, np.newaxis]
        spect[m].Em *= filter_array[:, np.newaxis]

    # Calculate the Ez field by ensuring that div(E) = 0
    for m in range(fld.Nm):
        inv_kz = np.where( spect[m].kz==0, 0,
                1./np.where( spect[m].kz==0, 1., spect[m].kz ) ) # Avoid nan
        spect[m].Ez[:,:] = 1.j*spect[m].kr*(spect[m].Ep - spect[m].Em)*inv_kz

    # Calculate the B field by ensuring that d_t B = - curl(E)
    # i.e. -i w B = - curl(E), where the sign of w is chosen so that
    # the direction of propagation is given by `propag_direction`
    for m in range(fld.Nm):
        # Calculate w with the right sign
        w = c*np.sqrt( spect[m].kz**2 + spect[m].kr**2 )
        w *= np.sign( spect[m].kz ) * propag_direction
        inv_w = np.where( w==0, 0., 1./np.where( w==0, 1., w ) ) # Avoid nan
        # Calculate the components of the curl in spectral cylindrical
        spect[m].Bp[:,:] = -1.j*inv_w*( spect[m].kz * spect[m].Ep \
                                 - 0.5j*spect[m].kr * spect[m].Ez )
        spect[m].Bm[:,:] = -1.j*inv_w*( -spect[m].kz * spect[m].Em \
                                 - 0.5j*spect[m].kr * spect[m].Ez )
        spect[m].Bz[:,:] = inv_w * spect[m].kr * ( spect[m].Ep + spect[m].Em )

    # Go back to interpolation space
    fld.spect2interp('E')
    fld.spect2interp('B')
