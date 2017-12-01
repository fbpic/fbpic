# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines methods to directly inject the laser in the Simulation box
"""
import numpy as np
from scipy.constants import c
#from fbpic.fields import Fields

def add_laser_direct( sim, laser_profile, fw_propagating, boost ):
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
    TODO
    """
    print("Initializing laser pulse on the mesh...")

    # Get the azimuthally-decomposed laser fields Er and Et
    # on the interpolation grid of each local proc
    Er_m, Et_m = get_laser_Er_Et( sim, laser_profile, boost )
    # Overwrite previous values on the grid
    for m in range(sim.fld.Nm):
        sim.fld.interp[m].Er[:,:] = Er_m[:,:,m]
        sim.fld.interp[m].Et[:,:] = Et_m[:,:,m]

    # Calculate the Ez and B fields on the global grid
    # (For a multi-proc simulation: only performed by the first proc)
    if sim.comm.rank == 0:
        calculate_laser_fields( sim.fld, fw_propagating )

    print("Done.")


def get_laser_Er_Et( sim, laser_profile, boost ):
    """
    TODO
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


def calculate_laser_fields( fld, fw_propagating ):
    """
    TODO
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
    # the direction of propagation is given by the flag `fw_propagating`
    for m in range(fld.Nm):
        # Calculate w with the right sign
        w = c*np.sqrt( spect[m].kz**2 + spect[m].kr**2 )
        w *= np.sign( spect[m].kz )
        if not fw_propagating:
            w *= -1.
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
