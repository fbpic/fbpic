"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of utilities for laser-plasma acceleration.
"""
import numpy as np
from scipy.constants import m_e, c, e, epsilon_0, mu_0

def add_laser( fld, a0, w0, ctau, z0, lambda0=0.8e-6,
               theta_pol=0., fw_propagating=True, update_spectral=True ) :
    """
    Add a linearly-polarized, Gaussian laser pulse in the Fields object

    The laser profile is added to the interpolation grid, and then
    transformed into spectral space if update_spectral is True
    
    Parameters
    ----------
    fld : a Fields object
       The structure that contains the fields of the simulation
    
    a0 : float (unitless)
       The a0 of the pulse at focus
    
    w0 : float (in meters)
       The waist of the pulse at focus

    ctau : float (in meters)
       The "longitudinal waist" (or pulse length) of the pulse

    z0 : float (in meters)
       The position of the laser centroid
    
    lambda0 : float (in meters), optional
       The central wavelength of the laser
       Default : 0.8 microns (Ti:Sapph laser)

    theta_pol : float (in radians), optional
       The angle of polarization with respect to the x axis
       Default : 0 rad

    fw_propagating : bool, optional
       Wether the laser is propagating in the forward or backward direction
       Default : True (forward propagation)

    update_spectral : bool, optional
       Wether to update the fields in spectral space after modifying the
       fields on the interpolation grid.
       Default : True
    """
    # Extract the wavevector, and the amplitude of the E field at focus
    k0 = 2*np.pi/lambda0
    E0 = a0*m_e*c**2*k0/e
    # Get the polarization component
    # (Due to the Fourier transform along theta, the
    # polarization angle becomes a complex phase in the fields)
    exptheta = np.exp(1.j*theta_pol)

    # Sign for the propagation
    # (prop is 1 if the laser goes in the forward direction,
    # and -1 in the opposite case)
    prop = 2*int(fw_propagating) - 1.
    
    # Get the profile for the Er and Et fields
    z = fld.interp[1].z  # Position of the grid points in z
    r = fld.interp[1].r  # Position of the grid points in r
    long_profile = np.exp( -(z-z0)**2/ctau**2 ) * np.cos( k0*(z-z0) )
    trans_profile = np.exp( -r**2/w0**2 )
    profile = long_profile[:,np.newaxis] * trans_profile[np.newaxis,:]
    
    # Add the Er and Et fields to the mode m=1 (linearly polarized laser)
    # (The factor 0.5 is related to the fact that there is a factor 2
    # in the gathering function, for the particles)
    fld.interp[1].Er +=  0.5  * E0 * exptheta * profile
    fld.interp[1].Et += -0.5j * E0 * exptheta * profile
    fld.interp[1].Br +=  0.5j * prop * E0/c * exptheta * profile
    fld.interp[1].Bt +=  0.5  * prop * E0/c * exptheta * profile

    # Get the profile for the Ez fields (to ensure div(E) = 0)
    # (This uses the approximation lambda0 << ctau for long_profile )
    long_profile = np.exp( -(z-z0)**2/ctau**2 ) * np.sin( k0*(z-z0) ) / k0
    trans_profile = 2*r/w0**2 * np.exp( -r**2/w0**2 )
    profile = long_profile[:,np.newaxis] * trans_profile[np.newaxis,:]

    # Add the Ez fields to the mode m=1 (linearly polarized laser)
    fld.interp[1].Ez +=  0.5  * E0 * exptheta * profile
    fld.interp[1].Bz +=  0.5j * prop * E0/c * exptheta * profile
        
    # Up to now, only the spectral grid was modified.
    # Now convert the fields to spectral space.
    if update_spectral :
        fld.interp2spect('E')
        fld.interp2spect('B')


def get_space_charge_fields( fld, ptcl, gamma ) :
    """
    Calculate the space charge field on the grid

    This assumes that all the particles being passed have
    the same gamma factor.

    Parameters
    ----------
    fld : a Fields object
        Contains the values of the fields

    ptcl : a list of Particles object
        (one element per species)
        The list of the species which are relativistic and
        will produce a space charge field. (Do not pass the
        particles which are at rest.) 

    gamma : float
        The Lorentz factor of the particles
    """
    # Check that all the particles have the right gamma
    for species in ptcl :
        if np.allclose( species.inv_gamma, 1./gamma ) == False :
            raise ValueError("The particles in ptcl do not have "
                            "a Lorentz factor matching gamma. Please check "
                            "that they have been properly initialized.")

    # Project the charge and currents onto the grid
    fld.erase('rho')
    fld.erase('J')
    for species in ptcl :
        species.deposit( fld.interp, 'rho' )
        species.deposit( fld.interp, 'J' )
    fld.divide_by_volume('rho')
    fld.divide_by_volume('J')
    # Convert to the spectral grid
    fld.interp2spect('rho_next')
    fld.interp2spect('J')        

    # Get the space charge field in spectral space
    for m in range(fld.Nm) :
        get_space_charge_spect( fld.spect[m], gamma )

    # Convert to the interpolation grid
    fld.spect2interp( 'E' )
    fld.spect2interp( 'B' )

def get_space_charge_spect( spect, gamma ) :
    """
    Determine the space charge field in spectral space

    It is assumed that the charge density and current
    have been already deposited on the grid, and converted
    to spectral space.

    Parameters
    ----------
    spect : a SpectralGrid object
        Contains the values of the fields in spectral space

    gamma : float
        The Lorentz factor of the particles which produce the
        space charge field
    """
    # Speed of the beam
    beta = np.sqrt(1.-1./gamma**2)
    
    # Get the denominator
    K2 = spect.kr**2 + spect.kz**2 * 1./gamma**2
    inv_K2 = np.where( K2 != 0, 1./K2, 0. )
    
    # Get the potentials
    phi = spect.rho_next[:,:]*inv_K2[:,:]/epsilon_0
    Ap = spect.Jp[:,:]*inv_K2[:,:]*mu_0
    Am = spect.Jm[:,:]*inv_K2[:,:]*mu_0
    Az = spect.Jz[:,:]*inv_K2[:,:]*mu_0

    # Deduce the E field
    spect.Ep[:,:] = 0.5*spect.kr * phi + 1.j*beta*c*spect.kz * Ap
    spect.Em[:,:] = -0.5*spect.kr * phi + 1.j*beta*c*spect.kz * Am
    spect.Ez[:,:] = -1.j*spect.kz * phi + 1.j*beta*c*spect.kz * Az

    # Deduce the B field
    spect.Bp[:,:] = -0.5j*spect.kr * Az + spect.kz * Ap
    spect.Bm[:,:] = -0.5j*spect.kr * Az - spect.kz * Am
    spect.Bz[:,:] = 1.j*spect.kr * Ap + 1.j*spect.kr * Am    
    
    
