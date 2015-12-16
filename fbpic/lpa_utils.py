"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of utilities for laser-plasma acceleration.
"""
import numpy as np
from scipy.constants import m_e, c, e, epsilon_0, mu_0
from main import adapt_to_grid
from particles import Particles

def add_laser( fld, a0, w0, ctau, z0, zf=None, lambda0=0.8e-6,
               theta_pol=0., fw_propagating=True,
               update_spectral=True ) :
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
       The position of the laser centroid relative to z=0.

    zf : float (in meters), optional
       The position of the laser focus relative to z=0.
       If not provided, then the laser focus is at z0
    
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
    # Set a number of parameters for the laser
    k0 = 2*np.pi/lambda0
    E0 = a0*m_e*c**2*k0/e      # Amplitude at focus
    zr = np.pi*w0**2/lambda0   # Rayleigh length
    # Get the polarization component
    # (Due to the Fourier transform along theta, the
    # polarization angle becomes a complex phase in the fields)
    exptheta = np.exp(1.j*theta_pol)
    # Sign for the propagation
    # (prop is 1 for forward propagation, and -1 otherwise)
    prop = 2*int(fw_propagating) - 1.
    # Set default focusing position
    if zf is None : zf = z0

    # Get 2D mesh for z and r
    r, z = np.meshgrid( fld.interp[1].r, fld.interp[1].z )
    
    # Define functions for laser waist and curvature
    w = lambda z: w0*np.sqrt(1+(z/zr)**2)
    R = lambda z: z*(1+(zr/z)**2)
    waist =  w(z-zf)
    propag_phase = np.arctan((z-zf)/zr) - k0*r**2/(2*R(z-zf)) - k0*(z-zf)
    
    # Longitudinal and transverse profile
    long_profile = np.exp( -(z-z0)**2/ctau**2 )
    trans_profile = w0/waist * np.exp( -(r/waist)**2 )
    # Curvature and laser oscillations (cos part)
    curvature_oscillations_cos = np.cos( propag_phase )
    # Combine profiles
    profile = long_profile * trans_profile * curvature_oscillations_cos
    
    # Add the Er and Et fields to the mode m=1 (linearly polarized laser)
    # (The factor 0.5 is related to the fact that there is a factor 2
    # in the gathering function, for the particles)
    fld.interp[1].Er +=  0.5  * E0 * exptheta * profile
    fld.interp[1].Et += -0.5j * E0 * exptheta * profile
    fld.interp[1].Br +=  0.5j * prop * E0/c * exptheta * profile
    fld.interp[1].Bt +=  0.5  * prop * E0/c * exptheta * profile

    # Get the profile for the Ez fields (to ensure div(E) = 0)
    # (This uses the approximation lambda0 << ctau for long_profile )
    # Transverse profile
    trans_profile = - r/zr * ( w0/waist )**3 * np.exp( -(r/waist)**2 )
    # Curvature and laser oscillations (sin part)
    curvature_oscillations_sin = np.sin( propag_phase )
    # Combine profiles
    profile = long_profile * trans_profile * \
    ( curvature_oscillations_sin + (z-zf)/zr*curvature_oscillations_cos )

    # Add the Ez fields to the mode m=1 (linearly polarized laser)
    fld.interp[1].Ez +=  0.5  * E0 * exptheta * profile
    fld.interp[1].Bz +=  0.5j * prop * E0/c * exptheta * profile
        
    # Up to now, only the spectral grid was modified.
    # Now convert the fields to spectral space.
    if update_spectral :
        fld.interp2spect('E')
        fld.interp2spect('B')


def add_elec_bunch( sim, gamma0, n_e, p_zmin, p_zmax, p_rmin, p_rmax,
                p_nr=2, p_nz=2, p_nt=4, filter_currents=True, dens_func=None,
                direction = 'forward' ) :
    """
    Introduce a relativistic electron bunch in the simulation,
    along with its space charge field.

    sim : a Simulation object

    gamma0 : float
        The Lorentz factor of the electrons

    n_e : float (in particles per m^3)
        Density of the electron bunch
    
    p_zmin, p_zmax : floats
        z positions between which the particles are initialized

    p_rmin, p_rmax : floats
        r positions between which the fields are initialized

    p_nz, p_nr : ints
        Number of macroparticles per cell along the z and r directions

    p_nt : int
        Number of macroparticles along the theta direction

    dens_func : callable, optional
        A function of the form :
        def dens_func( z, r ) ...
        where z and r are 1d arrays, and which returns
        a 1d array containing the density *relative to n*
        (i.e. a number between 0 and 1) at the given positions

    filter_currents : bool, optional
        Whether to filter the currents (in k space by default)

    direction : string, optional
        Can be either "forward" or "backward".
        Propagation direction of the beam.
    """
    # Modify the input parameters p_zmin, p_zmax, r_zmin, r_zmax, so that
    # they fall exactly on the grid, and infer the number of particles
    p_zmin, p_zmax, Npz = adapt_to_grid( sim.fld.interp[0].z,
                                p_zmin, p_zmax, p_nz )
    p_rmin, p_rmax, Npr = adapt_to_grid( sim.fld.interp[0].r,
                                p_rmin, p_rmax, p_nr )

    # Create the electrons
    relat_elec = Particles( q=-e, m=m_e, n=n_e,
                            Npz=Npz, zmin=p_zmin, zmax=p_zmax,
                            Npr=Npr, rmin=p_rmin, rmax=p_rmax,
                            Nptheta=p_nt, dt=sim.dt,
                            continuous_injection=False,
                            dens_func=dens_func, 
                            use_cuda=sim.use_cuda,
                            v_galilean = sim.v_galilean )

    # Give them the right velocity
    relat_elec.inv_gamma[:] = 1./gamma0
    relat_elec.uz[:] = np.sqrt( gamma0**2 -1.)
    
    # Electron beam moving in the background direction
    if direction == 'backward':
        relat_elec.uz[:] *= -1.

    # Add them to the particles of the simulation
    sim.ptcl.append( relat_elec )

    # Get the corresponding space-charge fields
    get_space_charge_fields( sim.fld, [relat_elec], gamma0, 
                             filter_currents, direction = direction)
    
def add_elec_bunch_file( sim, filename, Q_tot, z_off=0., 
                         filter_currents=True, direction = 'forward' ) :
    """
    Introduce a relativistic electron bunch in the simulation,
    along with its space charge field,
    load particles from text file.

    sim : a Simulation object

    filename : str
        the file containing the particle phase space in seven columns
        (like for Warp), all float, no header
        x [m]  y [m]  z [m]  vx [m/s]  vy [m/s]  vz [m/s]  1./gamma [1]

    Q_tot : float (in Coulomb)
        total charge in bunch

    z_center: float (m)
        shift phase space in z by z_off
 
    filter_currents : bool, optional
        Whether to filter the currents (in k space by default)

    direction : string, optional
        Can be either "forward" or "backward".
        Propagation direction of the beam.
    """

    # Load phase space to numpy array
    phsp = np.loadtxt(filename)
    # Extract number of particles and average gamma
    N_part = np.shape(phsp)[0]
    gamma0 = 1./np.mean(phsp[:,6])

    # Create dummy electrons with the correct number of particles
    relat_elec = Particles( q=-e, m=m_e, n=1.,
                            Npz=N_part, zmin=1., zmax=2.,
                            Npr=1, rmin=0., rmax=1.,
                            Nptheta=1, dt=sim.dt,
                            continuous_injection=False,
                            dens_func=None,
                            use_cuda=sim.use_cuda,
                            v_galilean = sim.v_galilean )

    # Replace dummy particle parameters with phase space from text file
    relat_elec.x[:] = phsp[:,0]
    relat_elec.y[:] = phsp[:,1]
    relat_elec.z[:] = phsp[:,2] + z_off
    # For momenta: convert velocity [m/s] to normalized momentum u = p/m_e/c [1]
    relat_elec.ux[:] = phsp[:,3]/phsp[:,6]/c
    relat_elec.uy[:] = phsp[:,4]/phsp[:,6]/c
    relat_elec.uz[:] = phsp[:,5]/phsp[:,6]/c
    relat_elec.inv_gamma[:] = phsp[:,6]
    # Calculate weights (charge of macroparticle)
    # assuming equally weighted particles as used in particle tracking codes
    # multiply by -1 to make them negatively charged
    relat_elec.w[:] = -1.*Q_tot/N_part
    
    # Add them to the particles of the simulation
    sim.ptcl.append( relat_elec )

    # Get the corresponding space-charge fields
    # include a larger tolerance of the deviation of inv_gamma from 1./gamma0
    # to allow for energy spread
    get_space_charge_fields( sim.fld, [relat_elec], gamma0,
                             filter_currents, check_gaminv=False,
                             direction = direction)
    
def get_space_charge_fields( fld, ptcl, gamma, filter_currents=True,
                             check_gaminv=True, direction = 'forward' ) :
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

    filter_currents : bool, optional
       Whether to filter the currents (in k space by default)

    check_gaminv : bool, optional
        Explicitly check that all particles have the same
        gamma factor (assumed by the model)

    direction : string, optional
        Can be either "forward" or "backward".
        Propagation direction of the beam.
    """
    # Check that all the particles have the right gamma
    if check_gaminv:
        for species in ptcl :
            if np.allclose( species.inv_gamma, 1./gamma ) == False :    
                raise ValueError("The particles in ptcl do not have "
                            "a Lorentz factor matching gamma. Please check "
                            "that they have been properly initialized.")

    # Project the charge and currents onto the grid
    fld.erase('rho')
    fld.erase('J')
    for species in ptcl :
        species.deposit( fld, 'rho' )
        species.deposit( fld, 'J' )
    fld.divide_by_volume('rho')
    fld.divide_by_volume('J')
    # Convert to the spectral grid
    fld.interp2spect('rho_next')
    fld.interp2spect('J')
    # Filter the currents
    if filter_currents :
        fld.filter_spect('rho_next')      
        fld.filter_spect('J')      

    # Get the space charge field in spectral space
    for m in range(fld.Nm) :
        get_space_charge_spect( fld.spect[m], gamma, direction )

    # Convert to the interpolation grid
    fld.spect2interp( 'E' )
    fld.spect2interp( 'B' )

    # Move the charge density to rho_prev    
    for m in range(fld.Nm) :
        fld.spect[m].push_rho()

def get_space_charge_spect( spect, gamma, direction = 'forward' ) :
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

    direction : string, optional
        Can be either "forward" or "backward".
        Propagation direction of the beam.
    """
    # Speed of the beam
    beta = np.sqrt(1.-1./gamma**2)

    # Propagation direction of the beam
    if direction == 'backward':
        beta *= -1.
        
    # Get the denominator
    K2 = spect.kr**2 + spect.kz**2 * 1./gamma**2
    K2_corrected = np.where( K2 != 0, K2, 1. )
    inv_K2 = np.where( K2 !=0, 1./K2_corrected, 0. )
    
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
    
    
