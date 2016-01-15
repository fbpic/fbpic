"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of utilities for the initialization of an electron bunch.
"""
import numpy as np
from scipy.constants import m_e, c, e, epsilon_0, mu_0
from fbpic.main import adapt_to_grid
from fbpic.particles import Particles

def add_elec_bunch( sim, gamma0, n_e, p_zmin, p_zmax, p_rmin, p_rmax,
                p_nr=2, p_nz=2, p_nt=4, filter_currents=True, dens_func=None ) :
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
                            dens_func=dens_func, use_cuda=sim.use_cuda,
                            grid_shape=sim.fld.interp[0].Ez.shape )

    # Give them the right velocity
    relat_elec.inv_gamma[:] = 1./gamma0
    relat_elec.uz[:] = np.sqrt( gamma0**2 -1.)
    
    # Add them to the particles of the simulation
    sim.ptcl.append( relat_elec )

    # Get the corresponding space-charge fields
    get_space_charge_fields( sim.fld, [relat_elec], gamma0, filter_currents )
    
def add_elec_bunch_file( sim, filename, Q_tot, z_off=0., filter_currents=True) :
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
                            dens_func=None, use_cuda=sim.use_cuda,
                            grid_shape=sim.fld.interp[0].Ez.shape )

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
                             filter_currents, check_gaminv=False)
    
def get_space_charge_fields( fld, ptcl, gamma, filter_currents=True,
                             check_gaminv=True ) :
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
        get_space_charge_spect( fld.spect[m], gamma )

    # Convert to the interpolation grid
    fld.spect2interp( 'E' )
    fld.spect2interp( 'B' )

    # Move the charge density to rho_prev    
    for m in range(fld.Nm) :
        fld.spect[m].push_rho()

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

