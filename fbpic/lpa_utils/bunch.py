# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Soeren Jalas
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of utilities for the initialization of an electron bunch.
"""
import numpy as np
from scipy.constants import m_e, c, e, epsilon_0, mu_0
from fbpic.main import adapt_to_grid
from fbpic.particles import Particles

def add_elec_bunch( sim, gamma0, n_e, p_zmin, p_zmax, p_rmin, p_rmax,
                p_nr=2, p_nz=2, p_nt=4, dens_func=None, boost=None,
                direction='forward', filter_currents=True ) :
    """
    Introduce a simple relativistic electron bunch in the simulation,
    along with its space charge field.

    Uniform particle distribution with weights according to density function.

    Parameters
    ----------
    sim : a Simulation object
        The structure that contains the simulation.

    gamma0 : float
        The Lorentz factor of the electrons

    n_e : float (in particles per m^3)
        Density of the electron bunch

    p_zmin : float (in meters)
        The minimal z position above which the particles are initialized

    p_zmax : float (in meters)
        The maximal z position below which the particles are initialized

    p_rmin : float (in meters)
        The minimal r position above which the particles are initialized

    p_rmax : float (in meters)
        The maximal r position below which the particles are initialized

    p_nz : int
        Number of macroparticles per cell along the z directions

    p_nr : int
        Number of macroparticles per cell along the r directions

    p_nt : int
        Number of macroparticles along the theta direction

    dens_func : callable, optional
        A function of the form :
        `def dens_func( z, r ) ...`
        where `z` and `r` are 1d arrays, and which returns
        a 1d array containing the density *relative to n_e*
        (i.e. a number between 0 and 1) at the given positions.

    boost : a BoostConverter object, optional
        A BoostConverter object defining the Lorentz boost of
        the simulation.

    filter_currents : bool, optional
        Whether to filter the currents in k space (True by default)

    direction : string, optional
        Can be either "forward" or "backward".
        Propagation direction of the beam.
    """

    # Convert parameters to boosted frame
    if boost is not None:
        beta0 = np.sqrt( 1. - 1./gamma0**2 )
        p_zmin, p_zmax = boost.copropag_length(
            [ p_zmin, p_zmax ], beta_object=beta0 )
        n_e, = boost.copropag_density( [n_e], beta_object=beta0 )
        gamma0, = boost.gamma( [gamma0] )

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

    # Electron beam moving in the background direction
    if direction == 'backward':
        relat_elec.uz[:] *= -1.

    # Add them to the particles of the simulation
    sim.ptcl.append( relat_elec )

    # Get the corresponding space-charge fields
    get_space_charge_fields( sim.fld, [relat_elec], gamma0,
                             filter_currents, direction=direction)

def add_elec_bunch_gaussian( sim, sig_r, sig_z, n_emit, gamma0, sig_gamma,
                        Q, N, tf=0., zf=0., boost=None,
                        filter_currents=True, save_beam=None ):
    """
    Introduce a relativistic Gaussian electron bunch in the simulation,
    along with its space charge field.

    The bunch is initialized with a normalized emittance `n_emit`,
    in such a way that it will be focused at time `tf`, at the position `zf`.
    Thus if `tf` is not 0, the bunch will be initially out of focus.
    (This does not take space charge effects into account.)

    Parameters
    ----------
    sim : a Simulation object
        The structure that contains the simulation.

    sig_r : float (in meters)
        The transverse RMS bunch size.

    sig_z : float (in meters)
        The longitudinal RMS bunch size.

    n_emit : float (in meters)
        The normalized emittance of the bunch.

    gamma0 : float
        The Lorentz factor of the electrons.

    sig_gamma : float
        The absolute energy spread of the bunch.

    Q : float (in Coulomb)
        The total charge of the bunch.

    N : int
        The number of particles the bunch should consist of.

    zf: float (in meters), optional
        Position of the focus.

    tf : float (in seconds), optional
        Time at which the bunch reaches focus.

    boost : a BoostConverter object, optional
        A BoostConverter object defining the Lorentz boost of
        the simulation.

    filter_currents : bool, optional
        Whether to filter the currents in k space (True by default)

    save_beam : string, optional
        Saves the generated beam distribution as an .npz file "string".npz
    """
    # Get Gaussian particle distribution in x,y,z
    x = np.random.normal(0., sig_r, N)
    y = np.random.normal(0., sig_r, N)
    z = np.random.normal(zf, sig_z, N) # with offset in z
    # Define sigma of ux and uy based on normalized emittance
    sig_ur = (n_emit/sig_r)
    # Get Gaussian distribution of transverse normalized momenta ux, uy
    ux = np.random.normal(0., sig_ur, N)
    uy = np.random.normal(0., sig_ur, N)
    # Now we imprint an energy spread on the gammas of each particle
    if sig_gamma > 0.:
        gamma = np.random.normal(gamma0, sig_gamma, N)
    else:
        # Or set it to zero
        gamma = np.full(N, gamma0)
        if sig_gamma < 0.:
            print("Warning: Negative energy spread sig_gamma detected."
                  " sig_gamma will be set to zero. \n")
    # Finally we calculate the uz of each particle
    # from the gamma and the transverse momenta ux, uy
    uz = np.sqrt((gamma**2-1) - ux**2 - uy**2)
    # Get inverse gamma
    inv_gamma = 1./gamma
    # Get weight of each particle
    w = -1. * Q / N * np.ones_like(x)

    # Propagate distribution to an out-of-focus position tf.
    # (without taking space charge effects into account)
    if tf != 0.:
        x = x - ux*inv_gamma*c*tf
        y = y - uy*inv_gamma*c*tf
        z = z - uz*inv_gamma*c*tf

    # Save beam distribution to an .npz file
    if save_beam is not None:
        np.savez(save_beam, x=x, y=y, z=z, ux=ux, uy=uy, uz=uz,
            inv_gamma=inv_gamma, w=w)

    # Add the electrons to the simulation
    add_elec_bunch_from_arrays( sim, x, y, z, ux, uy, uz, w,
        boost=boost, filter_currents=filter_currents )


def add_elec_bunch_file( sim, filename, Q_tot, z_off=0., boost=None,
                    filter_currents=True, direction='forward' ):
    """
    Introduce a relativistic electron bunch in the simulation,
    along with its space charge field, loading particles from text file.

    Parameters
    ----------
    sim : a Simulation object
        The structure that contains the simulation.

    filename : string
        the file containing the particle phase space in seven columns
        all float, no header
        x [m]  y [m]  z [m]  ux [unitless]  uy [unitless]  uz [unitless]

    Q_tot : float (in Coulomb)
        total charge in bunch

    z_off: float (in meters)
        Shift the particle positions in z by z_off

    boost : a BoostConverter object, optional
        A BoostConverter object defining the Lorentz boost of
        the simulation.

    filter_currents : bool, optional
        Whether to filter the currents in k space (True by default)

    direction : string, optional
        Can be either "forward" or "backward".
        Propagation direction of the beam.
    """
    # Load particle data to numpy array
    particle_data = np.loadtxt(filename)

    # Extract positions and momenta
    x = particle_data[:,0]
    y = particle_data[:,1]
    z = particle_data[:,2] + z_off
    ux = particle_data[:,3]
    uy = particle_data[:,4]
    uz = particle_data[:,5]
    # Calculate weights (charge of macroparticle)
    # assuming equally weighted particles as used in particle tracking codes
    # multiply by -1 to make them negatively charged
    N_part = len(x)
    w = -1.*Q_tot/N_part * np.ones_like( x )

    # Add the electrons to the simulation
    add_elec_bunch_from_arrays( sim, x, y, z, ux, uy, uz, w,
        boost=boost, filter_currents=filter_currents, direction=direction )


def add_elec_bunch_openPMD( sim, ts_path, z_off=0., species=None, select=None,
                            iteration=None, boost=None, filter_currents=True):
    """
    Introduce a relativistic electron bunch in the simulation,
    along with its space charge field, loading particles from an openPMD
    timeseries.

    Parameters
    ----------
    sim : a Simulation object
        The structure that contains the simulation.

    ts_path : string
        The path to the directory where the openPMD files are.
        For the moment, only HDF5 files are supported. There should be
        one file per iteration, and the name of the files should end
        with the iteration number, followed by '.h5' (e.g. data0005000.h5)

    z_off: float (in meters)
        Shift the particle positions in z by z_off. By default the initialized
        phasespace is centered at z=0.

    species: string
        A string indicating the name of the species
        This is optional if there is only one species

    select: dict, optional
        Either None or a dictionary of rules
        to select the particles, of the form
        'x' : [-4., 10.]   (Particles having x between -4 and 10 microns)
        'ux' : [-0.1, 0.1] (Particles having ux between -0.1 and 0.1 mc)
        'uz' : [5., None]  (Particles with uz above 5 mc)

    iteration: integer (optional)
        The iteration number of the openPMD file from which to extract the
        particles.

    boost : a BoostConverter object, optional
        A BoostConverter object defining the Lorentz boost of
        the simulation.

    filter_currents : bool, optional
        Whether to filter the currents in k space (True by default)
    """
    # Import openPMD viewer
    try:
        from opmd_viewer import OpenPMDTimeSeries
    except ImportError:
        raise ImportError(
        'The package `opmd_viewer` is required to restart from checkpoints.'
        '\nPlease install it from https://github.com/openPMD/openPMD-viewer')
    ts = OpenPMDTimeSeries(ts_path)
    # Extract phasespace and particle weights
    x, y, z, ux, uy, uz, w = ts.get_particle(
                                ['x', 'y', 'z', 'ux', 'uy', 'uz', 'w'],
                                iteration=iteration, species=species,
                                select=select)
    # Convert the positions from microns to meters
    x *= 1.e-6
    y *= 1.e-6
    z *= 1.e-6
    # Shift the center of the phasespace to z_off
    z = z - (np.amax(z) + np.amin(z)) / 2 + z_off

    # Add the electrons to the simulation, and calculate the space charge
    add_elec_bunch_from_arrays( sim, x, y, z, ux, uy, uz, w,
                                boost=boost, filter_currents=filter_currents)


def add_elec_bunch_from_arrays( sim, x, y, z, ux, uy, uz, w,
                    boost=None, filter_currents=True, direction='forward' ):
    """
    Introduce a relativistic electron bunch in the simulation,
    along with its space charge field, loading particles from numpy arrays.

    Parameters
    ----------
    sim : a Simulation object
        The structure that contains the simulation.

    x, y, z: 1d arrays of length (N_macroparticles,)
        The positions of the particles in x, y, z in meters

    ux, uy, uz: 1d arrays of length (N_macroparticles,)
        The dimensionless momenta of the particles in each direction

    w: 1d array of length (N_macroparticles,)
        The weight of the particles, i.e. the number of physical particles
        that each macroparticle corresponds to.

    boost : a BoostConverter object, optional
        A BoostConverter object defining the Lorentz boost of
        the simulation.

    filter_currents : bool, optional
        Whether to filter the currents in k space (True by default)

    direction : string, optional
        Can be either "forward" or "backward".
        Propagation direction of the beam.
    """
    # Extract the number of macroparticles
    N_part = len(x)

    # Create dummy electrons with the correct number of particles
    relat_elec = Particles( q=-e, m=m_e, n=1.,
                        Npz=N_part, zmin=1., zmax=2.,
                        Npr=1, rmin=0., rmax=1.,
                        Nptheta=1, dt=sim.dt,
                        continuous_injection=False,
                        dens_func=None, use_cuda=sim.use_cuda,
                        grid_shape=sim.fld.interp[0].Ez.shape )

    # Replace dummy particle parameters with the provided arrays
    relat_elec.x[:] = x[:]
    relat_elec.y[:] = y[:]
    relat_elec.z[:] = z[:]
    relat_elec.ux[:] = ux[:]
    relat_elec.uy[:] = uy[:]
    relat_elec.uz[:] = uz[:]
    relat_elec.inv_gamma[:] = 1./np.sqrt( \
        1. + relat_elec.ux**2 + relat_elec.uy**2 + relat_elec.uz**2 )
    # Convert number of particle weights to particle charge
    relat_elec.w[:] = -e * w[:]

    # Transform particle distribution in
    # the Lorentz boosted frame, if gamma_boost != 1.
    if boost is not None:
        boost.boost_particles( relat_elec )

    # Add them to the particles of the simulation
    sim.ptcl.append( relat_elec )

    # Get the corresponding space-charge fields
    # include a larger tolerance of the deviation of inv_gamma from 1./gamma0
    # to allow for energy spread
    gamma0 = 1. / np.mean(relat_elec.inv_gamma)
    get_space_charge_fields( sim.fld, [relat_elec], gamma0,
      filter_currents=filter_currents, check_gaminv=False, direction=direction)

def get_space_charge_fields( fld, ptcl, gamma, filter_currents=True,
                             check_gaminv=True, direction='forward' ) :
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


def get_space_charge_spect( spect, gamma, direction='forward' ) :
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
    spect.Ep[:,:] += 0.5*spect.kr * phi + 1.j*beta*c*spect.kz * Ap
    spect.Em[:,:] += -0.5*spect.kr * phi + 1.j*beta*c*spect.kz * Am
    spect.Ez[:,:] += -1.j*spect.kz * phi + 1.j*beta*c*spect.kz * Az

    # Deduce the B field
    spect.Bp[:,:] += -0.5j*spect.kr * Az + spect.kz * Ap
    spect.Bm[:,:] += -0.5j*spect.kr * Az - spect.kz * Am
    spect.Bz[:,:] += 1.j*spect.kr * Ap + 1.j*spect.kr * Am
