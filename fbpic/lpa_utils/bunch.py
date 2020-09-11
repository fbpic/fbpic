# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Soeren Jalas
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of utilities for the initialization of an electron bunch.
"""
import numpy as np
from scipy.constants import m_e, c, e, epsilon_0, mu_0
from fbpic.fields import Fields
from fbpic.particles.elementary_process.cuda_numba_utils import \
    reallocate_and_copy_old
from fbpic.particles.injection import BallisticBeforePlane
from fbpic.utils.cuda import GpuMemoryManager
import warnings


def add_particle_bunch(sim, q, m, gamma0, n, p_zmin, p_zmax, p_rmin, p_rmax,
                       p_nr=2, p_nz=2, p_nt=4, dens_func=None, boost=None,
                       direction='forward', z_injection_plane=None,
                       initialize_self_field=True):
    """
    Introduce a simple relativistic particle bunch in the simulation,
    along with its space charge field.

    Uniform particle distribution with weights according to density function.

    Parameters
    ----------
    sim : a Simulation object
        The structure that contains the simulation.

    q : float (in Coulomb)
        Charge of the particle species

    m : float (in kg)
        Mass of the particle species

    gamma0 : float
        The Lorentz factor of the particles

    n : float (in particles per m^3)
        Density of the bunch

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
        a 1d array containing the density *relative to n*
        (i.e. a number between 0 and 1) at the given positions.

    boost : a BoostConverter object, optional
        A BoostConverter object defining the Lorentz boost of
        the simulation.

    direction : string, optional
        Can be either "forward" or "backward".
        Propagation direction of the beam.

    z_injection_plane: float (in meters) or None
        When `z_injection_plane` is not None, then particles have a ballistic
        motion for z<z_injection_plane. This is sometimes useful in
        boosted-frame simulations.
        `z_injection_plane` is always given in the lab frame.

    initialize_self_field: bool, optional
       Whether to calculate the initial space charge fields of the bunch
       and add these fields to the fields on the grid (Default: True)
    """
    # Calculate the electron momentum
    uz_m = ( gamma0**2 - 1. )**0.5
    if direction == 'backward':
        uz_m *= -1.
    # Create the electron species
    ptcl_bunch = sim.add_new_species( q=q, m=m, n=n,
                            p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
                            p_zmin=p_zmin, p_zmax=p_zmax,
                            p_rmin=p_rmin, p_rmax=p_rmax,
                            continuous_injection=False,
                            dens_func=dens_func, uz_m=uz_m )

    # Initialize the injection plane for the particles
    if z_injection_plane is not None:
        assert ptcl_bunch.injector is None #Don't overwrite a previous injector
        ptcl_bunch.injector = BallisticBeforePlane( z_injection_plane, boost )

    # Get the corresponding space-charge fields
    if initialize_self_field:
        get_space_charge_fields( sim, ptcl_bunch, direction=direction )
    return ptcl_bunch


def add_particle_bunch_gaussian(sim, q, m, sig_r, sig_z, n_emit, gamma0,
                                sig_gamma, n_physical_particles,
                                n_macroparticles, tf=0., zf=0., boost=None,
                                save_beam=None, z_injection_plane=None,
                                initialize_self_field=True):
    """
    Introduce a relativistic Gaussian particle bunch in the simulation,
    along with its space charge field.

    The bunch is initialized with a normalized emittance `n_emit`,
    in such a way that it will be focused at time `tf`, at the position `zf`.
    Thus if `tf` is not 0, the bunch will be initially out of focus.
    (This does not take space charge effects into account.)

    Parameters
    ----------
    sim : a Simulation object
        The structure that contains the simulation.

    q : float (in Coulomb)
        Charge of the particle species

    m : float (in kg)
        Mass of the particle species

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

    n_physical_particles : float
        The number of physical particles (e.g. electrons) the bunch should
        consist of.

    n_macroparticles : int
        The number of macroparticles the bunch should consist of.

    zf: float (in meters), optional
        Position of the focus.

    tf : float (in seconds), optional
        Time at which the bunch reaches focus.

    boost : a BoostConverter object, optional
        A BoostConverter object defining the Lorentz boost of
        the simulation.

    save_beam : string, optional
        Saves the generated beam distribution as an .npz file "string".npz

    z_injection_plane: float (in meters) or None
        When `z_injection_plane` is not None, then particles have a ballistic
        motion for z<z_injection_plane. This is sometimes useful in
        boosted-frame simulations.
        `z_injection_plane` is always given in the lab frame.

    initialize_self_field: bool, optional
       Whether to calculate the initial space charge fields of the bunch
       and add these fields to the fields on the grid (Default: True)
    """
    # Generate Gaussian gamma distribution of the beam
    if sig_gamma > 0.:
        gamma = np.random.normal(gamma0, sig_gamma, n_macroparticles)
    else:
        # Zero energy spread beam
        gamma = np.full(n_macroparticles, gamma0)
        if sig_gamma < 0.:
            warnings.warn(
                "Negative energy spread sig_gamma detected."
                " sig_gamma will be set to zero. \n")
    # Get inverse gamma
    inv_gamma = 1. / gamma
    # Get Gaussian particle distribution in x,y,z
    x = sig_r * np.random.normal(0., 1., n_macroparticles)
    y = sig_r * np.random.normal(0., 1., n_macroparticles)
    z = zf + sig_z * np.random.normal(0., 1., n_macroparticles)

    # Define sigma of ux and uy based on normalized emittance
    sig_ur = (n_emit / sig_r)
    # Get Gaussian distribution of transverse normalized momenta ux, uy
    ux = sig_ur * np.random.normal(0., 1., n_macroparticles)
    uy = sig_ur * np.random.normal(0., 1., n_macroparticles)

    # Finally we calculate the uz of each particle
    # from the gamma and the transverse momenta ux, uy
    uz_sqr = (gamma ** 2 - 1) - ux ** 2 - uy ** 2

    # Check for unphysical particles with uz**2 < 0
    mask = uz_sqr >= 0
    N_new = np.count_nonzero(mask)
    if N_new < n_macroparticles:
        warnings.warn(
              "Particles with uz**2<0 detected."
              " %d Particles will be removed from the beam. \n"
              "This will truncate the distribution of the beam"
              " at gamma ~= 1. \n"
              "However, the charge will be kept constant. \n"%(n_macroparticles
                                                               - N_new))
        # Remove unphysical particles with uz**2 < 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        ux = ux[mask]
        uy = uy[mask]
        inv_gamma = inv_gamma[mask]
        uz_sqr = uz_sqr[mask]
    # Calculate longitudinal momentum of the bunch
    uz = np.sqrt(uz_sqr)
    # Get weight of each particle

    w = n_physical_particles / N_new * np.ones_like(x)
    # Propagate distribution to an out-of-focus position tf.
    # (without taking space charge effects into account)
    if tf != 0.:
        x = x - ux * inv_gamma * c * tf
        y = y - uy * inv_gamma * c * tf
        z = z - uz * inv_gamma * c * tf

    # Save beam distribution to an .npz file
    if save_beam is not None:
        np.savez(save_beam, x=x, y=y, z=z, ux=ux, uy=uy, uz=uz,
            inv_gamma=inv_gamma, w=w)

    # Add the electrons to the simulation
    ptcl_bunch = add_particle_bunch_from_arrays(sim, q, m, x, y, z, ux, uy, uz,
                    w, boost=boost, z_injection_plane=z_injection_plane,
                    initialize_self_field=initialize_self_field)
    return ptcl_bunch


def add_particle_bunch_file(sim, q, m, filename, n_physical_particles,
                            z_off=0., boost=None, direction='forward',
                            z_injection_plane=None,
                            initialize_self_field=True):
    """
    Introduce a relativistic particle bunch in the simulation,
    along with its space charge field, loading particles from text file.

    Parameters
    ----------
    sim : a Simulation object
        The structure that contains the simulation.

    q : float (in Coulomb)
        Charge of the particle species

    m : float (in kg)
        Mass of the particle species

    filename : string
        the file containing the particle phase space in seven columns
        all float, no header
        x [m]  y [m]  z [m]  ux [unitless]  uy [unitless]  uz [unitless]

    n_physical_particles : float
        The number of physical particles (e.g. electrons) the bunch should
        consist of.

    z_off: float (in meters)
        Shift the particle positions in z by z_off

    boost : a BoostConverter object, optional
        A BoostConverter object defining the Lorentz boost of
        the simulation.

    direction : string, optional
        Can be either "forward" or "backward".
        Propagation direction of the beam.

    z_injection_plane: float (in meters) or None
        When `z_injection_plane` is not None, then particles have a ballistic
        motion for z<z_injection_plane. This is sometimes useful in
        boosted-frame simulations.
        `z_injection_plane` is always given in the lab frame.

    initialize_self_field: bool, optional
       Whether to calculate the initial space charge fields of the bunch
       and add these fields to the fields on the grid (Default: True)
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
    n_macroparticles = len(x)
    w = n_physical_particles / n_macroparticles * np.ones_like(x)

    # Add the electrons to the simulation
    ptcl_bunch = add_particle_bunch_from_arrays(sim, q, m, x, y, z, ux, uy, uz,
                           w, boost=boost, direction=direction,
                           z_injection_plane=z_injection_plane,
                           initialize_self_field=initialize_self_field)
    return ptcl_bunch


def add_particle_bunch_openPMD( sim, q, m, ts_path, z_off=0., species=None,
                                select=None, iteration=None, boost=None,
                                z_injection_plane=None,
                                initialize_self_field=True ):
    """
    Introduce a relativistic particle bunch in the simulation,
    along with its space charge field, loading particles from an openPMD
    timeseries.

    Parameters
    ----------
    sim : a Simulation object
        The structure that contains the simulation.

    q : float (in Coulomb)
        Charge of the particle species

    m : float (in kg)
        Mass of the particle species

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

    z_injection_plane: float (in meters) or None
        When `z_injection_plane` is not None, then particles have a ballistic
        motion for z<z_injection_plane. This is sometimes useful in
        boosted-frame simulations.
        `z_injection_plane` is always given in the lab frame.

    initialize_self_field: bool, optional
       Whether to calculate the initial space charge fields of the bunch
       and add these fields to the fields on the grid (Default: True)
    """
    # Try to import openPMD-viewer, version 1
    try:
        from openpmd_viewer import OpenPMDTimeSeries
        openpmd_viewer_version = 1
    except ImportError:
        # If not available, try to import openPMD-viewer, version 0
        try:
            from opmd_viewer import OpenPMDTimeSeries
            openpmd_viewer_version = 0
        except ImportError:
            openpmd_viewer_version = None
    # Otherwise, raise an error
    if openpmd_viewer_version is None:
        raise ImportError(
        'The package openPMD-viewer is required to load a particle bunch from on openPMD file.'
        '\nPlease install it from https://github.com/openPMD/openPMD-viewer')
    ts = OpenPMDTimeSeries(ts_path)
    # Extract phasespace and particle weights
    x, y, z, ux, uy, uz, w = ts.get_particle(
                                ['x', 'y', 'z', 'ux', 'uy', 'uz', 'w'],
                                iteration=iteration, species=species,
                                select=select)
    if openpmd_viewer_version == 0:
        # Convert the positions from microns to meters
        x *= 1.e-6
        y *= 1.e-6
        z *= 1.e-6
    # Shift the center of the phasespace to z_off
    z = z - np.average(z, weights=w) + z_off

    # Add the electrons to the simulation, and calculate the space charge
    ptcl_bunch = add_particle_bunch_from_arrays(sim, q, m, x, y, z, ux, uy, uz,
                            w, boost=boost,
                            z_injection_plane=z_injection_plane,
                            initialize_self_field=initialize_self_field)
    return ptcl_bunch


def add_particle_bunch_from_arrays(sim, q, m, x, y, z, ux, uy, uz, w,
                                   boost=None, direction='forward',
                                   z_injection_plane=None,
                                   initialize_self_field=True):
    """
    Introduce a relativistic particle bunch in the simulation,
    along with its space charge field, loading particles from numpy arrays.

    Parameters
    ----------
    sim : a Simulation object
        The structure that contains the simulation.

    q : float
        Charge of the particle species

    m : float
        Mass of the particle species

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

    direction : string, optional
        Can be either "forward" or "backward".
        Propagation direction of the beam.

    z_injection_plane: float (in meters) or None
        When `z_injection_plane` is not None, then particles have a ballistic
        motion for z<z_injection_plane. This is sometimes useful in
        boosted-frame simulations.
        `z_injection_plane` is always given in the lab frame.

    initialize_self_field: bool, optional
       Whether to calculate the initial space charge fields of the bunch
       and add these fields to the fields on the grid (Default: True)
    """
    inv_gamma = 1./np.sqrt( 1. + ux**2 + uy**2 + uz**2 )
    # Convert the particles to the boosted-frame
    if boost is not None:
        x, y, z, ux, uy, uz, inv_gamma = boost.boost_particle_arrays(
                                        x, y, z, ux, uy, uz, inv_gamma )

    # Select the particles that are in the local subdomain
    zmin, zmax = sim.comm.get_zmin_zmax(
        local=True, with_damp=False, with_guard=False, rank=sim.comm.rank )
    selected = (z >= zmin) & (z < zmax)
    x = x[selected]
    y = y[selected]
    z = z[selected]
    ux = ux[selected]
    uy = uy[selected]
    uz = uz[selected]
    w = w[selected]
    inv_gamma = inv_gamma[selected]

    # Create electron species with no macroparticles
    ptcl_bunch = sim.add_new_species( q=q, m=m )

    # Reallocate empty arrays with the right number of electrons
    Ntot = len(x)
    reallocate_and_copy_old( ptcl_bunch, ptcl_bunch.use_cuda, 0, Ntot )

    # Fill the empty particle arrays with the right values
    ptcl_bunch.x[:] = x[:]
    ptcl_bunch.y[:] = y[:]
    ptcl_bunch.z[:] = z[:]
    ptcl_bunch.ux[:] = ux[:]
    ptcl_bunch.uy[:] = uy[:]
    ptcl_bunch.uz[:] = uz[:]
    ptcl_bunch.inv_gamma[:] = inv_gamma[:]
    ptcl_bunch.w[:] = w[:]

    # Initialize the injection plane for the particles
    if z_injection_plane is not None:
        assert ptcl_bunch.injector is None #Don't overwrite a previous injector
        ptcl_bunch.injector = BallisticBeforePlane( z_injection_plane, boost )

    # Get the corresponding space-charge fields
    if initialize_self_field:
        get_space_charge_fields(sim, ptcl_bunch, direction=direction)
    return ptcl_bunch


def add_elec_bunch( sim, gamma0, n_e, p_zmin, p_zmax, p_rmin, p_rmax,
                p_nr=2, p_nz=2, p_nt=4, dens_func=None, boost=None,
                direction='forward', z_injection_plane=None ) :
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

    direction : string, optional
        Can be either "forward" or "backward".
        Propagation direction of the beam.

    z_injection_plane: float (in meters) or None
        When `z_injection_plane` is not None, then particles have a ballistic
        motion for z<z_injection_plane. This is sometimes useful in
        boosted-frame simulations.
        `z_injection_plane` is always given in the lab frame.
    """
    elec_bunch = add_particle_bunch(sim, -e, m_e, gamma0, n_e, p_zmin, p_zmax,
                       p_rmin, p_rmax, p_nr=p_nr, p_nz=p_nz, p_nt=p_nz,
                       dens_func=dens_func, boost=boost, direction=direction,
                       z_injection_plane=z_injection_plane)
    return elec_bunch


def add_elec_bunch_gaussian( sim, sig_r, sig_z, n_emit, gamma0,
                        sig_gamma, Q, N, tf=0., zf=0., boost=None,
                        save_beam=None, z_injection_plane=None ):
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
        The total charge of the bunch (in absolute value)
        (if a negative number is given, its absolute value will
        automatically be taken)

    N : int
        The number of particles the bunch should consist of.

    zf: float (in meters), optional
        Position of the focus.

    tf : float (in seconds), optional
        Time at which the bunch reaches focus.

    boost : a BoostConverter object, optional
        A BoostConverter object defining the Lorentz boost of
        the simulation.

    save_beam : string, optional
        Saves the generated beam distribution as an .npz file "string".npz

    z_injection_plane: float (in meters) or None
        When `z_injection_plane` is not None, then particles have a ballistic
        motion for z<z_injection_plane. This is sometimes useful in
        boosted-frame simulations.
        `z_injection_plane` is always given in the lab frame.
    """
    # Generate Gaussian gamma distribution of the beam
    n_physical_particles = Q/e
    elec_bunch = add_particle_bunch_gaussian(sim, -e, m_e, sig_r, sig_z,
                                n_emit, gamma0, sig_gamma,
                                n_physical_particles, N, tf=tf,
                                zf=zf, boost=boost, save_beam=save_beam,
                                z_injection_plane=z_injection_plane)
    return elec_bunch


def add_elec_bunch_file( sim, filename, Q_tot, z_off=0., boost=None,
                        direction='forward', z_injection_plane=None ):
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
        The total charge of the bunch (in absolute value)
        (if a negative number is given, its absolute value will
        automatically be taken)

    z_off: float (in meters)
        Shift the particle positions in z by z_off

    boost : a BoostConverter object, optional
        A BoostConverter object defining the Lorentz boost of
        the simulation.

    direction : string, optional
        Can be either "forward" or "backward".
        Propagation direction of the beam.

    z_injection_plane: float (in meters) or None
        When `z_injection_plane` is not None, then particles have a ballistic
        motion for z<z_injection_plane. This is sometimes useful in
        boosted-frame simulations.
        `z_injection_plane` is always given in the lab frame.
    """
    n_physical_particles = Q_tot / e
    elec_bunch = add_particle_bunch_file(sim, -e, m_e, filename,
                            n_physical_particles, z_off=z_off,
                            boost=boost, direction=direction,
                            z_injection_plane=z_injection_plane)
    return elec_bunch


def add_elec_bunch_openPMD( sim, ts_path, z_off=0., species=None, select=None,
                        iteration=None, boost=None, z_injection_plane=None ):
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

    z_injection_plane: float (in meters) or None
        When `z_injection_plane` is not None, then particles have a ballistic
        motion for z<z_injection_plane. This is sometimes useful in
        boosted-frame simulations.
        `z_injection_plane` is always given in the lab frame.
    """
    elec_bunch = add_particle_bunch_openPMD(sim, -e, m_e, ts_path, z_off=z_off,
                               species=species, select=select,
                               iteration=iteration, boost=boost,
                               z_injection_plane=z_injection_plane)
    return elec_bunch


def add_elec_bunch_from_arrays( sim, x, y, z, ux, uy, uz, w,
                    boost=None, direction='forward', z_injection_plane=None ):
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

    direction : string, optional
        Can be either "forward" or "backward".
        Propagation direction of the beam.

    z_injection_plane: float (in meters) or None
        When `z_injection_plane` is not None, then particles have a ballistic
        motion for z<z_injection_plane. This is sometimes useful in
        boosted-frame simulations.
        `z_injection_plane` is always given in the lab frame.
    """
    elec_bunch = add_particle_bunch_from_arrays(sim, -e, m_e, x, y, z,
                                   ux, uy, uz, w, boost=boost,
                                   direction=direction,
                                   z_injection_plane=z_injection_plane)
    return elec_bunch


def get_space_charge_fields( sim, ptcl, direction='forward' ):
    """
    Add the space charge field from `ptcl` the interpolation grid

    This assumes that all the particles being passed have the same gamma.

    Parameters
    ----------
    sim : a Simulation object
        Contains the values of the fields, and the MPI communicator

    ptcl : a Particles object
        The list of the species which are relativistic and
        will produce a space charge field. (Do not pass the
        particles which are at rest.)

    direction : string, optional
        Can be either "forward" or "backward".
        Propagation direction of the beam.
    """
    if sim.comm.rank == 0:
        print("Calculating initial space charge field...")

    # Calculate the mean gamma by computing weighted sum on each subdomain
    w_sum_local = ptcl.w.sum()
    w_gamma_sum_local = (ptcl.w*1./ptcl.inv_gamma).sum()
    if sim.comm.mpi_comm is None:
        w_sum = w_sum_local
        w_gamma_sum = w_gamma_sum_local
    else:
        w_sum = sim.comm.mpi_comm.allreduce(w_sum_local)
        w_gamma_sum = sim.comm.mpi_comm.allreduce(w_gamma_sum_local)
    # Check that the number of particles is not 0
    if w_sum == 0:
        warnings.warn(
            "Tried to calculate space charge, but found 0 macroparticles in \n"
            "the corresponding species. Skipping space charge calculation...\n")
        return
    else:
        gamma = w_gamma_sum/w_sum

    # Project the charge and currents onto the local subdomain
    # (Move data to GPU if needed, for this step)
    with GpuMemoryManager(sim):
        sim.deposit( 'rho', exchange=True, species_list=[ptcl],
                        update_spectral=False )
        sim.deposit( 'J', exchange=True, species_list=[ptcl],
                        update_spectral=False )

    # Create a global field object across all subdomains, and copy the sources
    # (Space-charge calculation is a global operation)
    # Note: in the single-proc case, this is also useful in order not to
    # erase the pre-existing E and B field in sim.fld
    global_Nz, _ = sim.comm.get_Nz_and_iz(
                    local=False, with_damp=True, with_guard=False )
    global_zmin, global_zmax = sim.comm.get_zmin_zmax(
                    local=False, with_damp=True, with_guard=False )
    global_fld = Fields( global_Nz, global_zmax,
            sim.fld.Nr, sim.fld.rmax, sim.fld.Nm, sim.fld.dt,
            n_order=sim.fld.n_order, smoother=sim.fld.smoother,
            zmin=global_zmin, use_cuda=False)
    # Gather the sources on the interpolation grid of global_fld
    for m in range(sim.fld.Nm):
        for field in ['Jr', 'Jt', 'Jz', 'rho']:
            local_array = getattr( sim.fld.interp[m], field )
            gathered_array = sim.comm.gather_grid_array(
                                            local_array, with_damp=True )
            setattr( global_fld.interp[m], field, gathered_array )

    # Calculate the space-charge fields on the global grid
    # (For a multi-proc simulation: only performed by the first proc)
    if sim.comm.rank == 0:
        # - Convert the sources to spectral space
        global_fld.interp2spect('rho_prev')
        global_fld.interp2spect('J')
        if sim.filter_currents:
            global_fld.filter_spect('rho_prev')
            global_fld.filter_spect('J')
        # - Get the space charge fields in spectral space
        for m in range(global_fld.Nm) :
            get_space_charge_spect( global_fld.spect[m], gamma, direction )
        # - Convert the fields back to real space
        global_fld.spect2interp( 'E' )
        global_fld.spect2interp( 'B' )

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
                                    global_array, with_damp=True )
            # Add it to the fields of sim.fld
            local_field = getattr( sim.fld.interp[m], field )
            local_field[ iz_in_array:iz_in_array+Nz_local, : ] += local_array

    if sim.comm.rank == 0:
        print("Done.\n")


def get_space_charge_spect( spect, gamma, direction='forward',
                             neglect_transverse_currents=True ) :
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

    neglect_transverse_currents: bool
        Whether to neglect the fields generated by the transverse components
        of the current. This approximation is very often made when calculating
        space charge. Also, when this is False, spurious fields were sometimes
        observed in the modes m>0, for high-energy particle bunches.
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
    phi = spect.rho_prev[:,:]*inv_K2[:,:]/epsilon_0
    if not neglect_transverse_currents:
        Ap = spect.Jp[:,:]*inv_K2[:,:]*mu_0
        Am = spect.Jm[:,:]*inv_K2[:,:]*mu_0
    Az = spect.Jz[:,:]*inv_K2[:,:]*mu_0

    # Deduce the E field
    spect.Ep[:,:] += 0.5*spect.kr * phi
    spect.Em[:,:] += -0.5*spect.kr * phi
    spect.Ez[:,:] += -1.j*spect.kz * phi + 1.j*beta*c*spect.kz * Az
    if not neglect_transverse_currents:
        spect.Ep[:,:] += 1.j*beta*c*spect.kz * Ap
        spect.Em[:,:] += 1.j*beta*c*spect.kz * Am

    # Deduce the B field
    spect.Bp[:,:] += -0.5j*spect.kr * Az
    spect.Bm[:,:] += -0.5j*spect.kr * Az
    if not neglect_transverse_currents:
        spect.Bp[:,:] += spect.kz * Ap
        spect.Bm[:,:] -= spect.kz * Am
        spect.Bz[:,:] += 1.j*spect.kr * Ap + 1.j*spect.kr * Am
