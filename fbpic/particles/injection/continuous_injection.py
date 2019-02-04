# Copyright 2018, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a class for continuous particle injection with a moving window.
"""
import warnings
import numpy as np
from scipy.constants import c

class ContinuousInjector( object ):
    """
    Class that stores a number of attributes that are needed for
    continuous injection by a moving window.
    """

    def __init__(self, Npz, zmin, zmax, dz_particles, Npr, rmin, rmax,
                Nptheta, n, dens_func, ux_m, uy_m, uz_m, ux_th, uy_th, uz_th ):
        """
        Initialize continuous injection

        Parameters
        ----------
        See the docstring of the `Particles` object
        """
        # Register properties of the injected plasma
        self.Npr = Npr
        self.rmin = rmin
        self.rmax = rmax
        self.Nptheta = Nptheta
        self.n = n
        self.dens_func = dens_func
        self.ux_m = ux_m
        self.uy_m = uy_m
        self.uz_m = uz_m
        self.ux_th = ux_th
        self.uy_th = uy_th
        self.uz_th = uz_th

        # Register spacing between evenly-spaced particles in z
        if Npz != 0:
            self.dz_particles = (zmax - zmin)/Npz
        else:
            # Fall back to the user-provided `dz_particles`.
            # Note: this is an optional argument of `Particles` and so
            # it is not always available.
            self.dz_particles = dz_particles

        # Register variables that define the positions
        # where the plasma is injected.
        self.v_end_plasma = c * uz_m / np.sqrt(1 + ux_m**2 + uy_m**2 + uz_m**2)
        # These variables are set by `initialize_injection_positions`
        self.nz_inject = None
        self.z_inject = None
        self.z_end_plasma = None


    def initialize_injection_positions( self, comm, v_moving_window,
                                        species_z, dt ):
        """
        Initialize the positions that keep track of the injection of particles.
        This is automatically called at the beginning of `step`.

        Parameters
        ----------
        comm: a BoundaryCommunicator object
            Contains information about grid MPI decomposition
        v_moving_window: float (in m/s)
            The speed of the moving window
        species_z: 1darray of float (in m)
            (One element per macroparticle)
            Used in order to infer the position of the end of the plasma
        dt: float (in s)
            Timestep of the simulation
        """
        # The injection position is only initialized for the last proc
        if comm.rank != comm.size-1:
            return
        # Initialize the injection position only if it has not be initialized
        if self.z_inject is not None:
            return

        # Initialize plasma *ahead* of the right *physical*
        # boundary of the box in the damping region (including the
        # injection area) so that after `exchange_period` iterations
        # (without adding new plasma), there will still be plasma
        # inside the physical domain and the damping region (without the
        # injection area). This ensures that there are never particles in the
        # rightmost guard region and that there are always particles inside
        # the damped region, where the field can be non-zero. New particles,
        # which are injected in the Injection region, do not see any fields.
        _, zmax_global_domain_with_damp = comm.get_zmin_zmax( local=False,
                                    with_damp=True, with_guard=False )
        self.z_inject = zmax_global_domain_with_damp \
                + (3-comm.n_inject)*comm.dz \
                + comm.exchange_period*dt*(v_moving_window-self.v_end_plasma)
        self.nz_inject = 0
        # Try to detect the position of the end of the plasma:
        # Find the maximal position of the continously-injected particles
        if len( species_z ) > 0:
            # Add half of the spacing between particles (the
            # injection function itself will add a half-spacing again)
            self.z_end_plasma = species_z.max() + 0.5*self.dz_particles
        else:
            # Default value for empty species
            _, zmax_global_physical_domain = comm.get_zmin_zmax( local=False,
                                    with_damp=False, with_guard=False )
            self.z_end_plasma = zmax_global_physical_domain

        # Check that the particle spacing has been properly calculated
        if self.dz_particles is None:
            raise ValueError(
                'The simulation uses continuous injection of particles, \n'
                'but was unable to calculate the spacing between particles.\n'
                'This may be because you used the `Particles` API directly.\n'
                'In this case, please pass the argument `dz_particles` \n'
                'initializing the `Particles` object.')

    def reset_injection_positions( self ):
        """
        Reset the variables that keep track of continuous injection to `None`
        This is typically called when restarting a simulation from a checkpoint
        """
        self.nz_inject = None
        self.z_inject = None
        self.z_end_plasma = None

    def increment_injection_positions( self, v_moving_window, duration ):
        """
        Update the positions between which the new particles will be generated,
        the next time when `generate_particles` is called.
        This function is automatically called when the moving window moves.

        Parameters
        ----------
        v_moving_window: float (in m/s)
            The speed of the moving window

        duration: float (in seconds)
            The duration since the last time that the moving window moved.
        """
        # Move the injection position
        self.z_inject += v_moving_window * duration
        # Take into account the motion of the end of the plasma
        self.z_end_plasma += self.v_end_plasma * duration

        # Increment the number of particle to add along z
        nz_new = int( (self.z_inject - self.z_end_plasma)/self.dz_particles )
        self.nz_inject += nz_new
        # Increment the virtual position of the end of the plasma
        # (When `generate_particles` is called, then the plasma
        # is injected between z_end_plasma - nz_inject*dz_particles
        # and z_end_plasma, and afterwards nz_inject is set to 0.)
        self.z_end_plasma += nz_new * self.dz_particles


    def generate_particles( self, time ):
        """
        Generate new particles at the right end of the plasma
        (i.e. between z_end_plasma - nz_inject*dz and z_end_plasma)

        Parameters
        ----------
        time: float (in second)
            The current physical time of the simulation
        """
        # Create a temporary density function that takes into
        # account the fact that the plasma has moved
        if self.dens_func is not None:
            def dens_func( z, r ):
                return( self.dens_func( z-self.v_end_plasma*time, r ) )
        else:
            dens_func = None

        # Create new particle cells
        # Determine the positions between which new particles will be created
        Npz = self.nz_inject
        zmax = self.z_end_plasma
        zmin = self.z_end_plasma - self.nz_inject*self.dz_particles
        # Create the particles
        Ntot, x, y, z, ux, uy, uz, inv_gamma, w = generate_evenly_spaced(
                Npz, zmin, zmax, self.Npr, self.rmin, self.rmax,
                self.Nptheta, self.n, dens_func,
                self.ux_m, self.uy_m, self.uz_m,
                self.ux_th, self.uy_th, self.uz_th )

        # Reset the number of particle cells to be created
        self.nz_inject = 0

        return( Ntot, x, y, z, ux, uy, uz, inv_gamma, w )


# Utility functions
# -----------------

def generate_evenly_spaced( Npz, zmin, zmax, Npr, rmin, rmax,
    Nptheta, n, dens_func, ux_m, uy_m, uz_m, ux_th, uy_th, uz_th ):
    """
    Generate evenly-spaced particles, according to the density function
    `dens_func`, and with the momenta given by the `ux/y/z` arguments.

    Parameters
    ----------
    See the docstring of the `Particles` object
    """
    # Generate the particles and eliminate the ones that have zero weight ;
    # infer the number of particles Ntot
    if Npz*Npr*Nptheta > 0:
        # Get the 1d arrays of evenly-spaced positions for the particles
        dz = (zmax-zmin)*1./Npz
        z_reg =  zmin + dz*( np.arange(Npz) + 0.5 )
        dr = (rmax-rmin)*1./Npr
        r_reg =  rmin + dr*( np.arange(Npr) + 0.5 )
        dtheta = 2*np.pi/Nptheta
        theta_reg = dtheta * np.arange(Nptheta)

        # Get the corresponding particles positions
        # (copy=True is important here, since it allows to
        # change the angles individually)
        zp, rp, thetap = np.meshgrid( z_reg, r_reg, theta_reg,
                                    copy=True, indexing='ij' )
        # Prevent the particles from being aligned along any direction
        unalign_angles( thetap, Npz, Npr, method='random' )
        # Flatten them (This performs a memory copy)
        r = rp.flatten()
        x = r * np.cos( thetap.flatten() )
        y = r * np.sin( thetap.flatten() )
        z = zp.flatten()
        # Get the weights (i.e. charge of each macroparticle), which
        # are equal to the density times the volume r d\theta dr dz
        w = n * r * dtheta*dr*dz
        # Modulate it by the density profile
        if dens_func is not None :
            w *= dens_func( z, r )

        # Select the particles that have a non-zero weight
        selected = (w > 0)
        if np.any(w < 0):
            warnings.warn(
            'The specified particle density returned negative densities.\n'
            'No particles were generated in areas of negative density.\n'
            'Please check the validity of the `dens_func`.')

        # Infer the number of particles and select them
        Ntot = int(selected.sum())
        x = x[ selected ]
        y = y[ selected ]
        z = z[ selected ]
        w = w[ selected ]
        # Initialize the corresponding momenta
        uz = uz_m * np.ones(Ntot) + uz_th * np.random.normal(size=Ntot)
        ux = ux_m * np.ones(Ntot) + ux_th * np.random.normal(size=Ntot)
        uy = uy_m * np.ones(Ntot) + uy_th * np.random.normal(size=Ntot)
        inv_gamma = 1./np.sqrt( 1 + ux**2 + uy**2 + uz**2 )
        # Return the particle arrays
        return( Ntot, x, y, z, ux, uy, uz, inv_gamma, w )
    else:
        # No particles are initialized ; the arrays are still created
        Ntot = 0
        return( Ntot, np.empty(0), np.empty(0), np.empty(0), np.empty(0),
                      np.empty(0), np.empty(0), np.empty(0), np.empty(0) )


def unalign_angles( thetap, Npz, Npr, method='irrational' ) :
    """
    Shift the angles so that the particles are
    not all aligned along the arms of a star transversely

    The fact that the particles are all aligned can produce
    numerical artefacts, especially if the polarization of the laser
    is aligned with this direction.

    Here, for each position in r and z, we add the *same*
    shift for all the Nptheta particles that are at this position.
    (This preserves the fact that certain modes are 0 initially.)
    How this shift varies from one position to another depends on
    the method chosen.

    Parameters
    ----------
    thetap : 3darray of floats
        An array of shape (Npr, Npz, Nptheta) containing the angular
        positions of the particles, and which is modified by this function.

    Npz, Npr : ints
        The number of macroparticles along the z and r directions

    method : string
        Either 'random' or 'irrational'
    """
    # Determine the angle shift
    if method == 'random' :
        angle_shift = 2*np.pi*np.random.rand(Npz, Npr)
    elif method == 'irrational' :
        # Subrandom sequence, by adding irrational number (sqrt(2) and sqrt(3))
        # This ensures that the sequence does not wrap around and induce
        # correlations
        shiftr = np.sqrt(2)*np.arange(Npr)
        shiftz = np.sqrt(3)*np.arange(Npz)
        angle_shift = 2*np.pi*( shiftz[:,np.newaxis] + shiftr[np.newaxis,:] )
        angle_shift = np.mod( angle_shift, 2*np.pi )
    else :
        raise ValueError(
      "method must be either 'random' or 'irrational' but is %s" %method )

    # Add the angle shift to thetap
    # np.newaxis ensures that the angles that are at the same positions
    # in r and z have the same shift
    thetap[:,:,:] = thetap[:,:,:] + angle_shift[:,:, np.newaxis]
