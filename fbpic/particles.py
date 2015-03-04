"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure and methods associated with the particles.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import double
from numba.decorators import jit, autojit
from scipy.constants import c

class Particles(object) :
    """
    Class that contains the particles data of the simulation

    Main attributes
    ---------------
    - x, y, z : 1darrays containing the Cartesian positions
                of the macroparticles (in meters)
    - uz, uy, uz : 1darrays containing the normalized momenta
                of the macroparticles (unitless)
    
    """

    def __init__(self, q, m, rho, Npz, zmin, zmax,
                    Npr, rmin, rmax, Nptheta, dt ) :
        """
        Initialize a uniform set of particles

        Parameters
        ----------
        q : float (in Coulombs)
           Charge of the particle species 

        m : float (in kg)
           Mass of the particle species 

        rho : float (in Coulombs per m^3)
           Uniform charge density of the macroparticles
           
        Npz : int
           Number of macroparticles along the z axis
           
        zmin, zmax : floats (in meters)
           z positions between which the particles are initialized

        Npr : int
           Number of macroparticles along the r axis

        rmin, rmax : floats (in meters)
           r positions between which the particles are initialized

        Nptheta : int
           Number of macroparticules along theta

        dt : float (in seconds)
           The timestep for the particle pusher
        """
        # Register the timestep
        self.dt = dt
        
        # Register the properties of the particles
        self.Ntot = Npz*Npr*Nptheta
        self.q = q
        self.m = m

        # Initialize the (normalized) momenta
        self.uz = np.zeros( self.Ntot )
        self.ux = np.zeros( self.Ntot )
        self.uy = np.zeros( self.Ntot )
        self.inv_gamma = np.ones( self.Ntot )

        # Initilialize the fields array (at the positions of the particles)
        self.Ez = np.zeros( self.Ntot )
        self.Ex = np.zeros( self.Ntot )
        self.Ey = np.zeros( self.Ntot )
        self.Bz = np.zeros( self.Ntot )
        self.Bx = np.zeros( self.Ntot )
        self.By = np.zeros( self.Ntot )
        
        # Get the 1d arrays of regularly-spaced positions for the particles
        dz = (zmax-zmin)*1./Npz
        z_reg =  zmin + dz*( np.arange(Npz) + 0.5 )
        dr = (rmax-rmin)*1./Npr
        r_reg =  rmin + dr*( np.arange(Npr) + 0.5 )
        dtheta = 2*np.pi/Nptheta
        theta_reg = dtheta * np.arange(Nptheta)

        # Get the corresponding particles positions (with no memory copy)
        zp, rp, thetap = np.meshgrid( z_reg, r_reg, theta_reg, copy=False)
        # Flatten them (This performs a memory copy)
        self.z = zp.flatten()
        self.x = rp.flatten()*np.cos( thetap.flatten() )
        self.y = rp.flatten()*np.sin( thetap.flatten() )

        # Get the weights (i.e. charge of each macroparticle), which are equal
        # to the density times the elementary volume r d\theta dr dz
        self.w = rho * rp.flatten() * dtheta*dr*dz
    
        
    def push_p(self) :
        """
        Advance the particles' momenta over one timestep, using the Vay pusher
        Reference : Vay, Physics of Plasmas 15, 056701 (2008)
        """
        # Set a few constants
        econst = self.q*self.dt/(self.m*c)
        bconst = 0.5*self.q*self.dt/self.m
        
        # Get the magnetic rotation vector
        taux = bconst*self.Bx
        tauy = bconst*self.By
        tauz = bconst*self.Bz
        tau2 = taux**2 + tauy**2 + tauz**2

        # Get the momenta at the half timestep
        ux = self.ux + econst*self.Ex \
          + self.invgamma*( self.uy*tauz - self.uz*tauy )
        uy = self.uy + econst*self.Ey \
          + self.invgamma*( self.uz*taux - self.ux*tauz )
        uz = self.uz + econst*self.Ez \
          + self.invgamma*( self.ux*tauy - self.uy*taux )
        sigma = 1 + ux**2 + uy**2 + uz**2 - tau2
        utau = ux*taux + uy*tauy + uz*tauz

        # Get the new 1./gamma
        self.invgamma = np.sqrt(
        2./( sigma + np.sqrt( sigma**2 + 4*(tau2 + utau**2 ) ) )
        )

        # Reuse the tau and utau arrays to save memory
        taux[:] = self.invgamma*taux
        tauy[:] = self.invgamma*tauy
        tauz[:] = self.invgamma*tauz
        utau[:] = self.invgamma*utau
        s = 1./( 1 + tau2*self.invgamma**2 )

        # Get the new u
        self.ux = s*( ux + taux*utau + uy*tauz - uz*tauy )
        self.uy = s*( uy + tauy*utau + uz*taux - ux*tauz )
        self.uz = s*( uz + tauz*utau + ux*tauy - uy*taux )
        

    def halfpush_x(self) :
        """
        Advance the particles' positions over one half-timestep
        """
        # Half timestep, multiplied by c
        chdt = c*0.5*self.dt

        # Particle push
        self.x = self.x + chdt*self.invgamma*self.ux
        self.y = self.y + chdt*self.invgamma*self.uy
        self.z = self.z + chdt*self.invgamma*self.uz
        
