"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure and methods associated with the particles.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
#from numba import double
#from numba.decorators import jit, autojit

class Particles(object) :
    """
    Class that contains the particles data of the simulation

    Main attributes
    ---------------
    - x, y, z : 1darrays containing the Cartesian positions
                of the macroparticles (in meters)
    - uz, uy, uz : 1darrays containing the unitless momenta
                (i.e. px/mc, py/mc, pz/mc)
    
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
        
        
    def gather(self, grid) :
        """
        Gather the fields onto the macroparticles using numpy
        
        Parameter
        ----------
        grid : a list of InterpolationGrid objects (one per azimuthal mode)
             Contains the field values on the interpolation grid
        """

        # Preliminary arrays for the cylindrical conversion
        r = np.sqrt( self.x**2 + self.y**2 )
        invr = 1./r
        c = self.x*invr  # Cosine
        s = self.y*invr  # Sine

        # Indices and weights
        iz_lower, iz_upper, Sz_lower = \
          linear_weights( z, grid[0].invdz, 0. )
        ir_lower, ir_upper, Sr_lower = \
          linear_weights( r, grid[0].invdr, 0.5*grid[0].dr )

        # -------------------------------
        # Gather the E field mode by mode
        # -------------------------------
        # Zero the previous fields
        self.Ex[:] = 0.
        self.Ey[:] = 0.
        self.Ez[:] = 0.
        # Prepare auxiliary matrices
        Ft = np.zeros(self.Ntot)
        Fr = np.zeros(self.Ntot)
        exptheta = np.ones(self.Ntot, dtype='complex')
        # exptheta takes the value exp(-im theta) throughout the loop
        for m in range(Nm) :
            gather_field( exptheta, m, grid[m].Ez, self.Ez, 
                iz_lower, iz_upper, Sz_lower,
                ir_lower, ir_upper, Sr_lower )
            gather_field( exptheta, m, grid[m].Er, Fr, 
                iz_lower, iz_upper, Sz_lower,
                ir_lower, ir_upper, Sr_lower )
            gather_field( exptheta, m, grid[m].Et, Ft, 
                iz_lower, iz_upper, Sz_lower,
                ir_lower, ir_upper, Sr_lower )
            # Increment exptheta (notice the - : backward Fourier transform)
            exptheta = exptheta*( c - 1.j*s )
        # Convert to Cartesian coordinates
        self.Ex[:] = c*Fr - s*Ft
        self.Ey[:] = s*Fr + c*Ft

        # -------------------------------
        # Gather the B field mode by mode
        # -------------------------------
        # Zero the previous fields
        self.Bx[:] = 0.
        self.By[:] = 0.
        self.Bz[:] = 0.
        # Prepare auxiliary matrices
        Ft[:] = 0.
        Fr[:] = 0.
        exptheta[:] = 1.
        # exptheta takes the value exp(-im theta) throughout the loop
        for m in range(Nm) :
            gather_field( exptheta, m, grid[m].Bz, self.Bz, 
                iz_lower, iz_upper, Sz_lower,
                ir_lower, ir_upper, Sr_lower )
            gather_field( exptheta, m, grid[m].Br, Fr, 
                iz_lower, iz_upper, Sz_lower,
                ir_lower, ir_upper, Sr_lower )
            gather_field( exptheta, m, grid[m].Bt, Ft, 
                iz_lower, iz_upper, Sz_lower,
                ir_lower, ir_upper, Sr_lower )
            # Increment exptheta (notice the - : backward Fourier transform)
            exptheta = exptheta*( c - 1.j*s )
        # Convert to Cartesian coordinates
        self.Bx[:] = c*Fr - s*Ft
        self.By[:] = s*Fr + c*Ft

        
    def deposit(self, grid, fieldtype) :
        """
        Deposit the particles charge or current onto the grid, using numpy
        
        Parameter
        ----------
        grid : a list of InterpolationGrid objects
             Contains the field values on the interpolation grid

        fieldtype : string
             Indicates which field to deposit
             Either 'J' or 'rho'
        """
        # Check the validity of fieldtype
        if ( fieldtype in ['J', 'rho'] ) == False :
            raise ValueError(
                "`fieldtype` should be either 'J' or 'rho', but is `%s`" \
                           %fieldtype )
        
        # Preliminary arrays for the cylindrical conversion
        r = np.sqrt( self.x**2 + self.y**2 )
        invr = 1./r
        c = self.x*invr  # Cosine
        s = self.y*invr  # Sine

        # Indices and weights
        iz_lower, iz_upper, Sz_lower = linear_weights( z, 1./dz, 0. )
        ir_lower, ir_upper, Sr_lower = linear_weights( r, 1./dr, 0.5*dr )

        if fieldtype == 'rho' :
            # ---------------------------------------
            # Deposit the charge density mode by mode
            # ---------------------------------------
            # Prepare auxiliary matrix
            exptheta = np.ones( self.Ntot, dtype='complex')
            # exptheta takes the value exp(im theta) throughout the loop
            for m in range(Nm) :
                deposit_field( self.w*exptheta, grid[m].rho, 
                    iz_lower, iz_upper, Sz_lower,
                    ir_lower, ir_upper, Sr_lower )
                # Increment exptheta (notice the + : forward Fourier transform)
                exptheta = exptheta*( c + 1.j*s )
            
        elif fieldtype == 'J' :
            # ----------------------------------------
            # Deposit the current density mode by mode
            # ----------------------------------------
            # Calculate the currents
            Jr = self.w*self.invgamma * ( c*self.ux + s*self.uy )
            Jt = self.w*self.invgamma * ( c*self.uy - s*self.ux )
            Jz = self.w*self.invgamma * self.uz
            # Prepare auxiliary matrix
            exptheta[:] = np.ones( self.Ntot, dtype='complex')
            # exptheta takes the value exp(-im theta) throughout the loop
            for m in range(Nm) :
                deposit_field( Jr*exptheta, grid[m].Jr, 
                    iz_lower, iz_upper, Sz_lower,
                    ir_lower, ir_upper, Sr_lower )
                deposit_field( Jt*exptheta, grid[m].Jt, 
                    iz_lower, iz_upper, Sz_lower,
                    ir_lower, ir_upper, Sr_lower )
                deposit_field( Jz*exptheta, grid[m].Jz, 
                    iz_lower, iz_upper, Sz_lower,
                    ir_lower, ir_upper, Sr_lower )
                # Increment exptheta (notice the + : forward Fourier transform)
                exptheta = exptheta*( c + 1.j*s )


def linear_weights(x, invdx, offset, Nx) :
    """
    Return the matrix indices and the shape factors, for linear shapes.

    Parameters
    ----------
    x : 1darray of floats (in meters)
        Array of particle positions along a given direction
        (one element per macroparticle)

    invdx : float (in meters^-1)
        Inverse of the grid step along the considered direction

    offset : float (in meters)
        Position of the first node of the grid along the considered direction

    Nx : int
        Number of gridpoints along the considered direction

    Returns
    -------
    A tuple containing :
    
    i_lower : 1darray of integers
        (one element per macroparticle)
        Contains the index of the cell immediately below each
        macroparticle, along the considered axis
    i_upper : 1darray of integers
        (one element per macroparticle)
        Contains the index of the cell immediately above each
        macroparticle, along the considered axis
    S_lower : 1darray of floats
        (one element per macroparticle)
        Contains the weight for the lower cell, for each macroparticle.
        The weight for the upper cell is just 1-S_lower.
    """
    
    # Index of the uppper and lower cell
    i_lower = np.floor( invdx*(x - offset) ).astype('int')  
    i_upper = i_lower + 1
    
    # Avoid out-of-bounds indices
    i_lower = np.where( i_lower < 0, 0, i_lower )
    i_lower = np.where( i_lower > Nx-1, Nx-1, i_lower )
    i_upper = np.where( i_upper < 0, 0, i_upper )
    i_upper = np.where( i_upper > Nx-1, Nx-1, i_upper )

    # Linear weight
    S_lower = 1. - ( invdx*(x - offset) - i_lower )

    return( i_lower, i_upper, S_lower )



def gather_field( exptheta, m, Fgrid, Fptcl, 
        iz_lower, iz_upper, Sz_lower, ir_lower, ir_upper, Sr_lower ) :
    """
    Perform the weighted sum from the 4 points that surround each particle,
    for one given field and one given azimuthal mode

    Parameters
    ----------
    exptheta : 1darray of complexs
        (one element per macroparticle)
        Contains exp(-im theta) for each macroparticle

    m : int
        Index of the mode.
        Determines wether a factor 2 should be applied
    
    Fgrid : 2darray of complexs
        Contains the fields on the interpolation grid,
        from which to do the gathering

    Fptcl : 1darray of floats
        (one element per macroparticle)
        Contains the fields for each macroparticle
        Is modified by this function

    iz_lower, iz_upper, ir_lower, ir_upper : 1darrays of integers
        (one element per macroparticle)
        Contains the index of the cells immediately below and
        immediately above each macroparticle, in z and r
        
    Sz_lower, Sr_lower : 1darrays of floats
        (one element per macroparticle)
        Contains the weight for the lower cell, for each macroparticle.
        The weight for the upper cell is just 1-S_lower.
    """
    # Temporary matrix that contains the complex fields
    F = np.zeros_like(exptheta)
    
    # Sum the fields from the 4 points
    # NB : These operations could be made maybe
    # twice faster with flattened indices and np.take
    # Lower cell in z, Lower cell in r
    F += Sz_lower*Sr_lower*Fgrid[ iz_lower, ir_lower ]
    # Lower cell in z, Upper cell in r
    F += Sz_lower*(1-Sr_lower)*Fgrid[ iz_lower, ir_upper ]
    # Upper cell in z, Lower cell in r
    F += (1-Sz_lower)*Sr_lower*Fgrid[ iz_upper, ir_lower ]
    # Upper cell in z, Upper cell in r
    F += (1-Sz_lower)*(1-Sr_lower)*Fgrid[ iz_upper, ir_upper ]

    # Add the complex phase
    if m == 0 :
        Fptcl += (F*exptheta).real
    if m > 0 :
        Fptcl += 2*(F*exptheta).real


def deposit_field( Fptcl, Fgrid, 
        iz_lower, iz_upper, Sz_lower, ir_lower, ir_upper, Sr_lower ) :
    """
    Perform the deposition on the 4 points that surround each particle,
    for one given field and one given azimuthal mode

    Parameters
    ----------
    Fptcl : 1darray of complexs
        (one element per macroparticle)
        Contains the charge or current for each macroparticle (already
        multiplied by exp(im theta), from which to do the deposition
    
    Fgrid : 2darray of complexs
        Contains the fields on the interpolation grid.
        Is modified by this function

    iz_lower, iz_upper, ir_lower, ir_upper : 1darrays of integers
        (one element per macroparticle)
        Contains the index of the cells immediately below and
        immediately above each macroparticle, in z and r
        
    Sz_lower, Sr_lower : 1darrays of floats
        (one element per macroparticle)
        Contains the weight for the lower cell, for each macroparticle.
        The weight for the upper cell is just 1-S_lower. 
    """
    # Deposit the particle quantity onto the grid
    # Lower cell in z, Lower cell in r
    np.add.at( Fgrid, (iz_lower, ir_lower), Sz_lower*Sr_lower*Fptcl ) 
    # Lower cell in z, Upper cell in r
    np.add.at( Fgrid, (iz_lower, ir_upper), Sz_lower*(1-Sr_lower)*Fptcl )
    # Upper cell in z, Lower cell in r
    np.add.at( Fgrid, (iz_upper, ir_lower), (1-Sz_lower)*Sr_lower*Fptcl )
    # Upper cell in z, Upper cell in r
    np.add.at( Fgrid, (iz_upper, ir_lower), (1-Sz_lower)*(1-Sr_lower)*Fptcl )
