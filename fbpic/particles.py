"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure and methods associated with the particles.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
# If numba is installed, it can make the code much faster
try :
    import numba
except ImportError :
    numba_installed = False
else :
    numba_installed = True

class Particles(object) :
    """
    Class that contains the particles data of the simulation

    Main attributes
    ---------------
    - x, y, z : 1darrays containing the Cartesian positions
                of the macroparticles (in meters)
    - uz, uy, uz : 1darrays containing the unitless momenta
                (i.e. px/mc, py/mc, pz/mc)
    At the end or start of any PIC cycle, the momenta should be
    one half-timestep *behind* the position.
    """

    def __init__(self, q, m, n, Npz, zmin, zmax,
                    Npr, rmin, rmax, Nptheta, dt, global_theta=0. ) :
        """
        Initialize a uniform set of particles

        Parameters
        ----------
        q : float (in Coulombs)
           Charge of the particle species 

        m : float (in kg)
           Mass of the particle species 

        n : float (in particles per m^3)
           Uniform density of particles
           
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

        global_theta : float (in rad)
           A global shift on all the theta of the particles
           This is useful when repetitively adding new particles
           (e.g. with the moving window), in order to avoid that
           the successively-added particles are aligned.
        """
        # Register the timestep
        self.dt = dt
        
        # Register the properties of the particles
        # (Necessary for the pusher, and when adding more particles later, )
        self.Ntot = Npz*Npr*Nptheta
        self.q = q
        self.m = m
        self.n = n
        self.rmin = rmin
        self.rmax = rmax
        self.Npr = Npr
        self.Nptheta = Nptheta

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
        zp, rp, thetap = np.meshgrid( z_reg, r_reg, theta_reg, copy=True)
        # Prevent the particles from being aligned along any direction
        unalign_angles( thetap, Npr,Npz, method='irrational' )
        thetap += global_theta
        # Flatten them (This performs a memory copy)
        self.z = zp.flatten()
        self.x = rp.flatten()*np.cos( thetap.flatten() )
        self.y = rp.flatten()*np.sin( thetap.flatten() )

        # Get the weights (i.e. charge of each macroparticle), which are equal
        # to the density times the elementary volume r d\theta dr dz
        self.w = q * n * rp.flatten() * dtheta*dr*dz
        
    def push_p(self, use_numba=numba_installed ) :
        """
        Advance the particles' momenta over one timestep, using the Vay pusher
        Reference : Vay, Physics of Plasmas 15, 056701 (2008)

        This assumes that the momenta (ux, uy, uz) are initially one
        half-timestep *behind* the positions (x, y, z), and it brings
        them one half-timestep *ahead* of the positions.

        Parameter
        ---------
        use_numba : bool, optional
            Whether to use numba rather than numpy
        """
        if use_numba :
            push_p_numba(self.ux, self.uy, self.uz, self.inv_gamma, 
                self.Ex, self.Ey, self.Ez, self.Bx, self.By, self.Bz,
                self.q, self.m, self.Ntot, self.dt )
        else :
            self.push_p_numpy()

    def push_p_numpy(self, use_numba=numba_installed ) :
        """
        Advance the particles' momenta, using numba
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
          + self.inv_gamma*( self.uy*tauz - self.uz*tauy )
        uy = self.uy + econst*self.Ey \
          + self.inv_gamma*( self.uz*taux - self.ux*tauz )
        uz = self.uz + econst*self.Ez \
          + self.inv_gamma*( self.ux*tauy - self.uy*taux )
        sigma = 1 + ux**2 + uy**2 + uz**2 - tau2
        utau = ux*taux + uy*tauy + uz*tauz

        # Get the new 1./gamma
        self.inv_gamma = np.sqrt(
        2./( sigma + np.sqrt( sigma**2 + 4*(tau2 + utau**2 ) ) )
        )

        # Reuse the tau and utau arrays to save memory
        taux[:] = self.inv_gamma*taux
        tauy[:] = self.inv_gamma*tauy
        tauz[:] = self.inv_gamma*tauz
        utau[:] = self.inv_gamma*utau
        s = 1./( 1 + tau2*self.inv_gamma**2 )

        # Get the new u
        self.ux = s*( ux + taux*utau + uy*tauz - uz*tauy )
        self.uy = s*( uy + tauy*utau + uz*taux - ux*tauz )
        self.uz = s*( uz + tauz*utau + ux*tauy - uy*taux )
        
    def halfpush_x(self) :
        """
        Advance the particles' positions over one half-timestep
        
        This assumes that the positions (x, y, z) are initially either
        one half-timestep *behind* the momenta (ux, uy, uz), or at the
        same timestep as the momenta.
        """
        # Half timestep, multiplied by c
        chdt = c*0.5*self.dt

        # Particle push
        self.x = self.x + chdt*self.inv_gamma*self.ux
        self.y = self.y + chdt*self.inv_gamma*self.uy
        self.z = self.z + chdt*self.inv_gamma*self.uz
        
        
    def gather(self, grid, use_numba=numba_installed) :
        """
        Gather the fields onto the macroparticles using numpy

        This assumes that the particle positions are currently at
        the same timestep as the field that is to be gathered.
        
        Parameter
        ----------
        grid : a list of InterpolationGrid objects
             (one InterpolationGrid object per azimuthal mode)
             Contains the field values on the interpolation grid

        use_numba : bool, optional
             Whether to use numba or numpy in the core deposition function
             Default : Use if numba is installed, use it
        """        
        # Preliminary arrays for the cylindrical conversion
        r = np.sqrt( self.x**2 + self.y**2 )
        invr = 1./r
        cos = self.x*invr  # Cosine
        sin = self.y*invr  # Sine

        # Indices and weights
        iz_lower, iz_upper, Sz_lower, Sz_upper = linear_weights(
           self.z, grid[0].invdz, grid[0].zmin, grid[0].Nz, direction='z' )
        ir_lower, ir_upper, Sr_lower, Sr_upper, Sr_guard = linear_weights(
            r, grid[0].invdr, grid[0].rmin, grid[0].Nr, direction='r' )

        # Number of modes considered :
        # number of elements in the grid list
        Nm = len(grid)
        
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
            # Increment exptheta (notice the - : backward transform)
            if m==1 :
                exptheta[:].real = cos
                exptheta[:].imag = -sin
            elif m>1 :
                exptheta[:] = exptheta*( cos - 1.j*sin )
            # Gather the fields
            # (The sign with which the guards are added
            # depends on whether the fields should be zero on axis)
            gather_field( exptheta, m, grid[m].Er, Fr, 
                iz_lower, iz_upper, Sz_lower, Sz_upper,
                ir_lower, ir_upper, Sr_lower, Sr_upper,
                -(-1.)**m, Sr_guard, use_numba )
            gather_field( exptheta, m, grid[m].Et, Ft, 
                iz_lower, iz_upper, Sz_lower, Sz_upper,
                ir_lower, ir_upper, Sr_lower, Sr_upper,
                -(-1.)**m, Sr_guard, use_numba )
            gather_field( exptheta, m, grid[m].Ez, self.Ez, 
                iz_lower, iz_upper, Sz_lower, Sz_upper,
                ir_lower, ir_upper, Sr_lower, Sr_upper,
                (-1.)**m, Sr_guard, use_numba )

        # Convert to Cartesian coordinates
        self.Ex[:] = cos*Fr - sin*Ft
        self.Ey[:] = sin*Fr + cos*Ft

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
            # Increment exptheta (notice the - : backward transform)
            if m==1 :
                exptheta[:].real = cos
                exptheta[:].imag = -sin
            elif m>1 :
                exptheta[:] = exptheta*( cos - 1.j*sin )
            # Gather the fields
            # (The sign with which the guards are added
            # depends on whether the fields should be zero on axis)
            gather_field( exptheta, m, grid[m].Br, Fr, 
                iz_lower, iz_upper, Sz_lower, Sz_upper,
                ir_lower, ir_upper, Sr_lower, Sr_upper,
                -(-1.)**m, Sr_guard, use_numba )
            gather_field( exptheta, m, grid[m].Bt, Ft, 
                iz_lower, iz_upper, Sz_lower, Sz_upper,
                ir_lower, ir_upper, Sr_lower, Sr_upper,
                -(-1.)**m, Sr_guard, use_numba )
            gather_field( exptheta, m, grid[m].Bz, self.Bz, 
                iz_lower, iz_upper, Sz_lower, Sz_upper,
                ir_lower, ir_upper, Sr_lower, Sr_upper,
                (-1.)**m, Sr_guard, use_numba )

        # Convert to Cartesian coordinates
        self.Bx[:] = cos*Fr - sin*Ft
        self.By[:] = sin*Fr + cos*Ft

        
    def deposit(self, grid, fieldtype, use_numba=numba_installed ) :
        """
        Deposit the particles charge or current onto the grid, using numpy
        
        This assumes that the particle positions (and momenta in the case of J)
        are currently at the same timestep as the field that is to be deposited.
        
        Parameter
        ----------
        grid : a list of InterpolationGrid objects
             (one InterpolationGrid object per azimuthal mode)
             Contains the field values on the interpolation grid

        fieldtype : string
             Indicates which field to deposit
             Either 'J' or 'rho'

        use_numba : bool, optional
             Whether to use numba or numpy in the core deposition function
             Default : Use if numba is installed, use it
        """        
        # Preliminary arrays for the cylindrical conversion
        r = np.sqrt( self.x**2 + self.y**2 )
        invr = 1./r
        cos = self.x*invr  # Cosine
        sin = self.y*invr  # Sine

        # Indices and weights
        iz_lower, iz_upper, Sz_lower, Sz_upper = linear_weights( 
            self.z, grid[0].invdz, grid[0].zmin, grid[0].Nz, direction='z' )
        ir_lower, ir_upper, Sr_lower, Sr_upper, Sr_guard = linear_weights(
            r, grid[0].invdr, grid[0].rmin, grid[0].Nr, direction='r' )

        # Number of modes considered :
        # number of elements in the grid list
        Nm = len(grid)

        if fieldtype == 'rho' :
            # ---------------------------------------
            # Deposit the charge density mode by mode
            # ---------------------------------------
            # Prepare auxiliary matrix
            exptheta = np.ones( self.Ntot, dtype='complex')
            # exptheta takes the value exp(im theta) throughout the loop
            for m in range(Nm) :
                # Increment exptheta (notice the + : forward transform)
                if m==1 :
                    exptheta[:].real = cos
                    exptheta[:].imag = sin
                elif m>1 :
                    exptheta[:] = exptheta*( cos + 1.j*sin )
                # Deposit the fields
                deposit_field( self.w*exptheta, grid[m].rho, 
                    iz_lower, iz_upper, Sz_lower, Sz_upper,
                    ir_lower, ir_upper, Sr_lower, Sr_upper,
                    (-1.)**m, Sr_guard, use_numba )
            
        elif fieldtype == 'J' :
            # ----------------------------------------
            # Deposit the current density mode by mode
            # ----------------------------------------
            # Calculate the currents
            Jr = self.w * c * self.inv_gamma*( cos*self.ux + sin*self.uy )
            Jt = self.w * c * self.inv_gamma*( cos*self.uy - sin*self.ux )
            Jz = self.w * c * self.inv_gamma*self.uz
            # Prepare auxiliary matrix
            exptheta = np.ones( self.Ntot, dtype='complex')
            # exptheta takes the value exp(im theta) throughout the loop
            for m in range(Nm) :
                # Increment exptheta (notice the + : forward transform)
                if m==1 :
                    exptheta[:].real = cos
                    exptheta[:].imag = sin
                elif m>1 :
                    exptheta[:] = exptheta*( cos + 1.j*sin )
                # Deposit the fields
                deposit_field( Jr*exptheta, grid[m].Jr, 
                    iz_lower, iz_upper, Sz_lower, Sz_upper,
                    ir_lower, ir_upper, Sr_lower, Sr_upper,
                    -(-1.)**m, Sr_guard, use_numba )
                deposit_field( Jt*exptheta, grid[m].Jt, 
                    iz_lower, iz_upper, Sz_lower, Sz_upper,
                    ir_lower, ir_upper, Sr_lower, Sr_upper,
                    -(-1.)**m, Sr_guard, use_numba )
                deposit_field( Jz*exptheta, grid[m].Jz, 
                    iz_lower, iz_upper, Sz_lower, Sz_upper,
                    ir_lower, ir_upper, Sr_lower, Sr_upper,
                    (-1.)**m, Sr_guard, use_numba )
        else :
            raise ValueError(
        "`fieldtype` should be either 'J' or 'rho', but is `%s`" %fieldtype )


def linear_weights(x, invdx, offset, Nx, direction) :
    """
    Return the matrix indices and the shape factors, for linear shapes.

    The boundary conditions are determined by direction :
    - direction='z' : periodic conditions
    - direction='r' : absorbing at the upper bound,
                      using guard cells at the lower bounds
    NB : the guard cells are not technically part of the field arrays 
    The weight deposited in the guard cells are added positively or
    negatively to the lower cell of the field array, depending on the
    exact field considered.

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

    direction : string
        Determines the boundary conditions. Either 'r' or 'z'

    Returns
    -------
    A tuple containing :
    
    i_lower, i_upper : 1darray of integers
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
    S_upper : 1darray of floats
    """

    # Positions of the particles, in the cell unit
    x_cell =  invdx*(x - offset)
    
    # Index of the uppper and lower cell
    i_lower = np.floor( x_cell ).astype('int')  
    i_upper = i_lower + 1

    # Linear weight
    S_lower = i_upper - x_cell
    S_upper = x_cell - i_lower
    
    # Treat the boundary conditions
    if direction=='r' :   # Radial boundary condition
        # Lower bound : place the weight in the guard cells
        out_of_bounds =  (i_lower < 0)
        S_guard = np.where( out_of_bounds, S_lower, 0. )
        S_lower = np.where( out_of_bounds, 0., S_lower )
        i_lower = np.where( out_of_bounds, 0, i_lower )
        # Upper bound : absorbing
        i_lower = np.where( i_lower > Nx-1, Nx-1, i_lower )
        i_upper = np.where( i_upper > Nx-1, Nx-1, i_upper )
        # Return the result
        return( i_lower, i_upper, S_lower, S_upper, S_guard )
        
    elif direction=='z' :  # Longitudinal boundary condition
        # Lower bound : periodic
        i_lower = np.where( i_lower < 0, i_lower+Nx, i_lower )
        i_upper = np.where( i_upper < 0, i_upper+Nx, i_upper )
        # Upper bound : periodic
        i_lower = np.where( i_lower > Nx-1, i_lower-Nx, i_lower )
        i_upper = np.where( i_upper > Nx-1, i_upper-Nx, i_upper )
        # Return the result
        return( i_lower, i_upper, S_lower, S_upper )

    else :
        raise ValueError("Unrecognized `direction` : %s" %direction)
        
# -----------------------------------------
# Utility functions for the field gathering
# -----------------------------------------

def gather_field( exptheta, m, Fgrid, Fptcl, 
    iz_lower, iz_upper, Sz_lower, Sz_upper,
    ir_lower, ir_upper, Sr_lower, Sr_upper,
    sign_guards, Sr_guard, use_numba ) :
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
        
    Sz_lower, Sz_upper, Sr_lower, Sr_upper : 1darrays of floats
        (one element per macroparticle)
        Contains the weight for the lower and upper cells.

    sign_guards : float
       The sign (+1 or -1) with which the weight of the guard cells should
       be added to the 0th cell.

    Sr_guard : 1darray of float
        (one element per macroparticle)
        Contains the weight in the guard cells
        
    use_numba : bool
        Whether to use numba rather than numpy for the gathering
    """
    if use_numba == True :
        gather_field_numba( exptheta, m, Fgrid, Fptcl, 
            iz_lower, iz_upper, Sz_lower, Sz_upper,
            ir_lower, ir_upper, Sr_lower, Sr_upper,
            sign_guards, Sr_guard )        
    else :
        gather_field_numpy( exptheta, m, Fgrid, Fptcl, 
            iz_lower, iz_upper, Sz_lower, Sz_upper,
            ir_lower, ir_upper, Sr_lower, Sr_upper,
            sign_guards, Sr_guard )

    
def gather_field_numpy( exptheta, m, Fgrid, Fptcl, 
        iz_lower, iz_upper, Sz_lower, Sz_upper,
        ir_lower, ir_upper, Sr_lower, Sr_upper,
        sign_guards, Sr_guard ) :
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
        
    Sz_lower, Sz_upper, Sr_lower, Sr_upper : 1darrays of floats
        (one element per macroparticle)
        Contains the weight for the lower and upper cells.
        
    sign_guards : float
       The sign (+1 or -1) with which the weight of the guard cells should
       be added to the 0th cell.

    Sr_guard : 1darray of float
        (one element per macroparticle)
        Contains the weight in the guard cells
    """    
    # Temporary matrix that contains the complex fields
    F = np.zeros_like(exptheta)
    
    # Sum the fields from the 4 points
    # Lower cell in z, Lower cell in r
    F += Sz_lower*Sr_lower*Fgrid[ iz_lower, ir_lower ]
    # Lower cell in z, Upper cell in r
    F += Sz_lower*Sr_upper*Fgrid[ iz_lower, ir_upper ]
    # Upper cell in z, Lower cell in r
    F += Sz_upper*Sr_lower*Fgrid[ iz_upper, ir_lower ]
    # Upper cell in z, Upper cell in r
    F += Sz_upper*Sr_upper*Fgrid[ iz_upper, ir_upper ]
    
    # Add the fields from the guard cells
    F += sign_guards * Sz_lower*Sr_guard * Fgrid[ iz_lower, 0]
    F += sign_guards * Sz_upper*Sr_guard * Fgrid[ iz_upper, 0]

    # Add the complex phase
    if m == 0 :
        Fptcl += (F*exptheta).real
    if m > 0 :
        Fptcl += 2*(F*exptheta).real


@numba.jit(nopython=True)
def gather_field_numba( exptheta, m, Fgrid, Fptcl, 
        iz_lower, iz_upper, Sz_lower, Sz_upper,
        ir_lower, ir_upper, Sr_lower, Sr_upper,
        sign_guards, Sr_guard ) :
    """
    Perform the weighted sum using numba

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
        
    Sz_lower, Sz_upper, Sr_lower, Sr_upper : 1darrays of floats
        (one element per macroparticle)
        Contains the weight for the lower and upper cells.
        
    sign_guards : float
       The sign (+1 or -1) with which the weight of the guard cells should
       be added to the 0th cell.

    Sr_guard : 1darray of float
        (one element per macroparticle)
        Contains the weight in the guard cells
    """
    # Get the total number of particles
    Ntot = len(Fptcl)
    
    # Loop over the particles
    for ip in xrange(Ntot) :
        # Erase the temporary variable
        F = 0.j
        # Sum the fields from the 4 points
        # Lower cell in z, Lower cell in r
        F += Sz_lower[ip]*Sr_lower[ip] * Fgrid[ iz_lower[ip], ir_lower[ip] ]
        # Lower cell in z, Upper cell in r
        F += Sz_lower[ip]*Sr_upper[ip] * Fgrid[ iz_lower[ip], ir_upper[ip] ]
        # Upper cell in z, Lower cell in r
        F += Sz_upper[ip]*Sr_lower[ip] * Fgrid[ iz_upper[ip], ir_lower[ip] ]
        # Upper cell in z, Upper cell in r
        F += Sz_upper[ip]*Sr_upper[ip] * Fgrid[ iz_upper[ip], ir_upper[ip] ]

        # Add the fields from the guard cells
        F += sign_guards * Sz_lower[ip]*Sr_guard[ip] * Fgrid[ iz_lower[ip], 0]
        F += sign_guards * Sz_upper[ip]*Sr_guard[ip] * Fgrid[ iz_upper[ip], 0]
        
        # Add the complex phase
        if m == 0 :
            Fptcl[ip] += (F*exptheta[ip]).real
        if m > 0 :
            Fptcl[ip] += 2*(F*exptheta[ip]).real


# -----------------------------------------------
# Utility functions for charge/current deposition
# -----------------------------------------------
        
def deposit_field( Fptcl, Fgrid, 
    iz_lower, iz_upper, Sz_lower, Sz_upper,
    ir_lower, ir_upper, Sr_lower, Sr_upper,
    sign_guards, Sr_guard, use_numba ) :
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
        
    Sz_lower, Sz_upper, Sr_lower, Sr_upper : 1darrays of floats
        (one element per macroparticle)
        Contains the weight for the lower and upper cells.
        
    sign_guards : float
       The sign (+1 or -1) with which the weight of the guard cells should
       be added to the 0th cell.

    Sr_guard : 1darray of float
        (one element per macroparticle)
        Contains the weight in the guard cells
        
    use_numba : bool
        Whether to use numba or numpy.add.at for the deposition
    """
    if use_numba == True :
        deposit_field_numba( Fptcl, Fgrid, 
            iz_lower, iz_upper, Sz_lower, Sz_upper,
            ir_lower, ir_upper, Sr_lower, Sr_upper,
            sign_guards, Sr_guard )
    else :
        deposit_field_numpy( Fptcl, Fgrid, 
            iz_lower, iz_upper, Sz_lower, Sz_upper,
            ir_lower, ir_upper, Sr_lower, Sr_upper,
            sign_guards, Sr_guard )


def deposit_field_numpy( Fptcl, Fgrid, 
        iz_lower, iz_upper, Sz_lower, Sz_upper,
        ir_lower, ir_upper, Sr_lower, Sr_upper,
        sign_guards, Sr_guard ) :
    """
    Perform the deposition using numpy.add.at

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
        
    Sz_lower, Sz_upper, Sr_lower, Sr_upper : 1darrays of floats
        (one element per macroparticle)
        Contains the weight for the lower and upper cells.
        
    sign_guards : float
       The sign (+1 or -1) with which the weight of the guard cells should
       be added to the 0th cell.

    Sr_guard : 1darray of float
        (one element per macroparticle)
        Contains the weight in the guard cells
    """
    # Deposit the particle quantity onto the grid
    # Lower cell in z, Lower cell in r
    np.add.at( Fgrid, (iz_lower, ir_lower), Sz_lower*Sr_lower*Fptcl ) 
    # Lower cell in z, Upper cell in r
    np.add.at( Fgrid, (iz_lower, ir_upper), Sz_lower*Sr_upper*Fptcl )
    # Upper cell in z, Lower cell in r
    np.add.at( Fgrid, (iz_upper, ir_lower), Sz_upper*Sr_lower*Fptcl )
    # Upper cell in z, Upper cell in r
    np.add.at( Fgrid, (iz_upper, ir_upper), Sz_upper*Sr_upper*Fptcl )

    # Add the fields from the guard cells
    np.add.at( Fgrid, (iz_lower, 0), sign_guards*Sz_lower*Sr_guard*Fptcl )
    np.add.at( Fgrid, (iz_upper, 0), sign_guards*Sz_upper*Sr_guard*Fptcl )


@numba.jit(nopython=True)
def deposit_field_numba( Fptcl, Fgrid, 
        iz_lower, iz_upper, Sz_lower, Sz_upper,
        ir_lower, ir_upper, Sr_lower, Sr_upper,
        sign_guards, Sr_guard ) :
    """
    Perform the deposition using numba

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
        
    Sz_lower, Sz_upper, Sr_lower, Sr_upper : 1darrays of floats
        (one element per macroparticle)
        Contains the weight for the lower and upper cells.
        
    sign_guards : float
       The sign (+1 or -1) with which the weight of the guard cells should
       be added to the 0th cell.

    Sr_guard : 1darray of float
        (one element per macroparticle)
        Contains the weight in the guard cells
    """
    # Get the total number of particles
    Ntot = len(Fptcl)
    
    # Deposit the particle quantity onto the grid
    # Lower cell in z, Lower cell in r
    for ip in xrange(Ntot) :
        Fgrid[ iz_lower[ip], ir_lower[ip] ] += \
          Sz_lower[ip] * Sr_lower[ip] * Fptcl[ip]
    # Lower cell in z, Upper cell in r
    for ip in xrange(Ntot) :
        Fgrid[ iz_lower[ip], ir_upper[ip] ] += \
          Sz_lower[ip] * Sr_upper[ip] * Fptcl[ip]
    # Upper cell in z, Lower cell in r
    for ip in xrange(Ntot) :
        Fgrid[ iz_upper[ip], ir_lower[ip] ] += \
          Sz_upper[ip] * Sr_lower[ip] * Fptcl[ip]
    # Upper cell in z, Upper cell in r
    for ip in xrange(Ntot) :
        Fgrid[ iz_upper[ip], ir_upper[ip] ] += \
          Sz_upper[ip] * Sr_upper[ip] * Fptcl[ip]

    # Add the fields from the guard cells in r
    for ip in xrange(Ntot) :
        Fgrid[ iz_lower[ip], 0 ] += \
            sign_guards * Sz_lower[ip]*Sr_guard[ip] * Fptcl[ip]
    for ip in xrange(Ntot) :
        Fgrid[ iz_upper[ip], 0 ] += \
            sign_guards * Sz_upper[ip]*Sr_guard[ip] * Fptcl[ip]

          
# -----------------------
# Particle pusher utility
# -----------------------
          
@numba.jit(nopython=True)
def push_p_numba( ux, uy, uz, inv_gamma, 
                Ex, Ey, Ez, Bx, By, Bz, q, m, Ntot, dt ) :
    """
    Advance the particles' momenta, using numba
    """
    # Set a few constants
    econst = q*dt/(m*c)
    bconst = 0.5*q*dt/m
        
    # Loop over the particles
    for ip in xrange(Ntot) :

        # Shortcut for initial 1./gamma
        inv_gamma_i = inv_gamma[ip]
            
        # Get the magnetic rotation vector
        taux = bconst*Bx[ip]
        tauy = bconst*By[ip]
        tauz = bconst*Bz[ip]
        tau2 = taux**2 + tauy**2 + tauz**2
            
        # Get the momenta at the half timestep
        uxp = ux[ip] + econst*Ex[ip] \
        + inv_gamma_i*( uy[ip]*tauz - uz[ip]*tauy )
        uyp = uy[ip] + econst*Ey[ip] \
        + inv_gamma_i*( uz[ip]*taux - ux[ip]*tauz )
        uzp = uz[ip] + econst*Ez[ip] \
        + inv_gamma_i*( ux[ip]*tauy - uy[ip]*taux )
        sigma = 1 + uxp**2 + uyp**2 + uzp**2 - tau2
        utau = uxp*taux + uyp*tauy + uzp*tauz

        # Get the new 1./gamma
        inv_gamma_f = math.sqrt(
            2./( sigma + np.sqrt( sigma**2 + 4*(tau2 + utau**2 ) ) )
        )
        inv_gamma[ip] = inv_gamma_f

        # Reuse the tau and utau arrays to save memory
        tx = inv_gamma_f*taux
        ty = inv_gamma_f*tauy
        tz = inv_gamma_f*tauz
        ut = inv_gamma_f*utau
        s = 1./( 1 + tau2*inv_gamma_f**2 )

        # Get the new u
        ux[ip] = s*( uxp + tx*ut + uyp*tz - uzp*ty )
        uy[ip] = s*( uyp + ty*ut + uzp*tx - uxp*tz )
        uz[ip] = s*( uzp + tz*ut + uxp*ty - uyp*tx )

# ----------------------------
# Angle initialization utility
# ----------------------------

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
        angle_shift = 2*np.pi*np.random.rand((Npr, Nprz))
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
    
    
