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
    numba_installed = False

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
                    Npr, rmin, rmax, Nptheta, dt ) :
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
        
        # Get the 1d arrays of evenly-spaced positions for the particles
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
            self.push_p_numba()
        else :
            self.push_p_numpy()
    
#    @numba.jit(nopython=True)
    def push_p_numba(self) :
        """
        Advance the particles' momenta, using numba
        """
        # Set a few constants
        econst = self.q*self.dt/(self.m*c)
        bconst = 0.5*self.q*self.dt/self.m
        
        # Loop over the particles
        for ip in xrange(self.Ntot) :

            # Shortcut for initial 1./gamma
            inv_gamma_i = self.inv_gamma[ip]
            
            # Get the magnetic rotation vector
            taux = bconst*self.Bx[ip]
            tauy = bconst*self.By[ip]
            tauz = bconst*self.Bz[ip]
            tau2 = taux**2 + tauy**2 + tauz**2
            
            # Get the momenta at the half timestep
            ux = self.ux[ip] + econst*self.Ex[ip] \
            + inv_gamma_i*( self.uy[ip]*tauz - self.uz[ip]*tauy )
            uy = self.uy[ip] + econst*self.Ey[ip] \
            + inv_gamma_i*( self.uz[ip]*taux - self.ux[ip]*tauz )
            uz = self.uz[ip] + econst*self.Ez[ip] \
            + inv_gamma_i*( self.ux[ip]*tauy - self.uy[ip]*taux )
            sigma = 1 + ux**2 + uy**2 + uz**2 - tau2
            utau = ux*taux + uy*tauy + uz*tauz

            # Get the new 1./gamma
            inv_gamma_f = math.sqrt(
                2./( sigma + np.sqrt( sigma**2 + 4*(tau2 + utau**2 ) ) )
            )
            self.inv_gamma[ip] = inv_gamma_f

            # Reuse the tau and utau arrays to save memory
            tx = inv_gamma_f*taux
            ty = inv_gamma_f*tauy
            tz = inv_gamma_f*tauz
            ut = inv_gamma_f*utau
            s = 1./( 1 + tau2*inv_gamma_f**2 )

            # Get the new u
            self.ux[ip] = s*( ux + tx*ut + uy*tz - uz*ty )
            self.uy[ip] = s*( uy + ty*ut + uz*tx - ux*tz )
            self.uz[ip] = s*( uz + tz*ut + ux*ty - uy*tx )
        

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
        
        
    def gather(self, grid, use_numba = numba_installed) :
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
        c = self.x*invr  # Cosine
        s = self.y*invr  # Sine

        # Indices and weights
        iz_lower, iz_upper, Sz_lower = linear_weights(
           self.z, grid[0].invdz, 0., grid[0].Nz )
        ir_lower, ir_upper, Sr_lower = linear_weights(
            r, grid[0].invdr, 0.5*grid[0].dr, grid[0].Nr )

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
            gather_field( exptheta, m, grid[m].Ez, self.Ez, 
                iz_lower, iz_upper, Sz_lower,
                ir_lower, ir_upper, Sr_lower, use_numba )
            gather_field( exptheta, m, grid[m].Er, Fr, 
                iz_lower, iz_upper, Sz_lower,
                ir_lower, ir_upper, Sr_lower, use_numba )
            gather_field( exptheta, m, grid[m].Et, Ft, 
                iz_lower, iz_upper, Sz_lower,
                ir_lower, ir_upper, Sr_lower, use_numba )
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
                ir_lower, ir_upper, Sr_lower, use_numba )
            gather_field( exptheta, m, grid[m].Br, Fr, 
                iz_lower, iz_upper, Sz_lower,
                ir_lower, ir_upper, Sr_lower, use_numba )
            gather_field( exptheta, m, grid[m].Bt, Ft, 
                iz_lower, iz_upper, Sz_lower,
                ir_lower, ir_upper, Sr_lower, use_numba )
            # Increment exptheta (notice the - : backward Fourier transform)
            exptheta = exptheta*( c - 1.j*s )
        # Convert to Cartesian coordinates
        self.Bx[:] = c*Fr - s*Ft
        self.By[:] = s*Fr + c*Ft

        
    def deposit(self, grid, fieldtype, use_numba = numba_installed ) :
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
        c = self.x*invr  # Cosine
        s = self.y*invr  # Sine

        # Indices and weights
        iz_lower, iz_upper, Sz_lower = linear_weights( 
            self.z, grid[0].invdz, 0., grid[0].Nz )
        ir_lower, ir_upper, Sr_lower = linear_weights(
            r, grid[0].invdr, 0.5*grid[0].dr, grid[0].Nr )

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
                deposit_field( self.w*exptheta, grid[m].rho, 
                    iz_lower, iz_upper, Sz_lower,
                    ir_lower, ir_upper, Sr_lower, use_numba )
                # Increment exptheta (notice the + : forward Fourier transform)
                exptheta = exptheta*( c + 1.j*s )
            
        elif fieldtype == 'J' :
            # ----------------------------------------
            # Deposit the current density mode by mode
            # ----------------------------------------
            # Calculate the currents
            Jr = self.w*self.inv_gamma * ( c*self.ux + s*self.uy )
            Jt = self.w*self.inv_gamma * ( c*self.uy - s*self.ux )
            Jz = self.w*self.inv_gamma * self.uz
            # Prepare auxiliary matrix
            exptheta = np.ones( self.Ntot, dtype='complex')
            # exptheta takes the value exp(-im theta) throughout the loop
            for m in range(Nm) :
                deposit_field( Jr*exptheta, grid[m].Jr, 
                    iz_lower, iz_upper, Sz_lower,
                    ir_lower, ir_upper, Sr_lower, use_numba )
                deposit_field( Jt*exptheta, grid[m].Jt, 
                    iz_lower, iz_upper, Sz_lower,
                    ir_lower, ir_upper, Sr_lower, use_numba )
                deposit_field( Jz*exptheta, grid[m].Jz, 
                    iz_lower, iz_upper, Sz_lower,
                    ir_lower, ir_upper, Sr_lower, use_numba )
                # Increment exptheta (notice the + : forward Fourier transform)
                exptheta = exptheta*( c + 1.j*s )

        else :
            raise ValueError(
        "`fieldtype` should be either 'J' or 'rho', but is `%s`" %fieldtype )


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


# -----------------------------------------
# Utility functions for the field gathering
# -----------------------------------------

def gather_field( exptheta, m, Fgrid, Fptcl, 
    iz_lower, iz_upper, Sz_lower, ir_lower, ir_upper, Sr_lower, use_numba ) :
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

    use_numba : bool
        Whether to use numba rather than numpy for the gathering
    """
    if use_numba == True :
        gather_field_numba( exptheta, m, Fgrid, Fptcl, 
            iz_lower, iz_upper, Sz_lower, ir_lower, ir_upper, Sr_lower )        
    else :
        gather_field_numpy( exptheta, m, Fgrid, Fptcl, 
            iz_lower, iz_upper, Sz_lower, ir_lower, ir_upper, Sr_lower )

    
def gather_field_numpy( exptheta, m, Fgrid, Fptcl, 
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

@numba.jit(nopython=True)
def gather_field_numba( exptheta, m, Fgrid, Fptcl, 
        iz_lower, iz_upper, Sz_lower, ir_lower, ir_upper, Sr_lower ) :
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
        
    Sz_lower, Sr_lower : 1darrays of floats
        (one element per macroparticle)
        Contains the weight for the lower cell, for each macroparticle.
        The weight for the upper cell is just 1-S_lower.
    """
    # Get the total number of particles
    Ntot = len(Fptcl)

    # Loop over the particles
    for ip in xrange(Ntot) :
        # Erase the temporary variable
        F = 0.
        # Sum the fields from the 4 points
        # Lower cell in z, Lower cell in r
        F += Sz_lower[ip]*Sr_lower[ip]*Fgrid[ iz_lower[ip], ir_lower[ip] ]
        # Lower cell in z, Upper cell in r
        F += Sz_lower[ip]*(1-Sr_lower[ip])*Fgrid[ iz_lower[ip], ir_upper[ip] ]
        # Upper cell in z, Lower cell in r
        F += (1-Sz_lower[ip])*Sr_lower[ip]*Fgrid[ iz_upper[ip], ir_lower[ip] ]
        # Upper cell in z, Upper cell in r
        F += (1-Sz_lower[ip])*(1-Sr_lower[ip])*Fgrid[iz_upper[ip],ir_upper[ip]]

    # Add the complex phase
    if m == 0 :
        for ip in xrange(Ntot) :
            Fptcl[ip] += (F*exptheta[ip]).real
    if m > 0 :
        for ip in xrange(Ntot) :
            Fptcl[ip] += 2*(F*exptheta[ip]).real


# -----------------------------------------------
# Utility functions for charge/current deposition
# -----------------------------------------------
        
def deposit_field( Fptcl, Fgrid, 
    iz_lower, iz_upper, Sz_lower, ir_lower, ir_upper, Sr_lower, use_numba ) :
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
        
    use_numba : bool
        Whether to use numba or numpy.add.at for the deposition
    """
    if use_numba == True :
        deposit_field_numba( Fptcl, Fgrid, 
            iz_lower, iz_upper, Sz_lower, ir_lower, ir_upper, Sr_lower )        
    else :
        deposit_field_numpy( Fptcl, Fgrid, 
            iz_lower, iz_upper, Sz_lower, ir_lower, ir_upper, Sr_lower )


def deposit_field_numpy( Fptcl, Fgrid, 
        iz_lower, iz_upper, Sz_lower, ir_lower, ir_upper, Sr_lower ) :
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
    np.add.at( Fgrid, (iz_upper, ir_upper), (1-Sz_lower)*(1-Sr_lower)*Fptcl )


@numba.jit(nopython=True)
def deposit_field_numba( Fptcl, Fgrid, 
        iz_lower, iz_upper, Sz_lower, ir_lower, ir_upper, Sr_lower ) :
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
        
    Sz_lower, Sr_lower : 1darrays of floats
        (one element per macroparticle)
        Contains the weight for the lower cell, for each macroparticle.
        The weight for the upper cell is just 1-S_lower. 
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
          Sz_lower[ip] * (1 - Sr_lower[ip]) * Fptcl[ip]
    # Upper cell in z, Lower cell in r
    for ip in xrange(Ntot) :
        Fgrid[ iz_upper[ip], ir_lower[ip] ] += \
          (1 - Sz_lower[ip]) * Sr_lower[ip] * Fptcl[ip]
    # Upper cell in z, Upper cell in r
    for ip in xrange(Ntot) :
        Fgrid[ iz_upper[ip], ir_upper[ip] ] += \
          (1 - Sz_lower[ip]) * (1 - Sr_lower[ip]) * Fptcl[ip]
    
