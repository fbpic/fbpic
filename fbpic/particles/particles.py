"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure and methods associated with the particles.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

# If numba is installed, it can make the code much faster
try :
    from numba_methods import *
except ImportError :
    numba_installed = False
else :
    numba_installed = True

# If numbapro is installed, it can
try :
    from cuda_methods import *
except :
    cuda_installed = False
else :
    cuda_installed = True
    tbp = 256 # Threads per block

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
                    Npr, rmin, rmax, Nptheta, dt,
                    dens_func=None, global_theta=0.,
                    continuous_injection=True,
                    use_cuda=False, use_numba=True ) :
        """
        Initialize a uniform set of particles

        Parameters
        ----------
        q : float (in Coulombs)
           Charge of the particle species 

        m : float (in kg)
           Mass of the particle species 

        n : float (in particles per m^3)
           Peak density of particles
           
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

        dens_func : callable, optional
           A function of the form :
           def dens_func( z, r ) ...
           where z and r are 1d arrays, and which returns
           a 1d array containing the density *relative to n*
           (i.e. a number between 0 and 1) at the given positions

        global_theta : float (in rad), optional
           A global shift on all the theta of the particles
           This is useful when repetitively adding new particles
           (e.g. with the moving window), in order to avoid that
           the successively-added particles are aligned.

        continuous_injection : bool, optional
           Whether to continuously inject the particles,
           in the case of a moving window

        use_numba : bool, optional
            Whether to use numba-compiled code on the CPU
           
        use_gpu : bool, optional
            Wether to use the GPU or not. Overrides use_numba.
        """
        # Register the timestep
        self.dt = dt

        # Define wether or not to use the GPU
        self.use_cuda = use_cuda
        if (self.use_cuda==True) and (cuda_installed==False) :
            print 'Cuda for numba is not installed ; running on the CPU.'
            self.use_cuda = False
        # Define whether or not to use numba on a CPU
        self.use_numba = use_numba
        if self.use_cuda == True :
            self.use_numba = False
        if (self.use_numba==True) and (numba_installed==False) :
            print 'Numba is not installed ; the code will be slow.'
        
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
        self.dens_func = dens_func
        self.continuous_injection = continuous_injection

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
        r = rp.flatten()
        self.x = r * np.cos( thetap.flatten() )
        self.y = r * np.sin( thetap.flatten() )
        self.z = zp.flatten()

        # Get the weights (i.e. charge of each macroparticle), which are equal
        # to the density times the elementary volume r d\theta dr dz
        self.w = q * n * r * dtheta*dr*dz
        # Modulate it by the density profile
        if dens_func is not None :
            self.w = self.w * dens_func( self.z, r )
        
    def push_p( self ) :
        """
        Advance the particles' momenta over one timestep, using the Vay pusher
        Reference : Vay, Physics of Plasmas 15, 056701 (2008)

        This assumes that the momenta (ux, uy, uz) are initially one
        half-timestep *behind* the positions (x, y, z), and it brings
        them one half-timestep *ahead* of the positions.
        """
        if self.use_numba :
            push_p_numba(self.ux, self.uy, self.uz, self.inv_gamma, 
                    self.Ex, self.Ey, self.Ez, self.Bx, self.By, self.Bz,
                    self.q, self.m, self.Ntot, self.dt )
        elif self.use_cuda :
            bpg = int(self.Ntot/tpb + 1)
            push_p_cuda[bpg, tpb](self.ux, self.uy, self.uz,
                    self.inv_gamma, self.Ex, self.Ey, self.Ez,
                    self.Bx, self.By, self.Bz,
                    self.q, self.m, self.Ntot, self.dt )
        else :
            self.push_p_numpy()

    def push_p_numpy(self) :
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
        
    def gather( self, grid ) :
        """
        Gather the fields onto the macroparticles using numpy

        This assumes that the particle positions are currently at
        the same timestep as the field that is to be gathered.
        
        Parameter
        ----------
        grid : a list of InterpolationGrid objects
             (one InterpolationGrid object per azimuthal mode)
             Contains the field values on the interpolation grid
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
                -(-1.)**m, Sr_guard, self.use_numba, self.use_cuda )
            gather_field( exptheta, m, grid[m].Et, Ft, 
                iz_lower, iz_upper, Sz_lower, Sz_upper,
                ir_lower, ir_upper, Sr_lower, Sr_upper,
                -(-1.)**m, Sr_guard, self.use_numba, self.use_cuda )
            gather_field( exptheta, m, grid[m].Ez, self.Ez, 
                iz_lower, iz_upper, Sz_lower, Sz_upper,
                ir_lower, ir_upper, Sr_lower, Sr_upper,
                (-1.)**m, Sr_guard, self.use_numba, self.use_cuda )

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
                -(-1.)**m, Sr_guard, self.use_numba, self.use_cuda )
            gather_field( exptheta, m, grid[m].Bt, Ft, 
                iz_lower, iz_upper, Sz_lower, Sz_upper,
                ir_lower, ir_upper, Sr_lower, Sr_upper,
                -(-1.)**m, Sr_guard, self.use_numba, self.use_cuda )
            gather_field( exptheta, m, grid[m].Bz, self.Bz, 
                iz_lower, iz_upper, Sz_lower, Sz_upper,
                ir_lower, ir_upper, Sr_lower, Sr_upper,
                (-1.)**m, Sr_guard, self.use_numba, self.use_cuda )

        # Convert to Cartesian coordinates
        self.Bx[:] = cos*Fr - sin*Ft
        self.By[:] = sin*Fr + cos*Ft

        
    def deposit(self, grid, fieldtype ) :
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
                # (The sign -1 with which the guards are added
                # is not trivial to derive but avoids artifacts on the axis)
                deposit_field( self.w*exptheta, grid[m].rho, 
                    iz_lower, iz_upper, Sz_lower, Sz_upper,
                    ir_lower, ir_upper, Sr_lower, Sr_upper,
                    -1., Sr_guard, self.use_numba, self.use_cuda )
            
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
                # (The sign -1 with which the guards are added
                # is not trivial to derive but avoids artifacts on the axis)
                deposit_field( Jr*exptheta, grid[m].Jr, 
                    iz_lower, iz_upper, Sz_lower, Sz_upper,
                    ir_lower, ir_upper, Sr_lower, Sr_upper,
                    -1., Sr_guard, self.use_numba, self.use_cuda )
                deposit_field( Jt*exptheta, grid[m].Jt, 
                    iz_lower, iz_upper, Sz_lower, Sz_upper,
                    ir_lower, ir_upper, Sr_lower, Sr_upper,
                    -1., Sr_guard, self.use_numba, self.use_cuda )
                deposit_field( Jz*exptheta, grid[m].Jz, 
                    iz_lower, iz_upper, Sz_lower, Sz_upper,
                    ir_lower, ir_upper, Sr_lower, Sr_upper,
                    -1., Sr_guard, self.use_numba, self.use_cuda )
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
    sign_guards, Sr_guard, use_numba, use_cuda ) :
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
        Whether to use numba on the CPU for the gathering

    use_cuda : bool
        Whether to use cuda on the GPU for the gathering
    """
    if use_numba :
        gather_field_numba( exptheta, m, Fgrid, Fptcl, 
                iz_lower, iz_upper, Sz_lower, Sz_upper,
                ir_lower, ir_upper, Sr_lower, Sr_upper,
                sign_guards, Sr_guard )
    elif use_cuda :
        Ntot = len(Fptcl)
        bpg = int(Ntot/tpb + 1)
        gather_field_cuda[bpg, tpb](exptheta, m, Fgrid, Fptcl, 
                iz_lower, iz_upper, Sz_lower, Sz_upper,
                ir_lower, ir_upper, Sr_lower, Sr_upper,
                sign_guards, Sr_guard)       
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


# -----------------------------------------------
# Utility functions for charge/current deposition
# -----------------------------------------------
        
def deposit_field( Fptcl, Fgrid, 
    iz_lower, iz_upper, Sz_lower, Sz_upper,
    ir_lower, ir_upper, Sr_lower, Sr_upper,
    sign_guards, Sr_guard, use_numba, use_cuda ) :
    """
    Perform the deposition on the 4 points that surround each particle,
    for one given field and one given azimuthal mode

    GPU Version uses atomic add to deposit the field to the grid in parallel.

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

    use_cuda : bool
        Whether to use cuda on the GPU for the particle deposition
    """
    if use_numba :
        deposit_field_numba( Fptcl, Fgrid, 
                iz_lower, iz_upper, Sz_lower, Sz_upper,
                ir_lower, ir_upper, Sr_lower, Sr_upper,
                sign_guards, Sr_guard )
    elif use_cuda :
        # sign_guards needs to be an array for sending to device
        sign_guards = np.array([sign_guards])
        # Send arrays to the GPU
        d_Fptcl = cuda.to_device(Fptcl)
        d_Fgrid = cuda.to_device(Fgrid)
        d_iz_lower = cuda.to_device(iz_lower)
        d_iz_upper = cuda.to_device(iz_upper)
        d_Sz_lower = cuda.to_device(Sz_lower)
        d_Sz_upper = cuda.to_device(Sz_upper)
        d_ir_lower = cuda.to_device(ir_lower)
        d_ir_upper = cuda.to_device(ir_upper)
        d_Sr_lower = cuda.to_device(Sr_lower)
        d_Sr_upper = cuda.to_device(Sr_upper)
        d_sign_guards = cuda.to_device(sign_guards)
        d_Sr_guard = cuda.to_device(Sr_guard)

        Ntot = len(Fptcl)
        nx, ny = Fgrid.shape
        # Define threads per block for 1d and 2d grid
        tpb1d = 256
        tpb2dx = 32
        tpb2dy = 8
        # Calculate blocks per grid for 1d and 2d grid
        bpg1d = int(Ntot / tpb1d + 1)
        bpg2dx = int(nx / tpb2dx + 1)
        bpg2dy = int(ny / tpb2dy + 1)
        # Initialize GPU arrays for real and imaginary part 
        # of Fgrid and Fptcl
        d_Fgrid_real = cuda.device_array(Fgrid.shape, dtype = np.float64)
        d_Fgrid_imag = cuda.device_array(Fgrid.shape, dtype = np.float64)
        d_Fptcl_real = cuda.device_array(Fptcl.shape, dtype = np.float64)
        d_Fptcl_imag = cuda.device_array(Fptcl.shape, dtype = np.float64)
        # Split complex128 array into two float64 arrays 
        # for Fgrid and Fptcl
        split_complex2d[(bpg2dx, bpg2dy), (tpb2dx, tpb2dy)](d_Fgrid, 
                                    d_Fgrid_real, d_Fgrid_imag)
        split_complex1d[bpg1d, tpb1d](d_Fptcl, d_Fptcl_real, d_Fptcl_imag)
        # Deposit the real part of the field using cuda.atomic.add
        deposit_field_cuda[bpg1d, tpb1d]( d_Fptcl_real, d_Fgrid_real, 
                d_iz_lower, d_iz_upper, d_Sz_lower, d_Sz_upper,
                d_ir_lower, d_ir_upper, d_Sr_lower, d_Sr_upper,
                d_sign_guards, d_Sr_guard )
        # Deposit the imaginary part of the field using cuda.atomic.add
        deposit_field_cuda[bpg1d, tpb1d]( d_Fptcl_imag, d_Fgrid_imag, 
                d_iz_lower, d_iz_upper, d_Sz_lower, d_Sz_upper,
                d_ir_lower, d_ir_upper, d_Sr_lower, d_Sr_upper,
                d_sign_guards, d_Sr_guard )
        # Merge the two float64 arrays into a single complex128 array 
        # for Fgrid and Fptcl
        merge_complex2d[(bpg2dx, bpg2dy), (tpb2dx, tpb2dy)](d_Fgrid_real,
                                                d_Fgrid_imag, d_Fgrid)
        merge_complex1d[bpg1d, tpb1d](d_Fptcl_real, d_Fptcl_imag, d_Fptcl)
        #Copy the GPU arrays from device to host
        d_Fptcl.copy_to_host(Fptcl)
        d_Fgrid.copy_to_host(Fgrid)
        d_iz_lower.copy_to_host(iz_lower)
        d_iz_upper.copy_to_host(iz_upper)
        d_Sz_lower.copy_to_host(Sz_lower)
        d_Sz_upper.copy_to_host(Sz_upper)
        d_ir_lower.copy_to_host(ir_lower)
        d_ir_upper.copy_to_host(ir_upper)
        d_Sr_lower.copy_to_host(Sr_lower)
        d_sign_guards.copy_to_host(sign_guards)
        d_Sr_guard.copy_to_host(Sr_guard)
    else :
        deposit_field_numpy( Fptcl, Fgrid, 
            iz_lower, iz_upper, Sz_lower, Sz_upper,
            ir_lower, ir_upper, Sr_lower, Sr_upper,
            sign_guards, Sr_guard )

def deposit_field_sorting( Fptcl, Fgrid, 
    iz_lower, iz_upper, Sz_lower, Sz_upper,
    ir_lower, ir_upper, Sr_lower, Sr_upper,
    sign_guards, Sr_guard, use_numba, use_cuda ) :

    """
    Perform the deposition on the 4 points that surround each particle,
    for one given field and one given azimuthal mode

    GPU Version sorts the particles per cell and iterates over all
    particles per cell in parallel avoiding the use of atomic operations.

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

    use_cuda : bool
        Whether to use cuda on the GPU for the particle deposition
    """
    if use_numba :
        deposit_field_numba( Fptcl, Fgrid, 
                iz_lower, iz_upper, Sz_lower, Sz_upper,
                ir_lower, ir_upper, Sr_lower, Sr_upper,
                sign_guards, Sr_guard )
    elif use_cuda :
        # sign_guards needs to be transformed to an array to comply
        # with the definition of the deposition functions
        sign_guards = np.array([sign_guards])

        Ntot = len(Fptcl)
        nx, ny = Fgrid.shape
        # Create empty arrays to store the cell index, the sorted
        # index of the particles, the frequency per cell (number of
        # particles per cell) and the prefix sum (indicates at wich
        # position a new value occurs in the cell index array)
        cell_idx = np.arange(Ntot, dtype = np.int32)
        sorted_idx = np.arange(Ntot, dtype = np.uint32)
        frequency_per_cell = np.zeros(nx*ny, dtype = np.int32)
        prefix_sum = np.zeros(nx*ny, dtype = np.float64)
        # Create empty arrays to store the four different possible
        # cell directions a particle can deposit to
        Fgrid_per_node0 = np.empty((nx, ny), dtype = np.complex128)
        Fgrid_per_node1 = np.empty((nx, ny), dtype = np.complex128)
        Fgrid_per_node2 = np.empty((nx, ny), dtype = np.complex128)
        Fgrid_per_node3 = np.empty((nx, ny), dtype = np.complex128)
        # Define the threads per block for 1D and 2D grid
        tpb1d = 256
        tpb2dx = 8
        tpb2dy = 4
        # Calucalte the Blocks per grid for 1D and 2D grid
        bpg1d = int(Ntot / tpb1d + 1) 
        bpg1d_grid = int((nx*ny) / tpb1d + 1)
        bpg2dx = int(nx / tpb2dx + 1)
        bpg2dy = int(ny / tpb2dy + 1)
        # Get the cell index of each particle 
        # (defined by iz_lower and ir_lower)
        get_cell_idx_per_particle[bpg1d, tpb1d](cell_idx, 
                iz_lower, ir_lower, nx, ny)
        # Sort the cell index array and modify the sorted_idx array
        # accordingly. The value of the sorted_idx array corresponds 
        # to the index of the sorted particle in the other particle 
        # arrays.
        sort_particles_per_cell(cell_idx, sorted_idx)
        # Count the particles per cell 
        # (This is not really needed in the future)
        count_particles_per_cell[bpg1d, tpb1d](cell_idx, frequency_per_cell)
        # Perform the inclusive parallel prefix sum (slow)
        incl_prefix_sum[bpg1d, tpb1d](cell_idx, prefix_sum)
        # Deposit the field per cell in parallel to the four 
        # field arrays
        deposit_per_cell[bpg1d_grid, tpb1d](Fptcl, 
                    Fgrid_per_node0, Fgrid_per_node1, 
                    Fgrid_per_node2, Fgrid_per_node3,
                    Sz_lower, Sz_upper, Sr_lower, Sr_upper,
                    sign_guards, Sr_guard, cell_idx, frequency_per_cell, 
                    prefix_sum, sorted_idx)
        # Merge the four field arrays by adding them in parallel
        add_field[(bpg2dx, bpg2dy), (tpb2dx, tpb2dy)](Fgrid, 
                Fgrid_per_node0, Fgrid_per_node1,
                Fgrid_per_node2, Fgrid_per_node3)
    else:
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
    
    
