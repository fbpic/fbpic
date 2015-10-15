"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to implement the moving window.
"""
import numpy as np
from scipy.constants import c
from .particles import Particles
from .lpa_utils.boosted_frame import BoostConverter

try :
    from numba import cuda
    from fbpic.cuda_utils import cuda_tpb_bpg_2d
    cuda_installed = True
except ImportError :
    cuda_installed = False

class MovingWindow(object) :
    """
    Class that contains the moving window's variables and methods

    One major problem of the moving window in a spectral code is that \
    the fields `wrap around` the moving window, .i.e the fields that
    disappear at the left end reappear at the right end, as a consequence
    of the periodicity of the Fourier transform.

    Attributes
    ----------
    - v : speed of the moving window
    - ncells_zero : number of cells in which the fields are set to zero
    - ncells_damp : number of cells in which the fields are damped

    Methods
    -------
    - move : shift the moving window by v*dt
    - damp : set the fields progressively to zero at the left end of the box
    """
    
    def __init__( self, interp, v=c, ncells_zero=1,
                  ncells_damp=1, period=1, damp_shape='cos',
                  gradual_damp_EB=True, ux_m=0., uy_m=0., uz_m=0.,
                  ux_th=0., uy_th=0., uz_th=0., gamma_boost=None ) :
        """
        Initializes a moving window object.

        Parameters
        ----------
        interp: an InterpolationGrid object
            Needed to obtain the initial position of the moving window
        
        v : float (meters per seconds), optional
            The speed of the moving window
        
        ncells_zero : int, optional
            Number of cells in which the fields are set to zero,
            at the left end of the box, and in which particles are
            suppressed

        ncells_damp : int, optional
            Number of cells over which the currents and density are
            progressively set to 0, at the left end of the box, after
            n_cells_zero.

        period : int, optional
            Number of iterations after which the moving window is moved

        damp_shape : string, optional
            How to damp the fields
            Either 'None', 'linear', 'sin', 'cos'

        gradual_damp_EB : bool, optional
            Whether to gradually damp the fields EB
            If False, no damping at all will be applied to the fields E and B

        ux_m, uy_m, uz_m: floats (dimensionless)
           Normalized mean momenta of the injected particles in each direction

        ux_th, uy_th, uz_th: floats (dimensionless)
           Normalized thermal momenta in each direction

        gamma_boost : float, optional
            When initializing the laser in a boosted frame, set the
            value of `gamma_boost` to the corresponding Lorentz factor.
            (uz_m is to be given in the lab frame ; for the moment, this
            will not work if any of ux_th, uy_th, uz_th, ux_m, uy_m is nonzero)
        """
        # Momenta parameters
        self.ux_m = ux_m
        self.uy_m = uy_m
        self.uz_m = uz_m
        self.ux_th = ux_th
        self.uy_th = uy_th
        self.uz_th = uz_th

        # When running the simulation in boosted frame, convert the arguments
        if gamma_boost is not None:
            boost = BoostConverter( gamma_boost )
            self.uz_m, = boost.longitudinal_momentum([ self.uz_m ])
        
        # Attach positions and speed
        # - Moving window speed
        self.v = v
        # - Initial position of the moving window (move with moving window)
        self.zmin = interp.zmin
        # - Initial injection position (move with moving window)
        #   (particles are not loaded in the last two upper cells)
        self.z_inject = interp.zmax - 2 * interp.dz
        # - Mean speed of the end of the plasma
        self.v_end_plasma = c * uz_m / np.sqrt(1 + ux_m**2 + uy_m**2 + uz_m**2)
        # - Position of the right end of the plasma (moves with the plasma)
        self.z_end_plasma = interp.zmax - 2 * interp.dz

        # Verify parameters, to prevent wrapping around of the particles
        if ncells_zero < period :
            raise ValueError('`ncells_zero` should be more than `period`')
        
        # Attach numerical parameters
        self.ncells_zero = ncells_zero
        self.ncells_damp = ncells_damp
        self.damp_shape = damp_shape
        self.period = period
        
        # Create the damping array for the density and currents
        if damp_shape == 'None' :
            self.damp_array_J = np.ones(ncells_damp)
        elif damp_shape == 'linear' :
            self.damp_array_J = np.linspace(0, 1, ncells_damp)
        elif damp_shape == 'sin' :
            self.damp_array_J = np.sin(np.linspace(0, np.pi/2, ncells_damp) )
        elif damp_shape == 'cos' :
            self.damp_array_J = 0.5-0.5*np.cos(
                np.linspace(0, np.pi, ncells_damp) )
        else :
            raise ValueError("Invalid string for damp_shape : %s"%damp_shape)

        # Create the damping array for the E and B fields
        self.damp_array_EB = np.ones(ncells_damp)
        if gradual_damp_EB :
            # Contrary to the fields rho and J which are recalculated
            # at each timestep, the fields E and B accumulate damping
            # over the successive timesteps. Therefore, the damping on
            # E and B should be lighter. The following formula ensures
            # (for a static field) that the successive applications of
            # damping result in the same damping shape as for J.
            self.damp_array_EB[:-1] = \
              self.damp_array_J[:-1]/self.damp_array_J[1:]
        # Copy the array to the GPU if possible
        if cuda_installed :
            self.d_damp_array_EB = cuda.to_device(self.damp_array_EB)
        
    def move( self, interp, ptcl, p_nz, dt ) :
        """
        Calculate by how many cells the moving window should be moved.
        If this is non-zero, shift the fields on the interpolation grid,
        and add new particles.

        NB : the spectral grid is not modified, as it is automatically
        updated after damping the fields (see main.py)
        
        Parameters
        ----------
        interp : a list of InterpolationGrid objects
            (one element per azimuthal mode)
            Contains the fields data of the simulation
    
        ptcl : a list of Particles objects
            Contains the particles data for each species
    
        p_nz : int
            Number of macroparticles per cell along the z direction
    
        dt : float (in seconds)
            Timestep of the simulation
        """
        # Move the position of the moving window object
        # and the position of injection
        self.zmin += self.v * dt * self.period
        self.z_inject += self.v * dt * self.period
        # Take into account the motion of the end of the plasma
        self.z_end_plasma += self.v_end_plasma * dt * self.period

        # Find the number of cells by which the window should move
        dz = interp[0].dz
        n_move = int( (self.zmin - interp[0].zmin)/dz )
        # Move the window
        if n_move > 0 :
            
            # Shift the fields
            Nm = len(interp)
            for m in range(Nm) :
                self.shift_interp_grid( interp[m], n_move )
        
            # Extract a few quantities of the new (shifted) grid
            zmin = interp[0].zmin
            zmax = interp[0].zmax

            # Determine the position below which the particles
            # should be removed
            z_zero = zmin + self.ncells_zero*dz
    
            # Now that the grid has moved, remove the particles that exited
            for species in ptcl :
                clean_outside_particles( species, z_zero )

        # Find the number of particle cells to add
        n_inject = int( (self.z_inject - self.z_end_plasma)/dz )
        # Add the new particle cells
        if n_inject > 0 :
            for species in ptcl :
                if species.continuous_injection == True :
                    add_particles( species, self.z_end_plasma,
                        self.z_end_plasma + n_inject*dz, n_inject*p_nz,
                        ux_m=self.ux_m, uy_m=self.uy_m, uz_m=self.uz_m,
                        ux_th=self.ux_th, uy_th=self.uy_th, uz_th=self.uz_th)
            self.z_end_plasma += n_inject*dz
                                
    def shift_interp_grid( self, grid, n_move, shift_currents=False ) :
        """
        Shift the interpolation grid by one cell
    
        Parameters
        ----------
        grid : an InterpolationGrid corresponding to one given azimuthal mode 
            Contains the values of the fields on the interpolation grid,
            and is modified by this function.

        n_move : int
            The number of cells by which the grid should be shifted

        shift_currents : bool, optional
            Whether to also shift the currents
            Default : False, since the currents are recalculated from
            scratch at each PIC cycle
        """
        # Modify the values of the corresponding z's 
        grid.z += n_move*grid.dz
        grid.zmin += n_move*grid.dz
        grid.zmax += n_move*grid.dz
    
        # Shift all the fields
        self.shift_interp_field( grid.Er, n_move )
        self.shift_interp_field( grid.Et, n_move )
        self.shift_interp_field( grid.Ez, n_move )
        self.shift_interp_field( grid.Br, n_move )
        self.shift_interp_field( grid.Bt, n_move )
        self.shift_interp_field( grid.Bz, n_move )
        if shift_currents :
            self.shift_interp_field( grid.Jr, n_move )
            self.shift_interp_field( grid.Jt, n_move )
            self.shift_interp_field( grid.Jz, n_move )
            self.shift_interp_field( grid.rho, n_move )

    def shift_interp_field( self, field_array, n_move ) :
        """
        Shift the field 'field_array' by one cell (backwards)
        
        Parameters
        ----------
        field_array : 2darray of complexs
            Contains the value of the fields, and is modified by
            this function

        n_move : int
            The number of cells by which the grid should be shifted
        """
        # Transfer the values to n_move cell before
        field_array[:-n_move,:] = field_array[n_move:,:]
        # Put the last cells to 0
        field_array[-n_move,:] = 0        

    def damp_EB( self, interp ) :
        """
        Set the fields E and B progressively to zero, at the left
        end and right end of the moving window.

        This is done by multiplying the first and last cells
        of the field array (along z) by self.damp_array_EB

        Parameters
        ----------
        interp : a list of InterpolationGrid objects
            (one element per azimuthal mode)
            Contains the field data on the interpolation grid
        """
        if interp[0].use_cuda :
            # Obtain cuda grid (the points beyond Nzeff are not modified)
            Nz_eff = self.ncells_zero + self.ncells_damp
            Nr = interp[0].Nr
            dim_grid, dim_block = cuda_tpb_bpg_2d( Nz_eff, Nr )
            
            # Damp the fields on the GPU
            cuda_damp_EB[dim_grid, dim_block](
                interp[0].Er, interp[0].Et, interp[0].Ez,
                interp[0].Br, interp[0].Bt, interp[0].Bz,
                interp[1].Er, interp[1].Et, interp[1].Ez,
                interp[1].Br, interp[1].Bt, interp[1].Bz,
                self.d_damp_array_EB, self.ncells_zero, self.ncells_damp )
        else :
            # Damp the fields on the CPU
            
            # Extract the length of the grid
            Nm = len(interp)

            # Loop over the azimuthal modes
            for m in range( Nm ) :
                damp_field( interp[m].Er, self.damp_array_EB,
                            self.ncells_damp, self.ncells_zero )
                damp_field( interp[m].Et, self.damp_array_EB,
                            self.ncells_damp, self.ncells_zero )
                damp_field( interp[m].Ez, self.damp_array_EB,
                            self.ncells_damp, self.ncells_zero )
                damp_field( interp[m].Br, self.damp_array_EB,
                            self.ncells_damp, self.ncells_zero )
                damp_field( interp[m].Bt, self.damp_array_EB,
                            self.ncells_damp, self.ncells_zero )
                damp_field( interp[m].Bz, self.damp_array_EB,
                            self.ncells_damp, self.ncells_zero )
            
# ---------------------------------------
# Utility functions for the moving window
# ---------------------------------------

def clean_outside_particles( species, zmin ) :
    """
    Removes the particles that are below `zmin`.

    Parameters
    ----------
    species : a Particles object
        Contains the data of this species

    zmin : float
        The lower bound under which particles are removed
    """

    # Select the particles that are still inside the box
    selec = ( species.z > zmin )

    # Keep only this selection, in the different arrays that contains the
    # particle properties (x, y, z, ux, uy, uz, etc...)
    # Instead of hard-coding x = x[selec], y=y[selec], etc... here we loop
    # over the particles attributes, and resize the attributes that are
    # arrays with one element per particles.
    # The advantage is that nothing needs to be added to this piece of code,
    # if a new particle attribute is later added in particles.py.

    # Loop over the attributes
    for key, attribute in vars(species).items() :
        # Detect if it is an array
        if type(attribute) is np.ndarray :
            # Detect if it has one element per particle
            if attribute.shape == ( species.Ntot ,) :
                # Affect the resized array to the object
                setattr( species, key, attribute[selec] )

    # Adapt the number of particles accordingly
    species.Ntot = len( species.w )

def add_particles( species, zmin, zmax, Npz, ux_m=0., uy_m=0., uz_m=0.,
                  ux_th=0., uy_th=0., uz_th=0. ) :
    """
    Create new particles between zmin and zmax, and add them to `species`

    Parameters
    ----------
    species : a Particles object
       Contains the particle data of that species

    zmin, zmax : floats (meters)
       The positions between which the new particles are created

    Npz : int
        The total number of particles to be added along the z axis
        (The number of particles along r and theta is the same as that of
        `species`)

    ux_m, uy_m, uz_m: floats (dimensionless)
        Normalized mean momenta of the injected particles in each direction

    ux_th, uy_th, uz_th: floats (dimensionless)
        Normalized thermal momenta in each direction     
    """
    # Create the particles that will be added
    new_ptcl = Particles( species.q, species.m, species.n,
        Npz, zmin, zmax, species.Npr, species.rmin, species.rmax,
        species.Nptheta, species.dt, species.dens_func,
        ux_m=ux_m, uy_m=uy_m, uz_m=uz_m,
        ux_th=ux_th, uy_th=uy_th, uz_th=uz_th )

    # Add the properties of these new particles to species object
    # Loop over the attributes of the species
    for key, attribute in vars(species).items() :
        # Detect if it is an array
        if type(attribute) is np.ndarray :
            # Detect if it has one element per particle
            if attribute.shape == ( species.Ntot ,) :
                # Concatenate the attribute of species and of new_ptcl
                new_attribute = np.hstack(
                    ( getattr(species, key), getattr(new_ptcl, key) )  )
                # Affect the resized array to the species object
                setattr( species, key, new_attribute )

    # Add the number of new particles to the global count of particles
    species.Ntot += new_ptcl.Ntot
    
def damp_field( field_array, damp_array, n_damp, n_zero ) :
    """
    Put the fields to 0 in the n_zero first cells
    Multiply the fields by damp_array_EB in the n_damp next cells
    (at the left and right boundary)

    Parameters
    ----------
    field_array : 2darray of complexs
    damp_array : 2darray of reals
    n_damp, n_zero : int
    """
    field_array[:n_zero,:] = 0
    field_array[-n_zero:,:] = 0
    field_array[n_zero:n_zero+n_damp,:] = \
        damp_array[:,np.newaxis] * field_array[n_zero:n_zero+n_damp,:]
    field_array[-n_zero-n_damp:-n_zero,:] = \
        damp_array[::-1,np.newaxis] * field_array[-n_zero-n_damp:-n_zero,:]

if cuda_installed :

    @cuda.jit('void(complex128[:,:], complex128[:,:], complex128[:,:], \
                    complex128[:,:], complex128[:,:], complex128[:,:], \
                    complex128[:,:], complex128[:,:], complex128[:,:], \
                    complex128[:,:], complex128[:,:], complex128[:,:], \
                    float64[:], int32, int32)')
    def cuda_damp_EB( Er0, Et0, Ez0, Br0, Bt0, Bz0,
                      Er1, Et1, Ez1, Br1, Bt1, Bz1,
                      damp_array_EB, ncells_zero, ncells_damp ) :
        """
        Put the fields to 0 in the ncells_zero first cells
        Multiply the fields by damp_array_EB in the next cells
        (at the left and right boundary)

        Parameters :
        ------------
        Er0, Et0, Ez0, Br0, Bt0, Bz0, 
        Er1, Et1, Ez1, Br1, Bt1, Bz1 : 2darrays of complexs
            Contain the fields to be damped
            The first axis corresponds to z and the second to r

        damp_array_EB : 1darray of floats
            An array of length ncells_damp
            Contains the values of the damping factor
        """
        # Obtain Cuda grid
        iz, ir = cuda.grid(2)

        # Obtain the size of the array along z and r
        Nz, Nr = Er0.shape
        
        # Modify the fields
        if ir < Nr :
            if iz < ncells_zero :
                # At the left end
                Er0[iz, ir] = 0.
                Et0[iz, ir] = 0.
                Ez0[iz, ir] = 0.
                Br0[iz, ir] = 0.
                Bt0[iz, ir] = 0.
                Bz0[iz, ir] = 0.
                Er1[iz, ir] = 0.
                Et1[iz, ir] = 0.
                Ez1[iz, ir] = 0.
                Br1[iz, ir] = 0.
                Bt1[iz, ir] = 0.
                Bz1[iz, ir] = 0.
                # At the right end
                iz_right = Nz - iz - 1
                Er0[iz_right, ir] = 0.
                Et0[iz_right, ir] = 0.
                Ez0[iz_right, ir] = 0.
                Br0[iz_right, ir] = 0.
                Bt0[iz_right, ir] = 0.
                Bz0[iz_right, ir] = 0.
                Er1[iz_right, ir] = 0.
                Et1[iz_right, ir] = 0.
                Ez1[iz_right, ir] = 0.
                Br1[iz_right, ir] = 0.
                Bt1[iz_right, ir] = 0.
                Bz1[iz_right, ir] = 0.
            elif iz < ncells_zero + ncells_damp :
                damp_factor = damp_array_EB[iz - ncells_zero]
                # At the left end
                Er0[iz, ir] *= damp_factor
                Et0[iz, ir] *= damp_factor
                Ez0[iz, ir] *= damp_factor
                Br0[iz, ir] *= damp_factor
                Bt0[iz, ir] *= damp_factor
                Bz0[iz, ir] *= damp_factor
                Er1[iz, ir] *= damp_factor
                Et1[iz, ir] *= damp_factor
                Ez1[iz, ir] *= damp_factor
                Br1[iz, ir] *= damp_factor
                Bt1[iz, ir] *= damp_factor
                Bz1[iz, ir] *= damp_factor
                # At the right end
                iz_right = Nz - iz - 1
                Er0[iz_right, ir] *= damp_factor
                Et0[iz_right, ir] *= damp_factor
                Ez0[iz_right, ir] *= damp_factor
                Br0[iz_right, ir] *= damp_factor
                Bt0[iz_right, ir] *= damp_factor
                Bz0[iz_right, ir] *= damp_factor
                Er1[iz_right, ir] *= damp_factor
                Et1[iz_right, ir] *= damp_factor
                Ez1[iz_right, ir] *= damp_factor
                Br1[iz_right, ir] *= damp_factor
                Bt1[iz_right, ir] *= damp_factor
                Bz1[iz_right, ir] *= damp_factor
