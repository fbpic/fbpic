"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to implement the moving window.
"""
import numpy as np
from scipy.constants import c
from particles import Particles

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
    
    def __init__( self, zmin=0, v=c, ncells_zero=1, ncells_damp=1,
                  period=1, damp_shape='cos', gradual_damp_EB=True ) :
        """
        Initializes a moving window object.

        Parameters
        ----------
        zmin : float (meters), optional
            The starting position of the moving window
        
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
        """
        # Attach position and speed
        self.zmin = zmin
        self.v = v

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
        
    def move( self, interp, ptcl, p_nz, dt, comm=None ) :
        """
        Calculate by how many cells the moving window should be moved.
        If this is non-zero, shift the fields on the interpolation grid,
        and add new particles.

        NB : the spectral grid is not modified, as it is automatically
        updated after damping the fields (see main.py)
        
        Parameters
        ----------
        interp : a list of InterpolationGrid objects
            Contains the fields data of the simulation
    
        ptcl : a list of Particles objects
            Contains the particles data for each species
    
        p_nz : int
            Number of macroparticles per cell along the z direction
    
        dt : float (in seconds)
            Timestep of the simulation

        comm : a Communicator object
            Defines how to send fields and particles with MPI
            When not using MPI, this object is None
        """
        # Move the position of the moving window object
        self.zmin = self.zmin + self.v*dt*self.period
        
        # Find the number of cells by which the window should move
        dz = interp[0].dz
        zmin_global = interp[0].zmin \
          + dz*(comm.n_guard - comm.rank*comm.Nz_domain)
        n_move = int( (self.zmin-zmin_global)/dz )

        # Move the window
        if n_move > 0 :
            
            # Exchange the paticles, when using MPI
            if comm is not None :
                # Exchange only if this was not done previously in main.py
                if self.period % comm.exchange_part_period != 0 :
                    for species in ptcl:
                        comm.exchange_particles( species,
                            interp[0].zmin, interp[0].zmax )
            
            # Shift the fields
            Nm = len(interp)
            for m in range(Nm) :
                self.shift_interp_grid( interp[m], n_move )
        
            # Extract a few quantities of the new (shifted) grid
            zmin = interp[0].zmin
            zmax = interp[0].zmax

            # The first proc removes the particles that are outside of the box
            if (comm is None) or (comm.rank == 0) :
                # Determine the position below which the particles are removed
                z_zero = zmin + self.ncells_zero*dz
                if comm is not None :
                    z_zero = z_zero + comm.n_guard*dz
                # Remove the outside particles
                for species in ptcl :
                    clean_outside_particles( species, z_zero )
                    
            # The last proc adds new particles
            if (comm is None) or (comm.rank == comm.size-1) :
                # Determine the position below which particles are added
                # (Leave the last two cells empty.)
                p_zmax = zmax - 2*dz
                if comm is not None :
                    p_zmax = p_zmax - comm.n_guard*dz
                # Add the new particles
                for species in ptcl :
                    if species.continuous_injection == True :
                        add_particles( species, p_zmax-n_move*dz,
                                        p_zmax, n_move*p_nz )

            # Exchange the fields and the particles 
            # in the guard cells between domains when using MPI
            if comm is not None :
                # NB : rho and J are not exchanged since they are
                # recalculated at each time step
                comm.exchange_fields( interp, 'EB')
                for species in ptcl:
                    comm.exchange_particles( species,
                            interp[0].zmin, interp[0].zmax )
            
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
        end of the moving window.

        This is done by multiplying the self.ncells_damp first cells
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
                interp[1].Er, interp[1].Et, interp[1].Ez,
                self.d_damp_array_EB, self.ncells_zero, self.ncells_damp, Nr )
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

def add_particles( species, zmin, zmax, Npz ) :
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
    """

    # Take the angle of the last particle as a global shift in theta,
    # in order to prevent the successively-added particles from being aligned
    global_theta = np.angle( species.x[-1] + 1.j*species.y[-1] )
    # Create the particles that will be added
    new_ptcl = Particles( species.q, species.m, species.n,
        Npz, zmin, zmax, species.Npr, species.rmin, species.rmax,
        species.Nptheta, species.dt, species.dens_func,
        global_theta=global_theta )

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
    Multiply the n first cells and last n cells of field_array
    by damp_array, along the first axis.

    Parameters
    ----------
    field_array : 2darray of complexs
    damp_array : 2darray of reals
    n_damp, n_zero : int
    """
    field_array[:n_zero,:] = 0
    field_array[n_zero:n_zero+n_damp,:] = \
        damp_array[:,np.newaxis] * field_array[n_zero:n_zero+n_damp,:]

if cuda_installed :

    @cuda.jit('void(complex128[:,:], complex128[:,:], complex128[:,:], \
                    complex128[:,:], complex128[:,:], complex128[:,:], \
                    complex128[:,:], complex128[:,:], complex128[:,:], \
                    complex128[:,:], complex128[:,:], complex128[:,:], \
                    float64[:], int32, int32, int32)')
    def cuda_damp_EB( Er0, Et0, Ez0, Br0, Bt0, Bz0,
                      Er1, Et1, Ez1, Br1, Bt1, Bz1,
                      damp_array_EB, ncells_zero, ncells_damp, Nr ) :
        """
        Put the fields to 0 in the ncells_zero first cells
        Multiply the fields by damp_array_EB in the next cells

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

        # Modify the fields
        if ir < Nr :
            if iz < ncells_zero :
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
            elif iz < ncells_zero + ncells_damp :
                damp_factor = damp_array_EB[iz - ncells_zero]
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
