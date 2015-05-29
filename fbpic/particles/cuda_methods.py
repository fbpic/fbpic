"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the optimized particles methods that use cuda on a GPU
"""
from numbapro import cuda
import math

# -----------------------
# Particle pusher utility
# -----------------------

@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], float64[:], \
                float64, float64, int64, float64)')
def push_p_cuda( ux, uy, uz, inv_gamma, 
                Ex, Ey, Ez, Bx, By, Bz, q, m, Ntot, dt ) :
    """
    Advance the particles' momenta, using numba on the GPU
    """
    # Set a few constants
    econst = q*dt/(m*c)
    bconst = 0.5*q*dt/m
    
    #Cuda 1D grid
    ip = cuda.grid(1)

    # Loop over the particles
    if ip < Ntot:

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
            2./( sigma + math.sqrt( sigma**2 + 4*(tau2 + utau**2 ) ) )
        )
        inv_gamma[ip] = inv_gamma_f

        # Reuse the tau and utau variables to save memory
        tx = inv_gamma_f*taux
        ty = inv_gamma_f*tauy
        tz = inv_gamma_f*tauz
        ut = inv_gamma_f*utau
        s = 1./( 1 + tau2*inv_gamma_f**2 )

        # Get the new u
        ux[ip] = s*( uxp + tx*ut + uyp*tz - uzp*ty )
        uy[ip] = s*( uyp + ty*ut + uzp*tx - uxp*tz )
        uz[ip] = s*( uzp + tz*ut + uxp*ty - uyp*tx )

# -----------------------
# Field gathering utility
# -----------------------

@cuda.jit('void(complex128[:], int32, complex128[:,:], float64[:], \
           int32[:], int32[:], float64[:], float64[:], \
           int32[:], int32[:], float64[:], float64[:], \
           float64, float64[:])')
def gather_field_cuda( exptheta, m, Fgrid, Fptcl, 
        iz_lower, iz_upper, Sz_lower, Sz_upper,
        ir_lower, ir_upper, Sr_lower, Sr_upper,
        sign_guards, Sr_guard ) :
    """
    Perform the weighted sum using numba on gpu

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
    # Get the CUDA Grid iterator
    ip = cuda.grid(1)
    # Calculate in parallel (only threads < total number of particles)
    if ip < Fptcl.shape[0]:
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
        #if m == 0 :
        Fptcl[ip] += (F*exptheta[ip]).real
        #if m > 0 :
        Fptcl[ip] += 2*(F*exptheta[ip]).real
        
@cuda.jit('void(complex128[:], float64[:], float64[:])')
def split_complex1d(a, b, c):
    """
    Split a 1D complex128 array into two float64 arrays
    on the GPU.

    Parameters
    ----------
    a : 1darray of complexs
    
    b, c : 1darrays of floats
    """
    i = cuda.grid(1)
    if i < len(a):
        b[i] = complex(a[i]).real
        c[i] = complex(a[i]).imag
    cuda.syncthreads()

@cuda.jit('void(complex128[:, :], float64[:, :], float64[:, :])')
def split_complex2d(a, b, c):
    """
    Split a 2D complex128 array into two float64 arrays
    on the GPU.

    Parameters
    ----------
    a : 2darray of complexs
    
    b, c : 2darrays of floats
    """
    i, j = cuda.grid(2)
    if (i < a.shape[0] and j < a.shape[1]):
        b[i, j] = complex(a[i, j]).real
        c[i, j] = complex(a[i, j]).imag
    cuda.syncthreads()

@cuda.jit('void(float64[:], float64[:], complex128[:])')
def merge_complex1d(a, b, c):
    """
    Merge two 1D float64 arrays to one complex128 array
    on the GPU.

    Parameters
    ----------
    a, b : 1darrays of complexs
    
    c : 1darray of floats
    """
    i = cuda.grid(1)
    if i < len(a):
        c[i] = complex(a[i], b[i])
    cuda.syncthreads()

@cuda.jit('void(float64[:, :], float64[:, :], complex128[:, :])')
def merge_complex2d(a, b, c):
    """
    Merge two 2D float64 arrays to one complex128 array
    on the GPU.

    Parameters
    ----------
    a, b : 2darrays of complexs
    
    c : 2darray of floats
    """
    i, j = cuda.grid(2)
    if (i < a.shape[0] and j < a.shape[1]):
        c[i, j] = complex(a[i, j], b[i, j])
    cuda.syncthreads()

@cuda.jit('void(int32[:], int32[:], int32[:], int32, int32)')
def get_cell_idx_per_particle(cell_idx, iz_lower, ir_lower, nz, nr):
    """
    Get the cell index of each particle.
    The cell index is 1d and calculated by:
    cell index in z + cell index in r * number of cells in z

    Parameters
    ----------
    cell_idx : 1darray of integers
        The cell index of the particle
    
    iz_lower : 1darray of integers
        The lower cell in z of the particle

    ir_lower : 1darray of integers
        The lower cell in r of the particle

    nz : int
        The number of cells in z
    nr : int
        The number of cells in r
    """
    i = cuda.grid(1)
    if i < cell_idx.shape[0]:
        cell_idx[i] = iz_lower[i] + ir_lower[i] * nz

def sort_particles_per_cell(cell_idx, sorted_idx):
    """
    Sort the cell index of the particles and 
    modify the sorted index array accordingly.

    Parameters
    ----------
    cell_idx : 1darray of integers
        The cell index of the particle
    
    sorted_idx : 1darray of integers
        Represents the original index of the 
        particle before the sorting.
    """
    Ntot = cell_idx.shape[0]
    sorter = sorting.RadixSort(Ntot, dtype = np.int32)
    sorter.sort(cell_idx, vals = sorted_idx)

@cuda.jit('void(int32[:], int32[:])')
def count_particles_per_cell(cell_idx, frequency_per_cell):
    """
    Count the particle frequency per cell.

    Parameters
    ----------
    cell_idx : 1darray of integers
        The cell index of the particle
    
    frequency_per_cell : 1darray of integers
        Represents the number of particles per cell
    """
    i = cuda.grid(1)
    if i < cell_idx.shape[0]:
        cuda.atomic.add(frequency_per_cell, cell_idx[i], 1)

@cuda.jit('void(int32[:], float64[:])')
def incl_prefix_sum(cell_idx, prefix_sum):
    """
    Perform an inclusive parallel prefix sum on the sorted 
    cell index array. The prefix sum array represents the
    cumulative sum of the number of particles per cell
    for each cell index.

    Parameters
    ----------
    cell_idx : 1darray of integers
        The cell index of the particle
    
    prefix_sum : 1darray of integers
        Represents the cumulative sum of 
        the particles per cell
    """
    i = cuda.grid(1)
    if i < cell_idx.shape[0]:
        cuda.atomic.max(prefix_sum, cell_idx[i], i)

@cuda.jit('void(complex128[:], complex128[:,:], complex128[:,:], \
                    complex128[:,:],complex128[:,:], \
                    float64[:], float64[:], \
                    float64[:], float64[:], \
                    float64[:], float64[:], int32[:], \
                    int32[:], int32[:], uint32[:])')
def deposit_per_cell(Fptcl, Fgrid_per_node0, Fgrid_per_node1, 
                     Fgrid_per_node2, Fgrid_per_node3, 
                     Sz_lower, Sz_upper, Sr_lower, Sr_upper, 
                     sign_guards, Sr_guard, cell_idx, 
                     frequency_per_cell, prefix_sum, sorted_idx):
    """
    Deposit the field to the four field arrays per cell in parallel.
    Each thread corresponds to a cell and loops over all particles 
    within that cell. The deposited field is written to the 
    four separate arrays for each possible deposition direction.
    (for linear weights)

    Parameters
    ----------
    Fptcl : 1darray of complexs
        (one element per macroparticle)
        Contains the charge or current for each macroparticle (already
        multiplied by exp(im theta), from which to do the deposition
    
    Fgrid_per_node0, Fgrid_per_node1, 
    Fgrid_per_node2, Fgrid_per_node3 : 2darrays of complexs
        Contains the fields on the interpolation grid for the four
        possible deposition directions.
        Is modified by this function

    Sz_lower, Sz_upper, Sr_lower, Sr_upper : 1darrays of floats
        (one element per macroparticle)
        Contains the weight for the lower and upper cells.

    sign_guards : float
       The sign (+1 or -1) with which the weight of the guard cells should
       be added to the 0th cell.

    Sr_guard : 1darray of float
        (one element per macroparticle)
        Contains the weight in the guard cells

    cell_idx : 1darray of integers
        The sorted cell index of the particles

    frequency_per_cell : 1darray of integers
        The number of particles per cell

    prefix_sum : 1darray of integers
        The cummulative sum of the number of particles
        per cell for each cell.

    sorted_idx : 1darray of integers
        A sorted array containing the index of the particle
        before the sorting.
    """
    # Get the 1D CUDA grid
    i = cuda.grid(1)
    # Number of cells in each direction
    nz, nr = Fgrid_per_node0.shape

    # Deposit the field per cell in parallel (for threads < number of cells)
    if i < frequency_per_cell.shape[0]:
        # Calculate the cell index in 2D from the 1D cell index
        ir = int(i/nz)
        iz = int(i - ir*nz)
        # Calculate the inclusive offset for the current cell
        # It represents the number of particles contained in all other cells
        # with an index smaller than i + the total number of particles in the 
        # current cell (inclusive).
        incl_offset = int(prefix_sum[i])

        sgn_grds = sign_guards[0]
        # Initialize the local field value for all four possible deposition
        # directions
        F1 = 0.+0.j
        F2 = 0.+0.j
        F3 = 0.+0.j
        F4 = 0.+0.j
        # Loop over the number of particles per cell
        for j in range(frequency_per_cell[i]):
            # Get the particle index before the sorting
            ptcl_idx = sorted_idx[incl_offset-j]
            # Load the data of the particle field and the weights into 
            # the memory
            F = Fptcl[ptcl_idx]
            Szl = Sz_lower[ptcl_idx]
            Szu = Sz_upper[ptcl_idx]
            Srl = Sr_lower[ptcl_idx]
            Sru = Sr_upper[ptcl_idx]
            Srg = Sr_guard[ptcl_idx]
            # Caculate the weighted field values
            F1 += Szl*Srl*F
            F2 += Szl*Sru*F
            F3 += Szu*Srl*F
            F4 += Szu*Sru*F
            # Treat the guard cells
            if ir == 0:
                F1 += sgn_grds*Szl*Srg*F
                F3 += sgn_grds*Szu*Srg*F
        # Write the calculated field values to 
        # the field arrays defined on the interpolation grid
        if (iz < nz and ir < nr):
            Fgrid_per_node0[iz, ir] = F1
            Fgrid_per_node1[iz, ir] = F2 
            Fgrid_per_node2[iz, ir] = F3
            Fgrid_per_node3[iz, ir] = F4

@cuda.jit('void(complex128[:,:], complex128[:,:], \
                complex128[:,:], complex128[:,:], complex128[:,:])')
def add_field(Fgrid, Fgrid_per_node0, Fgrid_per_node1, 
              Fgrid_per_node2, Fgrid_per_node3):
    """
    Deposit the field to the four field arrays per cell in parallel.
    Each thread corresponds to a cell and loops over all particles 
    within that cell. The deposited field is written to the 
    four separate arrays for each possible deposition direction.
    (for linear weights)

    Parameters
    ----------
    Fgrid : 2darray of complexs
        Contains the fields on the interpolation grid
        Is modified by this function

    Fgrid_per_node0, Fgrid_per_node1, 
    Fgrid_per_node2, Fgrid_per_node3 : 2darrays of complexs
        Contains the fields on the interpolation grid for the four
        possible deposition directions.
    """
    i, j = cuda.grid(2)
    if (i < Fgrid.shape[0] and j < Fgrid.shape[1]):
        # Sum the four field arrays for the different deposition 
        # directions and write them to the global field array
        Fgrid[i, j] += Fgrid_per_node0[i, j] + \
                       Fgrid_per_node1[i, j-1] + \
                       Fgrid_per_node2[i-1, j] + \
                       Fgrid_per_node3[i-1, j-1]

@cuda.jit('void(float64[:], float64[:,:], \
                    int32[:], int32[:], float64[:], float64[:], \
                    int32[:], int32[:], float64[:], float64[:], \
                    float64[:], float64[:])')
def deposit_field_cuda( Fptcl, Fgrid, 
         iz_lower, iz_upper, Sz_lower, Sz_upper,
         ir_lower, ir_upper, Sr_lower, Sr_upper,
         sign_guards, Sr_guard ) :
    """
    Perform the deposition on the GPU using cuda.atomic.add

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
    ip = cuda.grid(1)

    if ip < Fptcl.shape[0]:
        # Deposit the particle quantity onto the grid
        # Lower cell in z, Lower cell in r
        cuda.atomic.add( Fgrid, (iz_lower[ip], ir_lower[ip]), Sz_lower[ip]*Sr_lower[ip]*Fptcl[ip] )
        # Lower cell in z, Upper cell in r
        cuda.atomic.add( Fgrid, (iz_lower[ip], ir_upper[ip]), Sz_lower[ip]*Sr_upper[ip]*Fptcl[ip] )
        # Upper cell in z, Lower cell in r
        cuda.atomic.add( Fgrid, (iz_upper[ip], ir_lower[ip]), Sz_upper[ip]*Sr_lower[ip]*Fptcl[ip])
        # Upper cell in z, Upper cell in r
        cuda.atomic.add( Fgrid, (iz_upper[ip], ir_upper[ip]), Sz_upper[ip]*Sr_upper[ip]*Fptcl[ip] )
        # Add the fields from the guard cells
        cuda.atomic.add( Fgrid, (iz_lower[ip], 0), sign_guards[0]*Sz_lower[ip]*Sr_guard[ip]*Fptcl[ip])
        cuda.atomic.add( Fgrid, (iz_upper[ip], 0), sign_guards[0]*Sz_upper[ip]*Sr_guard[ip]*Fptcl[ip])

