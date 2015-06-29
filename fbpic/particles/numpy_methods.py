"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the optimized particles methods that use numba on a CPU
"""
import numpy as np
from scipy.constants import c

# -----------------------
# Particle pusher utility
# -----------------------
          
def push_p_numpy( ux, uy, uz, inv_gamma,
        Ex, Ey, Ez, Bx, By, Bz,
        q, m, Ntot, dt ) :
    """
    Advance the particles' momenta, using numpy
    """
    # Set a few constants
    econst = q*dt/(m*c)
    bconst = 0.5*q*dt/m
    
    # Get the magnetic rotation vector
    taux = bconst*Bx
    tauy = bconst*By
    tauz = bconst*Bz
    tau2 = taux**2 + tauy**2 + tauz**2

    # Get the momenta at the half timestep
    ux_tmp = ux + econst*Ex \
      + inv_gamma*( uy*tauz - uz*tauy )
    uy_tmp = uy + econst*Ey \
      + inv_gamma*( uz*taux - ux*tauz )
    uz_tmp = uz + econst*Ez \
      + inv_gamma*( ux*tauy - uy*taux )
    sigma = 1 + ux_tmp**2 + uy_tmp**2 + uz_tmp**2 - tau2
    utau = ux_tmp*taux + uy_tmp*tauy + uz_tmp*tauz

    # Get the new 1./gamma
    inv_gamma = np.sqrt(
    2./( sigma + np.sqrt( sigma**2 + 4*(tau2 + utau**2 ) ) )
    )

    # Reuse the tau and utau arrays to save memory
    taux[:] = inv_gamma*taux
    tauy[:] = inv_gamma*tauy
    tauz[:] = inv_gamma*tauz
    utau[:] = inv_gamma*utau
    s = 1./( 1 + tau2*inv_gamma**2 )

    # Get the new u
    ux[:] = s*( ux_tmp + taux*utau + uy_tmp*tauz - uz_tmp*tauy )
    uy[:] = s*( uy_tmp + tauy*utau + uz_tmp*taux - ux_tmp*tauz )
    uz[:] = s*( uz_tmp + tauz*utau + ux_tmp*tauy - uy_tmp*taux )

def push_x_numpy( x, y, z, ux, uy, uz, inv_gamma, dt ) :
    """
    Advance the particles' positions over one half-timestep
    
    This assumes that the positions (x, y, z) are initially either
    one half-timestep *behind* the momenta (ux, uy, uz), or at the
    same timestep as the momenta.
    """
    # Half timestep, multiplied by c
    chdt = c*0.5*dt

    # Particle push
    x[:] += chdt*inv_gamma*ux
    y[:] += chdt*inv_gamma*uy
    z[:] += chdt*inv_gamma*uz

# -----------------------
# Field gathering utility
# -----------------------

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

# -------------------------
# Charge deposition utility
# -------------------------
            
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
