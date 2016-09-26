# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the optimized particles methods that use numba on a CPU
"""
import numba
import math
from scipy.constants import c

# -----------------------
# Particle pusher utility
# -----------------------

@numba.jit(nopython=True)
def push_x_numba( x, y, z, ux, uy, uz, inv_gamma, Ntot, dt ):
    """
    Advance the particles' positions over one half-timestep
    
    This assumes that the positions (x, y, z) are initially either
    one half-timestep *behind* the momenta (ux, uy, uz), or at the
    same timestep as the momenta.
    """
    # Half timestep, multiplied by c
    chdt = c*0.5*dt

    # Particle push
    for ip in range(Ntot) :
        x[ip] += chdt * inv_gamma[ip] * ux[ip]
        y[ip] += chdt * inv_gamma[ip] * uy[ip]
        z[ip] += chdt * inv_gamma[ip] * uz[ip]

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
    for ip in range(Ntot) :

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
    for ip in range(Ntot) :
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

# -------------------------
# Charge deposition utility
# -------------------------
            
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
    for ip in range(Ntot) :
        Fgrid[ iz_lower[ip], ir_lower[ip] ] += \
          Sz_lower[ip] * Sr_lower[ip] * Fptcl[ip]
    # Lower cell in z, Upper cell in r
    for ip in range(Ntot) :
        Fgrid[ iz_lower[ip], ir_upper[ip] ] += \
          Sz_lower[ip] * Sr_upper[ip] * Fptcl[ip]
    # Upper cell in z, Lower cell in r
    for ip in range(Ntot) :
        Fgrid[ iz_upper[ip], ir_lower[ip] ] += \
          Sz_upper[ip] * Sr_lower[ip] * Fptcl[ip]
    # Upper cell in z, Upper cell in r
    for ip in range(Ntot) :
        Fgrid[ iz_upper[ip], ir_upper[ip] ] += \
          Sz_upper[ip] * Sr_upper[ip] * Fptcl[ip]

    # Add the fields from the guard cells in r
    for ip in range(Ntot) :
        Fgrid[ iz_lower[ip], 0 ] += \
            sign_guards * Sz_lower[ip]*Sr_guard[ip] * Fptcl[ip]
    for ip in range(Ntot) :
        Fgrid[ iz_upper[ip], 0 ] += \
            sign_guards * Sz_upper[ip]*Sr_guard[ip] * Fptcl[ip]
