# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the optimized particles methods that use numba on a CPU
"""
import numba
import math
from scipy.constants import c, e

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
        ux[ip], uy[ip], uz[ip], inv_gamma[ip] = push_p_vay(
            ux[ip], uy[ip], uz[ip], inv_gamma[ip],
            Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], econst, bconst )

@numba.jit(nopython=True)
def push_p_ioniz_numba( ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz, m, Ntot, dt, ionization_level ) :
    """
    Advance the particles' momenta, using numba
    """
    # Set a few constants
    prefactor_econst = e*dt/(m*c)
    prefactor_bconst = 0.5*e*dt/m

    # Loop over the particles
    for ip in range(Ntot) :

        # For neutral macroparticles, skip this step
        if ionization_level[ip] == 0:
            continue

        # Calculate the charge dependent constants
        econst = prefactor_econst * ionization_level[ip]
        bconst = prefactor_bconst * ionization_level[ip]
        # Perform the push
        ux[ip], uy[ip], uz[ip], inv_gamma[ip] = push_p_vay(
            ux[ip], uy[ip], uz[ip], inv_gamma[ip],
            Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip],
            econst, bconst )

@numba.jit(nopython=True)
def push_p_vay( ux_i, uy_i, uz_i, inv_gamma_i,
                Ex, Ey, Ez, Bx, By, Bz, econst, bconst ):
    """
    Push at single macroparticle, using the Vay pusher
    """
    # Get the magnetic rotation vector
    taux = bconst*Bx
    tauy = bconst*By
    tauz = bconst*Bz
    tau2 = taux**2 + tauy**2 + tauz**2

    # Get the momenta at the half timestep
    uxp = ux_i + econst*Ex \
    + inv_gamma_i*( uy_i*tauz - uz_i*tauy )
    uyp = uy_i + econst*Ey \
    + inv_gamma_i*( uz_i*taux - ux_i*tauz )
    uzp = uz_i + econst*Ez \
    + inv_gamma_i*( ux_i*tauy - uy_i*taux )
    sigma = 1 + uxp**2 + uyp**2 + uzp**2 - tau2
    utau = uxp*taux + uyp*tauy + uzp*tauz

    # Get the new 1./gamma
    inv_gamma_f = math.sqrt(
        2./( sigma + math.sqrt( sigma**2 + 4*(tau2 + utau**2 ) ) ) )

    # Reuse the tau and utau variables to save memory
    tx = inv_gamma_f*taux
    ty = inv_gamma_f*tauy
    tz = inv_gamma_f*tauz
    ut = inv_gamma_f*utau
    s = 1./( 1 + tau2*inv_gamma_f**2 )

    # Get the new u
    ux_f = s*( uxp + tx*ut + uyp*tz - uzp*ty )
    uy_f = s*( uyp + ty*ut + uzp*tx - uxp*tz )
    uz_f = s*( uzp + tz*ut + uxp*ty - uyp*tx )

    return( ux_f, uy_f, uz_f, inv_gamma_f )

# -----------------------
# Field gathering utility
# -----------------------

@numba.jit(nopython=True)
def gather_field_numba(exptheta, m, Fgrid, Fptcl,
                       iz, ir, Sz, Sr, sign_guards):
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

    iz, ir : 2darray of ints
        Arrays of shape (shape_order+1, Ntot)
        where Ntot is the number of macroparticles
        Contains the index of the cells that each macroparticle
        will gather from.

    Sz, Sr: 2darray of floats
        Arrays of shape (shape_order+1, Ntot)
        where Ntot is the number of macroparticles
        Contains the weight for respective cells from iz and ir,
        for each macroparticle.

    sign_guards : float
       The sign (+1 or -1) with which the weight of the guard cells should
       be added to the 0th cell.
    """
    # Get the total number of particles
    Ntot = len(Fptcl)

    # Loop over the particles
    for ip in range(Ntot):
        # Erase the temporary variable
        F = 0.j
        # Loop over all the adjacent cells (given by shape order)
        # Use helper variables `ir_corr` and `Sr_corr`.
        # This is necessary, because ir and Sr should **not** be modified
        # **in-place**. (This is because ir and Sr are reused several
        # times, as we call the present function 3 times, with different
        # values for sign_guards.)
        for cell_index_r in range(ir.shape[0]):
            for cell_index_z in range(iz.shape[0]):
                # Correct the guard cell index and sign
                if ir[cell_index_r, ip] < 0:
                    ir_corr = abs(ir[cell_index_r, ip]) - 1
                    Sr_corr = sign_guards * Sr[cell_index_r, ip]
                else:
                    ir_corr = ir[cell_index_r, ip]
                    Sr_corr = Sr[cell_index_r, ip]
                # Gather the field value at the respective grid point
                F += Sz[cell_index_z, ip] * Sr_corr * \
                    Fgrid[ iz[cell_index_z, ip], ir_corr]

        # Add the complex phase
        if m == 0:
            Fptcl[ip] += (F * exptheta[ip]).real
        if m > 0:
            Fptcl[ip] += 2 * (F * exptheta[ip]).real

# -------------------------
# Charge deposition utility
# -------------------------

@numba.jit(nopython=True)
def deposit_field_numba(Fptcl, Fgrid,
        iz, ir, Sz, Sr, sign_guards):
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

    iz, ir : 2darray of ints
        Arrays of shape (shape_order+1, Ntot)
        where Ntot is the number of macroparticles
        Contains the index of the cells that each macroparticle
        will deposit to.

    Sz, Sr: 2darray of floats
        Arrays of shape (shape_order+1, Ntot)
        where Ntot is the number of macroparticles
        Contains the weight for respective cells from iz and ir,
        for each macroparticle.

    sign_guards : float
       The sign (+1 or -1) with which the weight of the guard cells should
       be added to the 0th cell.
    """

    # Get the total number of particles
    Ntot = len(Fptcl)

    # Loop over all particles
    for ip in range(Ntot):
        # Loop over adjacent cells (given by shape order)
        # Use helper variables `ir_corr` and `Sr_corr`, in order to avoid
        # modifying ir and Sr in place. (This is not strictly necessary,
        # but is just here as a safeguard.)
        for cell_index_r in range(ir.shape[0]):
            for cell_index_z in range(iz.shape[0]):
                # Correct the guard cell index and sign
                if ir[cell_index_r, ip] < 0:
                    ir_corr = abs(ir[cell_index_r, ip]) - 1
                    Sr_corr = sign_guards * Sr[cell_index_r, ip]
                else:
                    ir_corr = ir[cell_index_r, ip]
                    Sr_corr = Sr[cell_index_r, ip]
                # Deposit field from particle to the respective grid point
                Fgrid[ iz[cell_index_z, ip], ir_corr ] += \
                    Sz[cell_index_z,ip] * Sr_corr * Fptcl[ip]
