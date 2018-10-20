# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the particle push methods on the CPU with numba.
"""
import math
import numba
from fbpic.utils.threading import njit_parallel, prange
from scipy.constants import c, e, m_e
# Import inline functions
from .inline_functions import push_p_vay
# Compile the inline functions for CPU
push_p_vay = numba.njit(push_p_vay)

@njit_parallel
def push_x_numba( x, y, z, ux, uy, uz, inv_gamma, Ntot, dt,
                push_x, push_y, push_z ):
    """
    Advance the particles' positions over `dt` using the momenta ux, uy, uz,
    multiplied by the scalar coefficients x_push, y_push, z_push.
    """
    # Half timestep, multiplied by c
    chdt = c*dt

    # Particle push (in parallel if threading is installed)
    for ip in prange(Ntot) :
        x[ip] += chdt * inv_gamma[ip] * push_x * ux[ip]
        y[ip] += chdt * inv_gamma[ip] * push_y * uy[ip]
        z[ip] += chdt * inv_gamma[ip] * push_z * uz[ip]

    return x, y, z

@njit_parallel
def push_p_numba( ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz, q, m, Ntot, dt ) :
    """
    Advance the particles' momenta, using numba
    """
    # Set a few constants
    econst = q*dt/(m*c)
    bconst = 0.5*q*dt/m

    # Loop over the particles (in parallel if threading is installed)
    for ip in prange(Ntot) :
        ux[ip], uy[ip], uz[ip], inv_gamma[ip] = push_p_vay(
            ux[ip], uy[ip], uz[ip], inv_gamma[ip],
            Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], econst, bconst )

    return ux, uy, uz, inv_gamma

@njit_parallel
def push_p_after_plane_numba( z, z_plane, ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz, q, m, Ntot, dt ) :
    """
    Advance the particles' momenta, using numba.
    Only the particles that are located beyond the plane z=z_plane
    have their momentum modified ; the others particles move ballistically.
    """
    # Set a few constants
    econst = q*dt/(m*c)
    bconst = 0.5*q*dt/m

    # Loop over the particles (in parallel if threading is installed)
    for ip in prange(Ntot) :
        if z[ip] > z_plane:
            ux[ip], uy[ip], uz[ip], inv_gamma[ip] = push_p_vay(
                ux[ip], uy[ip], uz[ip], inv_gamma[ip],
                Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], econst, bconst)


@njit_parallel
def push_p_ioniz_numba( ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz, m, Ntot, dt, ionization_level ) :
    """
    Advance the particles' momenta, using numba
    """
    # Set a few constants
    prefactor_econst = e*dt/(m*c)
    prefactor_bconst = 0.5*e*dt/m

    # Loop over the particles (in parallel if threading is installed)
    for ip in prange(Ntot) :

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

    return ux, uy, uz, inv_gamma

# -----------------------
# Pusher with envelope
# -----------------------

@njit_parallel
def push_p_envelope_numba( ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz, a2, grad_a2_x, grad_a2_y, grad_a2_z,
                q, m, Ntot, dt, keep_momentum=True) :
    """
    Advance the particles' momenta, using numba
    """
    # Set a few constants
    econst = q*dt/(m*c)
    bconst = 0.5*q*dt/m
    # In order to update gamma
    scale_factor = 0.5 * ( q * m_e / (e * m) )**2
    # In order to push with the ponderomotive force over half a timestep
    aconst = 0.25 * ( q * m_e / (e * m) )**2 * c * 0.5 * dt

    # Loop over the particles (in parallel if threading is installed)
    for ip in prange(Ntot) :
        aux_x, aux_y, aux_z, inv_gamma[ip] = push_p_vay_envelope(
            ux[ip], uy[ip], uz[ip], inv_gamma[ip],
            Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], a2[ip],
            grad_a2_x[ip], grad_a2_y[ip], grad_a2_z[ip], econst, bconst,
            aconst, scale_factor)
        if keep_momentum:
            ux[ip], uy[ip], uz[ip] = aux_x, aux_y, aux_z
    return ux, uy, uz, inv_gamma

@njit_parallel
def push_p_after_plane_envelope_numba( z, z_plane, ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz, a2, grad_a2_x, grad_a2_y, grad_a2_z,
                q, m, Ntot, dt, keep_momentum = True ) :
    """
    Advance the particles' momenta, using numba.
    Only the particles that are located beyond the plane z=z_plane
    have their momentum modified ; the others particles move ballistically.
    """
    # Set a few constants
    econst = q*dt/(m*c)
    bconst = 0.5*q*dt/m
    scale_factor = 0.5 * ( q * m_e / (e * m) )**2
    aconst = c * scale_factor * dt * 0.25

    # Loop over the particles (in parallel if threading is installed)
    for ip in prange(Ntot) :
        if z[ip] > z_plane:
            aux_x, aux_y, aux_z, inv_gamma[ip] = push_p_vay_envelope(
                ux[ip], uy[ip], uz[ip], inv_gamma[ip],
                Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], a2[ip],
                grad_a2_x[ip], grad_a2_y[ip], grad_a2_z[ip], econst, bconst,
                aconst, scale_factor)
            if keep_momentum:
                ux[ip], uy[ip], uz[ip] = aux_x, aux_y, aux_z


@njit_parallel
def push_p_ioniz_envelope_numba( ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz, a2, grad_a2_x, grad_a2_y, grad_a2_z,
                m, Ntot, dt, ionization_level, keep_momentum = True ) :
    """
    Advance the particles' momenta, using numba
    """
    # Loop over the particles (in parallel if threading is installed)
    for ip in prange(Ntot) :

        # For neutral macroparticles, skip this step
        if ionization_level[ip] == 0:
            continue

        # Set a few constants
        q = ionization_level[ip] * e
        econst = q * dt/(m*c)
        bconst = 0.5 * q * dt/m
        # In order to update gamma
        scale_factor = 0.5 * ( q * m_e / (e * m) )**2
        # In order to push with the ponderomotive force over half timestep
        aconst = 0.25 * ( q * m_e / (e * m) )**2 * c * 0.5 * dt

        # Perform the push
        aux_x, aux_y, aux_z, inv_gamma[ip] = push_p_vay_envelope(
            ux[ip], uy[ip], uz[ip], inv_gamma[ip],
            Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], a2[ip],
            grad_a2_x[ip], grad_a2_y[ip], grad_a2_z[ip], econst, bconst,
            aconst, scale_factor)
        if keep_momentum:
            ux[ip], uy[ip], uz[ip] = aux_x, aux_y, aux_z

    return ux, uy, uz, inv_gamma


@numba.njit
def push_p_vay_envelope( ux_i, uy_i, uz_i, inv_gamma_i,
    Ex, Ey, Ez, Bx, By, Bz, a2_i, grad_a2_x_i, grad_a2_y_i, grad_a2_z_i,
    econst, bconst, aconst, scale_factor ):
    """
    Push at single macroparticle, using the Vay pusher
    """
    # First step: first half of the ponderomotive force
    inv_gamma_temp = 1. / math.sqrt( 1 + ux_i**2 + uy_i**2 + uz_i**2 \
                                    + scale_factor * a2_i)
    ux1 = ux_i - aconst * inv_gamma_temp * grad_a2_x_i
    uy1 = uy_i - aconst * inv_gamma_temp * grad_a2_y_i
    uz1 = uz_i - aconst * inv_gamma_temp * grad_a2_z_i
    # Update gamma accordingly
    inv_gamma_temp = 1. / math.sqrt(1 + ux1**2 + uy1**2 + uz1**2 \
                                    + scale_factor * a2_i)

    # Get the magnetic rotation vector
    taux = bconst*Bx
    tauy = bconst*By
    tauz = bconst*Bz
    tau2 = taux**2 + tauy**2 + tauz**2

    # Get the momenta at the half timestep
    uxp = ux1 + econst*Ex \
    + inv_gamma_temp*( uy1*tauz - uz1*tauy )
    uyp = uy1 + econst*Ey \
    + inv_gamma_temp*( uz1*taux - ux1*tauz )
    uzp = uz1 + econst*Ez \
    + inv_gamma_temp*( ux1*tauy - uy1*taux )
    sigma = 1 + uxp**2 + uyp**2 + uzp**2 + scale_factor * a2_i - tau2
    utau = uxp*taux + uyp*tauy + uzp*tauz

    # Get the new 1./gamma
    inv_gamma_f = math.sqrt(
        2./( sigma + math.sqrt( sigma**2 + 4*(tau2*(1 + scale_factor * a2_i) \
                                                + utau**2 ) ) ) )

    # Reuse the tau and utau arrays to save memory
    tx = inv_gamma_f*taux
    ty = inv_gamma_f*tauy
    tz = inv_gamma_f*tauz
    ut = inv_gamma_f*utau
    s = 1./( 1 + tau2*inv_gamma_f**2 )

    # Get the new u
    ux_f = s*( uxp + tx*ut + uyp*tz - uzp*ty )
    uy_f = s*( uyp + ty*ut + uzp*tx - uxp*tz )
    uz_f = s*( uzp + tz*ut + uxp*ty - uyp*tx )

    # Last step: second half of the ponderomotive force
    ux_f -= aconst * inv_gamma_f * grad_a2_x_i
    uy_f -= aconst * inv_gamma_f * grad_a2_y_i
    uz_f -= aconst * inv_gamma_f * grad_a2_z_i
    # Update gamma accordingly
    inv_gamma_f = 1. / math.sqrt(1 + ux_f**2 + uy_f**2 + uz_f**2 \
                                + scale_factor * a2_i)

    return( ux_f, uy_f, uz_f, inv_gamma_f )

@njit_parallel
def update_inv_gamma_numba(a2, ux, uy, uz, inv_gamma, q, m, Ntot):
    """
    Recompute the gamma factor of the particles, taking into account
    the quiver motion from the envelope 'a' field.

    Parameters
    ----------
    a2: 1d array of floats, dimensionless
        Envelope field acting on the particle

    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    q : float
        The charge of the particle species
    """
    scale_factor = 0.5 * ( q * m_e / (e * m) )**2
    for ip in prange(Ntot) :
        inv_gamma[ip] = 1./math.sqrt(1 + ux[ip]**2 + uy[ip]**2 + uz[ip]**2 \
                                + scale_factor * a2[ip] )
