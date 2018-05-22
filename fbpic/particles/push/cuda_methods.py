# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the particle push methods on the GPU using CUDA.
"""
from numba import cuda
import math
from scipy.constants import c, e, m_e

# -----------------------
# Pusher without envelope
# -----------------------

@cuda.jit(device=True, inline=True)
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

    return( ux_f, uy_f, uz_f, inv_gamma_f )


@cuda.jit
def push_x_gpu( x, y, z, ux, uy, uz, inv_gamma, dt,
                x_push, y_push, z_push ) :
    """
    Advance the particles' positions over `dt` using the momenta ux, uy, uz,
    multiplied by the scalar coefficients x_push, y_push, z_push.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)

    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    dt : float (seconds)
        The timestep by which the position is advanced

    x_push, y_push, z_push: float, dimensionless
        Multiplying coefficient for the momenta in x, y and z
        e.g. if x_push=1., the particles are pushed forward in x
             if x_push=-1., the particles are pushed backward in x
    """
    # Timestep multiplied by c
    cdt = c*dt

    i = cuda.grid(1)
    if i < x.shape[0]:
        # Particle push
        inv_g = inv_gamma[i]
        x[i] += cdt*x_push*inv_g*ux[i]
        y[i] += cdt*y_push*inv_g*uy[i]
        z[i] += cdt*z_push*inv_g*uz[i]


@cuda.jit
def push_p_gpu( ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz,
                q, m, Ntot, dt ) :
    """
    Advance the particles' momenta, using cuda on the GPU

    Parameters
    ----------
    ux, uy, uz : 1darray of floats
        The velocity of the particles
        (is modified by this function)

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    Ex, Ey, Ez : 1darray of floats
        The electric fields acting on the particles

    Bx, By, Bz : 1darray of floats
        The magnetic fields acting on the particles

    q : float
        The charge of the particle species

    m : float
        The mass of the particle species

    Ntot : int
        The total number of particles

    dt : float
        The time by which the momenta is advanced
    """
    # Set a few constants
    econst = q*dt/(m*c)
    bconst = 0.5*q*dt/m

    #Cuda 1D grid
    ip = cuda.grid(1)

    # Loop over the particles
    if ip < Ntot:
        ux[ip], uy[ip], uz[ip], inv_gamma[ip] = push_p_vay(
            ux[ip], uy[ip], uz[ip], inv_gamma[ip],
            Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], econst, bconst)


@cuda.jit
def push_p_after_plane_gpu( z, z_plane, ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz, q, m, Ntot, dt ) :
    """
    Advance the particles' momenta, using cuda on the GPU.
    Only the particles that are located beyond the plane z=z_plane
    have their momentum modified ; the others particles move ballistically.

    Parameters
    ----------
    z: 1darray of floats
        The position of the particles in the z direction

    z_plane: float
        Position beyond which the particles should be

    For the other parameters, see the docstring of push_p_gpu
    """
    # Set a few constants
    econst = q*dt/(m*c)
    bconst = 0.5*q*dt/m

    # Cuda 1D grid
    ip = cuda.grid(1)

    # Loop over the particles
    if ip < Ntot and z[ip] > z_plane:
        ux[ip], uy[ip], uz[ip], inv_gamma[ip] = push_p_vay(
            ux[ip], uy[ip], uz[ip], inv_gamma[ip],
            Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], econst, bconst)


@cuda.jit
def push_p_ioniz_gpu( ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz,
                m, Ntot, dt, ionization_level ) :
    """
    Advance the particles' momenta, using numba on the GPU
    This take into account that the particles are ionizable, and thus
    that their charge is determined by `ionization_level`

    Parameters
    ----------
    ionization_level : 1darray of ints
        The number of electrons that each ion is missing
        (compared to a neutral atom)

    For the other parameters, see the docstring of push_p_gpu
    """
    #Cuda 1D grid
    ip = cuda.grid(1)

    # Loop over the particles
    if ip < Ntot:
        if ionization_level[ip] != 0:
            # Set a few constants
            econst = ionization_level[ip] * e * dt/(m*c)
            bconst = 0.5 * ionization_level[ip] * e * dt/m
            # Use the Vay pusher
            ux[ip], uy[ip], uz[ip], inv_gamma[ip] = push_p_vay(
                ux[ip], uy[ip], uz[ip], inv_gamma[ip],
                Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], econst, bconst)

# -----------------------
# Pusher with envelope
# -----------------------


@cuda.jit
def push_p_envelope_gpu( ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz, a2, grad_a2_x, grad_a2_y, grad_a2_z,
                q, m, Ntot, dt , keep_momentum = True) :
    """
    Advance the particles' momenta, using cuda on the GPU

    Parameters
    ----------
    ux, uy, uz : 1darray of floats
        The velocity of the particles
        (is modified by this function)

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    Ex, Ey, Ez : 1darray of floats
        The electric fields acting on the particles

    Bx, By, Bz : 1darray of floats
        The magnetic fields acting on the particles

    a2, grad_a2_x, grad_a2_y, grad_a2_z : 1d array of floats
        The envelope fields acting on the particle for ponderomotive force and
        quiver motion

    q : float
        The charge of the particle species

    m : float
        The mass of the particle species

    Ntot : int
        The total number of particles

    dt : float
        The time by which the momenta is advanced

    keep_momentum : boolean
        Whether or not to register the new momentum obtained in the particles,
        or only the new gamma.
    """
    # Set a few constants
    econst = q*dt/(m*c)
    bconst = 0.5*q*dt/m
    scale_factor = 0.5 ( q * m_e / (e * m) )**2
    aconst = c * scale_factor * dt * 0.25

    #Cuda 1D grid
    ip = cuda.grid(1)

    # Loop over the particles
    if ip < Ntot:
        if keep_momentum:
            ux[ip], uy[ip], uz[ip], inv_gamma[ip] = push_p_vay_envelope(
                ux[ip], uy[ip], uz[ip], inv_gamma[ip],
                Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], a2[ip],
                grad_a2_x[ip], grad_a2_y[ip], grad_a2_z[ip], econst, bconst,
                aconst, scale_factor)
        else:
            _, _, _, inv_gamma[ip] = push_p_vay_envelope(
                ux[ip], uy[ip], uz[ip], inv_gamma[ip],
                Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], a2[ip],
                grad_a2_x[ip], grad_a2_y[ip], grad_a2_z[ip], econst, bconst,
                aconst, scale_factor)

@cuda.jit
def push_p_after_plane_envelope_gpu( z, z_plane, ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz, a2, grad_a2_x, grad_a2_y, grad_a2_z,
                q, m, Ntot, dt , keep_momentum = True ) :
    """
    Advance the particles' momenta, using cuda on the GPU.
    Only the particles that are located beyond the plane z=z_plane
    have their momentum modified ; the others particles move ballistically.

    Parameters
    ----------
    z: 1darray of floats
        The position of the particles in the z direction

    z_plane: float
        Position beyond which the particles should be

    For the other parameters, see the docstring of push_p_gpu
    """
    # Set a few constants
    econst = q*dt/(m*c)
    bconst = 0.5*q*dt/m
    scale_factor = 0.5 ( q * m_e / (e * m) )**2
    aconst = c * scale_factor * dt * 0.25

    # Cuda 1D grid
    ip = cuda.grid(1)

    # Loop over the particles
    if ip < Ntot and z[ip] > z_plane:
        if keep_momentum:
            ux[ip], uy[ip], uz[ip], inv_gamma[ip] = push_p_vay_envelope(
                ux[ip], uy[ip], uz[ip], inv_gamma[ip],
                Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], a2[ip],
                grad_a2_x[ip], grad_a2_y[ip], grad_a2_z[ip], econst, bconst,
                aconst, scale_factor)
        else:
            _, _, _, inv_gamma[ip] = push_p_vay_envelope(
                ux[ip], uy[ip], uz[ip], inv_gamma[ip],
                Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], a2[ip],
                grad_a2_x[ip], grad_a2_y[ip], grad_a2_z[ip], econst, bconst,
                aconst, scale_factor)


@cuda.jit
def push_p_ioniz_envelope_gpu( ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz, a2, grad_a2_x, grad_a2_y, grad_a2_z,
                m, Ntot, dt, ionization_level, keep_momentum = True ) :
    """
    Advance the particles' momenta, using numba on the GPU
    This take into account that the particles are ionizable, and thus
    that their charge is determined by `ionization_level`

    Parameters
    ----------
    ionization_level : 1darray of ints
        The number of electrons that each ion is missing
        (compared to a neutral atom)

    For the other parameters, see the docstring of push_p_gpu
    """
    #Cuda 1D grid
    ip = cuda.grid(1)

    # Loop over the particles
    if ip < Ntot:
        if ionization_level[ip] != 0:
            # Set a few constants
            econst = ionization_level[ip] * e * dt/(m*c)
            bconst = 0.5 * ionization_level[ip] * e * dt/m
            # Use the Vay pusher
            if keep_envelope:
                ux[ip], uy[ip], uz[ip], inv_gamma[ip] = push_p_vay_envelope(
                    ux[ip], uy[ip], uz[ip], inv_gamma[ip],
                    Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], a2[ip],
                    grad_a2_x[ip], grad_a2_y[ip], grad_a2_z[ip], econst, bconst,
                    aconst, scale_factor)
            else:
                _, _, _, inv_gamma[ip] = push_p_vay_envelope(
                    ux[ip], uy[ip], uz[ip], inv_gamma[ip],
                    Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], a2[ip],
                    grad_a2_x[ip], grad_a2_y[ip], grad_a2_z[ip], econst, bconst,
                    aconst, scale_factor)


@cuda.jit(device=True, inline=True)
def push_p_vay_envelope( ux_i, uy_i, uz_i, inv_gamma_i,
    Ex, Ey, Ez, Bx, By, Bz, a2_i, grad_a2_x_i, grad_a2_y_i, grad_a2_z_i,
    econst, bconst, aconst, scale_factor ):
    """
    Push at single macroparticle, using the Vay pusher
    """
    # First step, modelling first half of the ponderomotive force
    inv_gamma_temp = 1. / math.sqrt(1 + ux_i**2 + uy_i**2 + uz_i**2 + scale_factor * a2_i)

    ux1 = ux_i - aconst * inv_gamma_temp * grad_a2_x_i
    uy1 = uy_i - aconst * inv_gamma_temp * grad_a2_y_i
    uz1 = uz_i - aconst * inv_gamma_temp * grad_a2_z_i


    inv_gamma_temp = 1. / math.sqrt(1 + ux1**2 + uy1**2 + uz1**2 + scale_factor * a2_i)

    # Get the magnetic rotation vector
    taux = bconst*Bx
    tauy = bconst*By
    tauz = bconst*Bz
    tau2 = taux**2 + tauy**2 + tauz**2

    # Get the momenta at the half timestep
    uxp = ux_i + econst*Ex \
    + inv_gamma_temp*( uy_i*tauz - uz_i*tauy )
    uyp = uy_i + econst*Ey \
    + inv_gamma_temp*( uz_i*taux - ux_i*tauz )
    uzp = uz_i + econst*Ez \
    + inv_gamma_temp*( ux_i*tauy - uy_i*taux )
    sigma = 1 + uxp**2 + uyp**2 + uzp**2 + scale_factor * a2_i - tau2
    utau = uxp*taux + uyp*tauy + uzp*tauz

    # Get the new 1./gamma
    inv_gamma_f = math.sqrt(
        2./( sigma + math.sqrt( sigma**2 + 4*(tau2*(1 + scale_factor * a2_i) + utau**2 ) ) ) )

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

    ux_f -= aconst * inv_gamma_f * grad_a2_x_i
    uy_f -= aconst * inv_gamma_f * grad_a2_y_i
    uz_f -= aconst * inv_gamma_f * grad_a2_z_i

    inv_gamma_f = 1. / math.sqrt(1 + ux_f**2 + uy_f**2 + uz_f**2 + scale_factor * a2_i)

    return( ux_f, uy_f, uz_f, inv_gamma_f )
