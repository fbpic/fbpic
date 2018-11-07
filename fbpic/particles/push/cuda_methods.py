# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the particle push methods on the GPU using CUDA.
"""
import math
from scipy.constants import c, e, m_e
from numba import cuda
# Import inline function
from .inline_functions import push_p_vay, push_p_vay_envelope
# Compile the inline function for GPU
push_p_vay = cuda.jit( push_p_vay, device=True, inline=True )
push_p_vay_envelope = cuda.jit( push_p_vay_envelope,
                                device=True, inline=True )

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
                q, m, Ntot, dt, keep_momentum=True) :
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
    # In order to update gamma
    scale_factor = 0.5 * ( q * m_e / (e * m) )**2
    # In order to push with the ponderomotive force over half a timestep
    aconst = 0.25 * ( q * m_e / (e * m) )**2 * c * 0.5 * dt

    # Cuda 1D grid
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
                q, m, Ntot, dt, keep_momentum=True ) :
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
    # In order to update gamma
    scale_factor = 0.5 * ( q * m_e / (e * m) )**2
    # In order to push with the ponderomotive force over half a timestep
    aconst = 0.25 * ( q * m_e / (e * m) )**2 * c * 0.5 * dt

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
    # Cuda 1D grid
    ip = cuda.grid(1)
    # Loop over the particles
    if ip < Ntot:
        if ionization_level[ip] != 0:
            # Set a few constants
            q = ionization_level[ip] * e
            econst = q * dt/(m*c)
            bconst = 0.5 * q * dt/m
            # In order to update gamma
            scale_factor = 0.5 * ( q * m_e / (e * m) )**2
            # In order to push with the ponderomotive force over half timestep
            aconst = 0.25 * ( q * m_e / (e * m) )**2 * c * 0.5 * dt
            # Use the Vay pusher
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
def update_inv_gamma_gpu(a2, ux, uy, uz, inv_gamma, q, m):
    """
    Recompute the gamma factor of the particles, taking into account
    the quiver motion from the envelope 'a' field.

    Parameters
    ----------
    a2: 1d array of floats, dimensionless
        Envelope field acting on the particle

    ux, uy, uz : 1darray of floats (dimensionless)
        The velocity of the particles

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    q : float
        The charge of the particle species
    """
    scale_factor = 0.5 * ( q * m_e / (e * m) )**2
    i = cuda.grid(1)
    if i < ux.shape[0]:
        inv_gamma[i] = 1./math.sqrt(1 + ux[i]**2 + uy[i]**2 + uz[i]**2 \
                                + scale_factor * a2[i] )
