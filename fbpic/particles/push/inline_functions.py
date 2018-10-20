# Copyright 2018, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines inline functions that are compiled for both GPU and CPU, and
used in the functions for the particle pusher
"""
import math

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


def push_p_vay_envelope( ux_i, uy_i, uz_i, inv_gamma_i,
    Ex, Ey, Ez, Bx, By, Bz, a2_i, grad_a2_x_i, grad_a2_y_i, grad_a2_z_i,
    econst, bconst, aconst, scale_factor ):
    """
    Push at single macroparticle, using the Vay pusher
    """
    # First step: first half of the ponderomotive force
    inv_gamma_temp = 1. / math.sqrt(1 + ux_i**2 + uy_i**2 + uz_i**2 \
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
