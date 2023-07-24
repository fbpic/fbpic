"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines inline functions that are compiled for both GPU and CPU, and
used in the ionization code.
"""
import math
from scipy.constants import c, e, m_e

def get_angles_and_gamma( ux, uy, uz ):
    theta_x = math.atan2( ux, uz )
    theta_y = math.atan2( uy, uz )

    gamma = math.sqrt( 1 + ux**2 + uy**2 + uz**2 )

    return( theta_x, theta_y, gamma )

def get_linear_coefficients(x0, xmin, dx):

    s_ix = ( x0 - xmin ) / dx
    ix = math.floor( s_ix )
    s1 = s_ix - ix
    s0 = 1. - s1
    ix = int(ix)

    return ix, s0, s1

def get_particle_radiation(
    ux, uy, uz, w,
    Ex, Ey, Ez,
    cBx, cBy, cBz,
    gamma_p, Larmore_factor,
    SR_dxi, SR_xi_data,
    omega_ax, spect_loc
    ):

    d_omega = omega_ax[1] - omega_ax[0]

    # get normalized velocity beta
    bx = ux / gamma_p
    by = uy / gamma_p
    bz = uz / gamma_p

    # get normalized velocity beta
    E_dot_beta = Ex * bx + Ey * by + Ez * bz

    accel_factor = - e / ( m_e * c * gamma_p )

    dt_bx = accel_factor * ( Ex - bx * E_dot_beta + by * cBz - bz * cBy )
    dt_by = accel_factor * ( Ey - by * E_dot_beta + bz * cBx - bx * cBz )
    dt_bz = accel_factor * ( Ez - bz * E_dot_beta + bx * cBy - by * cBx )

    Energy_Larmor = Larmore_factor * w * gamma_p**6 * (
         dt_bx**2 + dt_by**2 + dt_bz**2 - \
         ( by * dt_bz - bz * dt_by )**2 - \
         ( bz * dt_bx - bx * dt_bz )**2 - \
         ( bx * dt_by - by * dt_bx )**2
        )

    v_abs = c * math.sqrt(bx * bx + by * by + bz * bz)
    dt_v_abs = c * math.sqrt(dt_bx * dt_bx + dt_by * dt_by + dt_bz * dt_bz)
    v_dot_dt_v = c**2 * (bx * dt_bx + by * dt_by + bz * dt_bz)

    omega_c = 1.5 * gamma_p**3 * c * math.sqrt(
        v_abs**2 * dt_v_abs**2 - v_dot_dt_v**2 ) / v_abs**3

    if (omega_c < d_omega):
        spect_loc[:] = 0.0
        return( spect_loc )

    omega_c_inv = 1. / omega_c

    N_omega_src = SR_xi_data.size

    S_xi_integral = 0.0
    d_xi_loc =  d_omega / omega_c
    for i_omega in range(omega_ax.size):

        xi_loc = omega_ax[i_omega] / omega_c

        s_ix_src = xi_loc / SR_dxi
        ix_src = math.floor( s_ix_src )

        if ( ix_src >= N_omega_src-1 ):
            spect_loc[i_omega] = 0.0
        else:
            s1 = s_ix_src - ix_src
            s0 = 1.0 - s1
            ix_src_int = int(ix_src)
            S_xi_loc = SR_xi_data[ix_src_int] * s0 + SR_xi_data[ix_src_int+1] * s1

            spect_loc[i_omega] = Energy_Larmor * omega_c_inv * S_xi_loc
            S_xi_integral += S_xi_loc * d_xi_loc

    if S_xi_integral > 0:
        spect_loc /= S_xi_integral

    return( spect_loc )
