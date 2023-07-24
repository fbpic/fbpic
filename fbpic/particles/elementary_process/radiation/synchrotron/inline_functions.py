"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines inline functions that are compiled for both GPU and CPU, and
used in the ionization code.
"""
import math
from scipy.constants import c, e, m_e

e_mc = e / ( m_e * c)

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
    gamma_p,
    Larmore_factor_density,
    Larmore_factor_momentum,
    SR_dxi, SR_xi_data,
    omega_ax, spect_loc
    ):

    d_omega = omega_ax[1] - omega_ax[0]
    N_omega_src = SR_xi_data.size

    gamma_p_inv = 1. / gamma_p

    # get normalized velocity
    beta_x = ux * gamma_p_inv
    beta_y = uy * gamma_p_inv
    beta_z = uz * gamma_p_inv

    # get momentum time derivative
    dt_ux = - e_mc * ( Ex + beta_y * cBz - beta_z * cBy )
    dt_uy = - e_mc * ( Ey + beta_z * cBx - beta_x * cBz )
    dt_uz = - e_mc * ( Ez + beta_x * cBy - beta_y * cBx )

    # get Lorentz factor derivative
    dt_gamma = - e_mc * (Ex * beta_x + Ey * beta_y + Ez * beta_z)

    # get acceleration
    dt_beta_x = (dt_ux - beta_x * dt_gamma) * gamma_p_inv
    dt_beta_y = (dt_uy - beta_y * dt_gamma) * gamma_p_inv
    dt_beta_z = (dt_uz - beta_z * dt_gamma) * gamma_p_inv

    # calculate Larmore energy
    Energy_norm = w * gamma_p**6 * (
         dt_beta_x**2 + dt_beta_y**2 + dt_beta_z**2 - \
         ( beta_y * dt_beta_z - beta_z * dt_beta_y )**2 - \
         ( beta_z * dt_beta_x - beta_x * dt_beta_z )**2 - \
         ( beta_x * dt_beta_y - beta_y * dt_beta_x )**2
        )

    # calculate emitted radiation momentum
    Momentum_Larmor = Larmore_factor_momentum * Energy_norm
    u_abs_inv = 1. / math.sqrt(ux * ux + uy * uy + uz * uz )
    ux_ph = Momentum_Larmor * ux * u_abs_inv
    uy_ph = Momentum_Larmor * uy * u_abs_inv
    uz_ph = Momentum_Larmor * uz * u_abs_inv

    # calculate critical frequency
    dt_beta_abs2 = dt_beta_x**2 + dt_beta_y**2 + dt_beta_z**2
    beta_abs2_inv = 1. / ( beta_x**2 + beta_y**2 + beta_z**2 )
    beta_dot_dt_beta = beta_x * dt_beta_x + beta_y * dt_beta_y + beta_z * dt_beta_z

    omega_c = 1.5 * gamma_p**3 * beta_abs2_inv  * \
        math.sqrt( dt_beta_abs2 - beta_dot_dt_beta**2 * beta_abs2_inv )

    # discard too low critical frequencies as not resolved
    # (still returning photon momentum, just in case)
    if (omega_c < 4 * d_omega):
        spect_loc[:] = 0.0
        return( spect_loc )

    omega_c_inv = 1. / omega_c

    Density_Larmore = Larmore_factor_density * Energy_norm  * omega_c_inv

    # Loop over the frequencies to project spectrum
    for i_omega in range(omega_ax.size):
        xi_loc = omega_ax[i_omega] * omega_c_inv
        s_ix_src = xi_loc / SR_dxi
        ix_src = math.floor( s_ix_src )

        if ( ix_src >= N_omega_src - 1 ):
            spect_loc[i_omega] = 0.0
        else:
            s1 = s_ix_src - ix_src
            s0 = 1.0 - s1
            ix_src_int = int(ix_src)
            S_xi_loc = SR_xi_data[ix_src_int] * s0 + SR_xi_data[ix_src_int+1] * s1
            spect_loc[i_omega] = Density_Larmore * S_xi_loc

    return( spect_loc )
