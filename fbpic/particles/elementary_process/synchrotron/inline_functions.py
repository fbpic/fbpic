"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines inline functions that are compiled for both GPU and CPU, and
used in the syncrotron radiation code.
"""
import math
from scipy.constants import c, e, m_e

e_mc = e / ( m_e * c)

def get_angles( ux, uy, uz ):
    """
    Calculate angular projections of particle momentum vector on
    `(theta_x, theta_y)` plane.

    Parameters
    ----------
    ux, uy, uz: floats
        Components of particles normalized momentum.

    Returns
    -------
    theta_x, theta_y: floats
        angular projections of particle momentum vector in radians
    """

    theta_x = math.atan2( ux, uz )
    theta_y = math.atan2( uy, uz )

    return( theta_x, theta_y )

def get_linear_coefficients(x, xmin, dx):
    """
    Calculate shape coefficients and index for the 1D linear
    interpolation on a uniform grid (used for angle projections)

    Parameters
    ----------
    x: float
        Coordinate to be projected.

    xmin: float
         Grid origin

    dx: float
        Grid step

    Returns
    -------
    ix: integer
        Index of the cell that contains the point

    s0, s1: floats
        Weights projected to the left and right nodes of the cell
    """
    s_ix = ( x - xmin ) / dx
    ix = math.floor( s_ix )
    s1 = s_ix - ix
    s0 = 1.0 - s1
    ix = int(ix)

    return ix, s0, s1

def get_particle_radiation(
    ux, uy, uz, w,
    Ex, Ey, Ez,
    cBx, cBy, cBz,
    gamma_inv,
    Larmore_factor_density,
    Larmore_factor_momentum,
    SR_dxi, SR_xi_data,
    omega_ax, spect_loc
    ):

    """
    Calculate spectal energy distribution emitted by the particle.

    Parameters
    ----------
    ux, uy, uz, w: floats
        Components momentum and weight of the particle

    Ex, Ey, Ez: float
         Components of electric field on the particle (V/m)

    cBx, cBy, cBz: float
         Components of magnetic field on the particle multiplied by
         the speed of light (V/m)

    gamma_inv: float
        Reciprocal of particle Lorentz factor

    Larmore_factor_density: float
        Normalization factor for spectral-angular density,
        `e**2 * dt / (6 * np.pi * epsilon_0 * c * hbar * d_theta_x * d_theta_y)`

    Larmore_factor_momentum: float
        Normalization factor for the photon momentum,
        `e**2 * dt / ( 6 * np.pi * epsilon_0 * c**2 )`

    omega_ax: 1D vector of floats
        frequencies on which spectrum is calculated

    spect_loc: 1D vector of floats
        calculated spectral density of the radiation

    Returns
    -------
    spect_loc: 1D vector of floats
        calculated spectral density of the radiation
    """
    d_omega = omega_ax[1] - omega_ax[0]
    N_omega_src = SR_xi_data.size
    gamma = 1. / gamma_inv

    # get normalized velocity
    beta_x = ux * gamma_inv
    beta_y = uy * gamma_inv
    beta_z = uz * gamma_inv

    # get momentum time derivative
    dt_ux = - e_mc * ( Ex + beta_y * cBz - beta_z * cBy )
    dt_uy = - e_mc * ( Ey + beta_z * cBx - beta_x * cBz )
    dt_uz = - e_mc * ( Ez + beta_x * cBy - beta_y * cBx )

    # get Lorentz factor derivative
    dt_gamma = - e_mc * (Ex * beta_x + Ey * beta_y + Ez * beta_z)

    # get acceleration
    dt_beta_x = (dt_ux - beta_x * dt_gamma) * gamma_inv
    dt_beta_y = (dt_uy - beta_y * dt_gamma) * gamma_inv
    dt_beta_z = (dt_uz - beta_z * dt_gamma) * gamma_inv

    # calculate Larmore energy
    Energy_norm = w * gamma**6 * (
         dt_beta_x**2 + dt_beta_y**2 + dt_beta_z**2 - \
         ( beta_y * dt_beta_z - beta_z * dt_beta_y )**2 - \
         ( beta_z * dt_beta_x - beta_x * dt_beta_z )**2 - \
         ( beta_x * dt_beta_y - beta_y * dt_beta_x )**2
        )

    # calculate emitted radiation momentum
    Momentum_Larmor = Larmore_factor_momentum * Energy_norm / w
    u_abs_inv = 1. / math.sqrt(ux * ux + uy * uy + uz * uz )
    # or
    # u_abs_inv = 1. / math.sqrt(1 + gamma*gamma )
    ux_ph = Momentum_Larmor * ux * u_abs_inv
    uy_ph = Momentum_Larmor * uy * u_abs_inv
    uz_ph = Momentum_Larmor * uz * u_abs_inv

    # calculate critical frequency
    dt_beta_abs2 = dt_beta_x**2 + dt_beta_y**2 + dt_beta_z**2
    beta_abs2_inv = 1. / ( beta_x**2 + beta_y**2 + beta_z**2 )
    beta_dot_dt_beta = beta_x * dt_beta_x + beta_y * dt_beta_y + beta_z * dt_beta_z

    omega_c = 1.5 * gamma**3 * beta_abs2_inv  * \
        math.sqrt( dt_beta_abs2 - beta_dot_dt_beta**2 * beta_abs2_inv )

    # discard too low critical frequencies as not resolved
    if (omega_c < 4 * d_omega):
        spect_loc[:] = 0.0
        return( spect_loc, 0.0, 0.0, 0.0 )

    omega_c_inv = 1. / omega_c

    Density_Larmore = Larmore_factor_density * Energy_norm  * omega_c_inv

    # Loop over the frequencies to project the spectrum
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

    return( spect_loc, ux_ph, uy_ph, uz_ph )
