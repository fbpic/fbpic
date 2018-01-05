# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines inline functions that are compiled for both GPU and CPU, and
used in the gathering kernels.
"""
def add_linear_gather_for_mode( m,
    Fr, Ft, Fz, exptheta_m, Fr_grid, Ft_grid, Fz_grid,
    iz_lower, iz_upper, ir_lower, ir_upper,
    S_ll, S_lu, S_lg, S_ul, S_uu, S_ug ):
    """
    Add the contribution of the gathered field from azimuthal mode `m` to the
    fields felt by one macroparticle (`Fr`, `Ft`, `Fz`), using linear weights.

    Parameters:
    -----------
    m: int
        The index of the azimuthal mode that is added.

    Fr, Ft, Fz: floats
        The fields felt by one macroparticle, which represent either E or B
        (before the contribution of mode `m` has been added)

    exptheta_m: complex
        The complex azimuthal factor $e^{-i m \theta}$ where $\theta$ is
        the azimuthal position of the macroparticle considered.

    Fr_grid, Ft_grid, Fz_grid: 2darrays of complexs
        The fields on the interpolation grid for mode `m`

    iz_lower, iz_upper, ir_lower, ir_upper: ints
        Lower and upper index in z and r from which the macroparticle
        considered should gather the fields (in the arrays F*_grid)

    S_ll, S_lu, S_lg, S_ul, S_uu, S_ug: floats
        The weights with which the fields are gathered, for the macroparticle
        considered. `S_lg` and `S_ug` are used for fields gathered from below
        the axis.

    Returns:
    --------
    Fr, Ft, Fz: floats
        The fields felt by one macroparticle, which represent either E or B
        (after the contribution of mode `m` has been added)
    """
    # Create temporary variables
    # for the "per mode" gathering
    Fr_m = 0.j
    Ft_m = 0.j
    Fz_m = 0.j
    # Lower cell in z, Lower cell in r
    Fr_m += S_ll * Fr_grid[ iz_lower, ir_lower ]
    Ft_m += S_ll * Ft_grid[ iz_lower, ir_lower ]
    Fz_m += S_ll * Fz_grid[ iz_lower, ir_lower ]
    # Lower cell in z, Upper cell in r
    Fr_m += S_lu * Fr_grid[ iz_lower, ir_upper ]
    Ft_m += S_lu * Ft_grid[ iz_lower, ir_upper ]
    Fz_m += S_lu * Fz_grid[ iz_lower, ir_upper ]
    # Upper cell in z, Lower cell in r
    Fr_m += S_ul * Fr_grid[ iz_upper, ir_lower ]
    Ft_m += S_ul * Ft_grid[ iz_upper, ir_lower ]
    Fz_m += S_ul * Fz_grid[ iz_upper, ir_lower ]
    # Upper cell in z, Upper cell in r
    Fr_m += S_uu * Fr_grid[ iz_upper, ir_upper ]
    Ft_m += S_uu * Ft_grid[ iz_upper, ir_upper ]
    Fz_m += S_uu * Fz_grid[ iz_upper, ir_upper ]
    # Add the fields from the guard cells
    if ir_lower == ir_upper == 0:
        flip_factor = (-1.)**m
        # Lower cell in z
        Fr_m += -flip_factor * S_lg * Fr_grid[ iz_lower, 0]
        Ft_m += -flip_factor * S_lg * Ft_grid[ iz_lower, 0]
        Fz_m +=  flip_factor * S_lg * Fz_grid[ iz_lower, 0]
        # Upper cell in z
        Fr_m += -flip_factor * S_ug * Fr_grid[ iz_upper, 0]
        Ft_m += -flip_factor * S_ug * Ft_grid[ iz_upper, 0]
        Fz_m +=  flip_factor * S_ug * Fz_grid[ iz_upper, 0]
    # Add the contribution from mode m to Fr, Ft, Fz
    # (Take into account factor 2 in the definition of azimuthal modes)
    if m == 0:
        factor = 1.
    else:
        factor = 2.
    Fr += factor*(Fr_m*exptheta_m).real
    Ft += factor*(Ft_m*exptheta_m).real
    Fz += factor*(Fz_m*exptheta_m).real

    return(Fr, Ft, Fz)


def add_cubic_gather_for_mode( m,
    Fr, Ft, Fz, exptheta_m, Fr_grid, Ft_grid, Fz_grid,
    ir, iz, Sr, Sz ):
    """
    Add the contribution of the gathered field from azimuthal mode `m` to the
    fields felt by one macroparticle (`Fr`, `Ft`, `Fz`), using cubic weights.

    Parameters:
    -----------
    m: int
        The index of the azimuthal mode that is added.

    Fr, Ft, Fz: floats
        The fields felt by one macroparticle, which represent either E or B
        (before the contribution of mode `m` has been added)

    exptheta_m: complex
        The complex azimuthal factor $e^{-i m \theta}$ where $\theta$ is
        the azimuthal position of the macroparticle considered.

    Fr_grid, Ft_grid, Fz_grid: 2darrays of complexs
        The fields on the interpolation grid for mode `m`

    ir, iz: 1darrays containing 4 ints
        The indices in r and z from which the macroparticle
        considered should gather the fields (in the arrays F*_grid)

    Sr, Sz: 1darrays containing 4 floats
        The weights in r and z with which the macroparticle
        considered should gather the fields (in the arrays F*_grid)

    Returns:
    --------
    Fr, Ft, Fz: floats
        The fields felt by one macroparticle, which represent either E or B
        (after the contribution of mode `m` has been added)
    """
    # Create temporary variables
    # for the "per mode" gathering
    Fr_m = 0.j
    Ft_m = 0.j
    Fz_m = 0.j
    # Add the fields for mode 0
    for index_r in range(4):
        for index_z in range(4):
            weight = Sz[index_z]*Sr[index_r]
            if weight < 0:
                flip_factor = (-1)**m
                Fr_m +=  flip_factor*weight*Fr_grid[iz[index_z], ir[index_r]]
                Ft_m +=  flip_factor*weight*Ft_grid[iz[index_z], ir[index_r]]
                Fz_m += -flip_factor*weight*Fz_grid[iz[index_z], ir[index_r]]
            else:
                Fr_m += weight*Fr_grid[iz[index_z], ir[index_r]]
                Ft_m += weight*Ft_grid[iz[index_z], ir[index_r]]
                Fz_m += weight*Fz_grid[iz[index_z], ir[index_r]]
    # Add the contribution from mode m to Fr, Ft, Fz
    # (Take into account factor 2 in the definition of azimuthal modes)
    if m == 0:
        factor = 1.
    else:
        factor = 2.
    Fr += factor*(Fr_m*exptheta_m).real
    Ft += factor*(Ft_m*exptheta_m).real
    Fz += factor*(Fz_m*exptheta_m).real

    return(Fr, Ft, Fz)
