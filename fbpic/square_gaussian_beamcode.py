import numpy as np

def square_gaussian_beam_density(x, y, z, n_b0, sigma_x, sigma_y, L_b):
    """
    Calculate the square beam density profile.

    Parameters
    ----------
    x, y, z : numpy arrays
        Coordinates of the simulation grid.
    n_b0 : float
        Peak density of the beam (particles/m^3).
    sigma_x, sigma_y : float
        R.m.s. transverse sizes of the beam.
    L_b : float
        Length of the beam.

    Returns
    -------
    n_b : numpy array
        Beam density at each grid point.
    """
    # Beam extent
    x_extent = 2 * sigma_x
    y_extent = 2 * sigma_y

    # Logical conditions for the beam region
    inside_x = np.abs(x) <= x_extent
    inside_y = np.abs(y) <= y_extent
    inside_z = (z >= 0) & (z <= L_b)
    
    # Gaussian transverse profile
    gaussian_profile = np.exp(-0.5 * ((x / sigma_x) ** 2 + (y / sigma_y) ** 2))


    # Apply the conditions
    n_b = np.where(inside_x & inside_y & inside_z, n_b0, 0)*gaussian_profile
    return n_b
