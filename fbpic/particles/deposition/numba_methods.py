# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the deposition methods for rho and J for linear and cubic
order shapes on the CPU with numba.
"""
import numba

@numba.njit
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
