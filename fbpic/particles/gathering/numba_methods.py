# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the field gathering methods linear and cubic order shapes 
on the CPU with numba.
"""
import numba

@numba.jit(nopython=True)
def gather_field_numba(exptheta, m, Fgrid, Fptcl,
                       iz, ir, Sz, Sr, sign_guards):
    """
    Perform the weighted sum using numba

    Parameters
    ----------
    exptheta : 1darray of complexs
        (one element per macroparticle)
        Contains exp(-im theta) for each macroparticle

    m : int
        Index of the mode.
        Determines wether a factor 2 should be applied

    Fgrid : 2darray of complexs
        Contains the fields on the interpolation grid,
        from which to do the gathering

    Fptcl : 1darray of floats
        (one element per macroparticle)
        Contains the fields for each macroparticle
        Is modified by this function

    iz, ir : 2darray of ints
        Arrays of shape (shape_order+1, Ntot)
        where Ntot is the number of macroparticles
        Contains the index of the cells that each macroparticle
        will gather from.

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

    # Loop over the particles
    for ip in range(Ntot):
        # Erase the temporary variable
        F = 0.j
        # Loop over all the adjacent cells (given by shape order)
        # Use helper variables `ir_corr` and `Sr_corr`.
        # This is necessary, because ir and Sr should **not** be modified
        # **in-place**. (This is because ir and Sr are reused several
        # times, as we call the present function 3 times, with different
        # values for sign_guards.)
        for cell_index_r in range(ir.shape[0]):
            for cell_index_z in range(iz.shape[0]):
                # Correct the guard cell index and sign
                if ir[cell_index_r, ip] < 0:
                    ir_corr = abs(ir[cell_index_r, ip]) - 1
                    Sr_corr = sign_guards * Sr[cell_index_r, ip]
                else:
                    ir_corr = ir[cell_index_r, ip]
                    Sr_corr = Sr[cell_index_r, ip]
                # Gather the field value at the respective grid point
                F += Sz[cell_index_z, ip] * Sr_corr * \
                    Fgrid[ iz[cell_index_z, ip], ir_corr]

        # Add the complex phase
        if m == 0:
            Fptcl[ip] += (F * exptheta[ip]).real
        if m > 0:
            Fptcl[ip] += 2 * (F * exptheta[ip]).real