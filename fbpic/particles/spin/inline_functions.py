# Copyright 2023, FBPIC contributors
# Author: Michael J. Quin
# Scientific supervision: Matteo Tamburini
# Code optimization: Kristjan Poder
# License: 3-Clause-BSD-LBNL
"""
This file is for use with the Fourier-Bessel Particle-In-Cell code (FB-PIC).

It defines a function to push the 'classical' spin vector, according to the
Bargmann-Michel-Telegdi equation and other spin-push related functions.
"""
import math
from scipy.constants import c


def push_s_BMT(sx_i, sy_i, sz_i, ux_i, uy_i, uz_i, ux_f, uy_f, uz_f,
               Ex, Ey, Ez, Bx, By, Bz, tauconst, anom):
    """
    Push spin of a single particle using the 'Tamburini' pusher, as
    detailed in https://arxiv.org/abs/2303.16966
    """
    # Momentum at a half-timestep forward using Boris assumption
    ux = (ux_i + ux_f)/2
    uy = (uy_i + uy_f)/2
    uz = (uz_i + uz_f)/2

    # Assume 1./gamma at midpoint can be calculated from (average)
    # momentum at midpoint
    inv_gamma = 1./math.sqrt( 1. + ux**2 + uy**2 + uz**2 )
    # Define parameter: 1./(1+gamma)
    inv_1pgamma = inv_gamma/(1. + inv_gamma)
    # Scalar product of momentum and B field
    uB = (ux*Bx + uy*By + uz*Bz)

    # Define E/B field dependent parameters: tau and upsilon (ups).
    # Tau vector is analogous to the 'tau' (or 't') vector used in
    # the Boris-Vay pusher.
    taux = tauconst*( (anom + inv_gamma)*Bx
                     - (anom + inv_1pgamma)*(uy*Ez - uz*Ey)*inv_gamma/c
                     - anom*inv_gamma*inv_1pgamma*uB*ux )
    tauy = tauconst*( (anom + inv_gamma)*By
                     - (anom + inv_1pgamma)*(uz*Ex - ux*Ez)*inv_gamma/c
                     - anom*inv_gamma*inv_1pgamma*uB*uy )
    tauz = tauconst*( (anom + inv_gamma)*Bz
                     - (anom + inv_1pgamma)*(ux*Ey - uy*Ex)*inv_gamma/c
                     - anom*inv_gamma*inv_1pgamma*uB*uz )

    tau2 = taux**2 + tauy**2 + tauz**2

    # ---------- Tamburini Method ---------------------------------------------
    ups = 1/(1. + tau2)
    stau = sx_i*taux + sy_i*tauy + sz_i*tauz

    # New spin components
    sx_f = ups*( sx_i + 2*(sy_i*tauz - sz_i*tauy)
                   + stau*taux + (sz_i*taux - sx_i*tauz)*tauz
                   - (sx_i*tauy - sy_i*taux)*tauy )
    sy_f = ups*( sy_i + 2*(sz_i*taux - sx_i*tauz)
                   + stau*tauy + (sx_i*tauy - sy_i*taux)*taux
                   - (sy_i*tauz - sz_i*tauy)*tauz )
    sz_f = ups*( sz_i + 2*(sx_i*tauy - sy_i*taux)
                   + stau*tauz + (sy_i*tauz - sz_i*tauy)*tauy
                   - (sz_i*taux - sx_i*tauz)*taux )
    # -------------------------------------------------------------------------

    return sx_f, sy_f, sz_f


def copy_ionized_electron_spin_batch(
        i_batch, batch_size, elec_old_Ntot, ion_Ntot,
        cumulative_n_ionized, i_level, ionized_from,
        store_electrons_per_level,
        elec_sx, elec_sy, elec_sz,
        ion_sx, ion_sy, ion_sz,
        rand_sx, rand_sy, rand_sz):
    # Electron index: this is incremented each time
    # an ionized electron is identified
    elec_index = elec_old_Ntot + cumulative_n_ionized[i_level, i_batch]
    # Loop through the ions in this batch
    N_max = min((i_batch + 1) * batch_size, ion_Ntot)
    for ion_index in range(i_batch * batch_size, N_max):

        # Determine if a new electron should be created from this ion
        create_electron = False
        # If electrons are not distinguished by level, take all ionized ions
        if (not store_electrons_per_level) and ionized_from[ion_index] >= 0:
            create_electron = True
        # If electrons are to be distinguished by level, take only those
        # of the corresponding i_level
        if store_electrons_per_level and ionized_from[ion_index] == i_level:
            create_electron = True

        if create_electron:
            if ionized_from[ion_index] == 0:
                # Copy the ion data to the current electron_index
                elec_sx[elec_index] = ion_sx[ion_index]
                elec_sy[elec_index] = ion_sy[ion_index]
                elec_sz[elec_index] = ion_sz[ion_index]
            else:
                # Or use one of the random spins
                rand_index = elec_index - elec_old_Ntot
                elec_sx[elec_index] = rand_sx[rand_index]
                elec_sy[elec_index] = rand_sy[rand_index]
                elec_sz[elec_index] = rand_sz[rand_index]

            # Update the electron_index
            elec_index += 1
