"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines inline functions that are compiled for both GPU and CPU, and
used in the ionization code.
"""
import math

# ----------------------------
# Function for field amplitude
# ----------------------------
def get_E_amplitude( ux, uy, uz, Ex, Ey, Ez, cBx, cBy, cBz ):
    """
    For one given macroparticle, calculate the amplitude of the E field
    *in the rest frame of the macroparticle*.

    (This is necessary so that the ADK probability is fully Lorentz invariant
    and thus also works in a boosted frame.)
    """
    u_dot_E = ux*Ex + uy*Ey + uz*Ez
    gamma = math.sqrt( 1 + ux**2 + uy**2 + uz**2 )

    E2_on_particle = - (u_dot_E)**2 \
        + ( gamma*Ex + uy*cBz - uz*cBy )**2 \
        + ( gamma*Ey + uz*cBx - ux*cBz )**2 \
        + ( gamma*Ez + ux*cBy - uy*cBx )**2

    return( math.sqrt( E2_on_particle ), gamma )

# ----------------------------
# Function for ADK probability
# ----------------------------
def get_ionization_probability( E, gamma, prefactor, power, exp_prefactor ):
    """
    For one given macroparticle, calculate the ADK probability that the
    particle is ionized during this timestep.

    (The proper time of the particle is used, so that the ADK probability
    is fully Lorentz invariant and thus also works in a boosted frame.)
    """
    # Avoid singular expression for E = 0
    if E == 0:
        return(0)
    # The gamma factor takes into account the fact that the ionization
    # rate is multiplied by dtau (proper time of the ion), i.e. dt/gamma
    w_dtau = 1./ gamma * prefactor * E**power * math.exp( exp_prefactor/E )
    p = 1. - math.exp( - w_dtau )
    return( p )

# -----------------
# Copying functions
# -----------------

def copy_ionized_electrons_batch(
    i_batch, batch_size, elec_old_Ntot, ion_Ntot,
    cumulative_n_ionized, ionized_from,
    i_level, store_electrons_per_level,
    elec_x, elec_y, elec_z, elec_inv_gamma,
    elec_ux, elec_uy, elec_uz, elec_w,
    elec_Ex, elec_Ey, elec_Ez, elec_Bx, elec_By, elec_Bz,
    ion_x, ion_y, ion_z, ion_inv_gamma,
    ion_ux, ion_uy, ion_uz, ion_w,
    ion_Ex, ion_Ey, ion_Ez, ion_Bx, ion_By, ion_Bz ):
    """
    Create the new electrons by copying the properties (position, momentum,
    etc) of the ions that they originate from.

    Particles are handled by batch: this functions goes through one batch
    of ion macroparticles and checks which macroparticles have been ionized
    during the present timestep (using the flag `ionized_from`, which
    is -1 if the ion is not ionized, and has the value of the original level
    otherwise).
    In order to know at which position, in the electron array, this data
    should be copied, the cumulated number of electrons `cumulative_n_ionized`
    (one element per batch) is used.
    """
    # Electron index: this is incremented each time
    # an ionized electron is identified
    elec_index = elec_old_Ntot + cumulative_n_ionized[i_level, i_batch]
    # Loop through the ions in this batch
    N_max = min( (i_batch+1)*batch_size, ion_Ntot )
    for ion_index in range( i_batch*batch_size, N_max ):

        # Determine if a new electron should be created from this ion
        create_electron = False
        # If electrons are not distinguished by level, take all ionized ions
        if (not store_electrons_per_level) and ionized_from[ion_index] >=0:
            create_electron = True
        # If electrons are to be distinguished by level, take only those
        # of the corresponding i_level
        if store_electrons_per_level and ionized_from[ion_index] == i_level:
            create_electron = True

        if create_electron:
            # Copy the ion data to the current electron_index
            elec_x[elec_index] = ion_x[ion_index]
            elec_y[elec_index] = ion_y[ion_index]
            elec_z[elec_index] = ion_z[ion_index]
            elec_ux[elec_index] = ion_ux[ion_index]
            elec_uy[elec_index] = ion_uy[ion_index]
            elec_uz[elec_index] = ion_uz[ion_index]
            elec_inv_gamma[elec_index] = ion_inv_gamma[ion_index]
            elec_w[elec_index] = ion_w[ion_index]
            elec_Ex[elec_index] = ion_Ex[ion_index]
            elec_Ey[elec_index] = ion_Ey[ion_index]
            elec_Ez[elec_index] = ion_Ez[ion_index]
            elec_Bx[elec_index] = ion_Bx[ion_index]
            elec_By[elec_index] = ion_By[ion_index]
            elec_Bz[elec_index] = ion_Bz[ion_index]

            # Update the electron_index
            elec_index += 1
