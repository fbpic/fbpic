"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines inline functions that are compiled for both GPU and CPU, and
used in the ionization code.
"""
import math
import numba
from scipy.constants import e
try:
    from numba import cuda
    cuda_installed = cuda.is_available()
except ImportError:
    cuda_installed = False

# ----------------------------
# Function for field amplitude
# ----------------------------
def get_E_amplitude( ux, uy, uz, Ex, Ey, Ez, cBx, cBy, cBz ):
    """
    # TODO
    """
    u_dot_E = ux*Ex + uy*Ey + uz*Ez
    gamma = math.sqrt( 1 + ux**2 + uy**2 + uz**2 )

    E2_on_particle = - (u_dot_E)**2 \
        + ( gamma*Ex + uy*cBz - uz*cBy )**2 \
        + ( gamma*Ey + uz*cBx - ux*cBz )**2 \
        + ( gamma*Ez + ux*cBy - uy*cBx )**2

    return( math.sqrt( E2_on_particle ) )

# Compile the function for CPU and GPU
if cuda_installed:
    get_E_amplitude_cuda = cuda.jit(get_E_amplitude, device=True, inline=True)
get_E_amplitude_numba = numba.jit(get_E_amplitude, nopython=True)

# ----------------------------
# Function for ADK probability
# ----------------------------
def get_ionization_probability( E, prefactor, power, exp_prefactor ):
    """
    TO DO
    """
    w_dt = prefactor * E**power * math.exp( exp_prefactor/E )
    p = 1. - math.exp( - w_dt )
    return( p )

# Compile the function for CPU and GPU
if cuda_installed:
    get_ionization_probability_cuda = \
        cuda.jit(get_ionization_probability, device=True, inline=True)
get_ionization_probability_numba = \
        numba.jit(get_ionization_probability, nopython=True)

# -----------------
# Copying functions
# -----------------

def copy_ionized_electrons_batch(
    i_batch, N_batches, elec_old_Ntot, ion_Ntot, n_ionized, is_ionized,
    elec_x, elec_y, elec_z, elec_ux, elec_uy, elec_uz, elec_w,
    ion_x, ion_y, ion_z, ion_ux, ion_uy, ion_uz, ion_neutral_weight ):
    """
    TO DO
    """
    # Electron index: this is incremented each time
    # an ionized electron is identified
    elec_index = elec_old_Ntot + n_ionized[i_batch]
    # Loop through the ions in this batch
    N_max = min( (i_batch+1)*N_batches, ion_Ntot )
    for ion_index in range( i_batch*N_batches, N_max ):

        if is_ionized[ion_index] == 1:
            # Copy the ion data to the current electron_index
            elec_x[elec_index] = ion_x[ion_index]
            elec_y[elec_index] = ion_y[ion_index]
            elec_z[elec_index] = ion_z[ion_index]
            elec_ux[elec_index] = ion_ux[ion_index]
            elec_uy[elec_index] = ion_uy[ion_index]
            elec_uz[elec_index] = ion_uz[ion_index]
            elec_w[elec_index] = - e * ion_neutral_weight[ion_index]

            # Update the electron_index
            elec_index += 1

# Compile the function for CPU and GPU
if cuda_installed:
    copy_ionized_electrons_batch_cuda = \
        cuda.jit(copy_ionized_electrons_batch, device=True, inline=True)
copy_ionized_electrons_batch_numba = \
        numba.jit(copy_ionized_electrons_batch, nopython=True)
