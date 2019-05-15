# Copyright 2019, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the charge and mass of standard particle types
"""
from scipy.constants import e, m_e, m_p
# Note: this should eventually be part of the picmistandard package

particle_charge = {
    'electron': -e,
    'positron': e,
    'H': 0.,
    'He': 0.,
    'N': 0.,
    'Ar': 0.
}

particle_mass = {
    'electron': m_e,
    'positron': m_e,
    'H': m_p,
    'He': 4.*m_p,
    'N': 14.*m_p,
    'Ar': 40.*m_p
}
