# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It contains atomic data, to be used by the ionization module.
"""
import numpy as np
from scipy.constants import e

# The dictionary below defines the ionization energies in Joules
# (energies in eV are multiplied by e)
ionization_energies_dict = {
    'H':  e * np.array([ 13.6 ]),
    'He': e * np.array([ 24.5874, 54.417760 ]),
    'N':  e * np.array([ 14.534, 29.601, 47.448, 77.472, 97.888,
                        552.067, 667.046 ]),
    'Ar': e * np.array([ 15.759, 27.6, 40.7, 59.81, 75.02, 91.007,
                        124.319, 143.456, 422.60, 479.76, 540.4, 619.0,
                        685.47, 755.13, 855.47, 918.374, 4120.67, 4426.22 ])
}
