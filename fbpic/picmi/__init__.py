# Copyright 2019, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the picmi interface
"""
# Check that the `picmistandard` package has been installed
try:
    import picmistandard
except ImportError:
    raise ImportError(
        "In order to use FBPIC with PICMI, you should install the \n"
        "`picmistandard` package, e.g. with: `pip install picmistandard`")

# Define general variables that each PICMI code should define
codename = 'fbpic'

# Import picmi objects
from picmistandard import PICMI_CylindricalGrid as CylindricalGrid
from picmistandard import PICMI_Species as Species
from picmistandard import PICMI_MultiSpecies as MultiSpecies
MultiSpecies.Species_class = Species # Set the Species class, as required by PICMI
from picmistandard import PICMI_GaussianLaser as GaussianLaser
from picmistandard import PICMI_GaussianBunchDistribution as GaussianBunchDistribution
from picmistandard import PICMI_AnalyticDistribution as AnalyticDistribution
from picmistandard import PICMI_BinomialSmoother as BinomialSmoother
from picmistandard import PICMI_ElectromagneticSolver as ElectromagneticSolver


from .simulation import Simulation
