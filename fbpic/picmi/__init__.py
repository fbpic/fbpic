# Copyright 2019, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the picmi interface
"""
# Define general variables that each PICMI code should define
codename = 'fbpic'

# Check that the `picmistandard` package has been installed
try:
    from picmistandard.base import register_codename
    register_codename(codename)
except ImportError:
    raise ImportError(
        "In order to use FBPIC with PICMI, you should install the \n"
        "`picmistandard` package, e.g. with: `pip install picmistandard`")

# Import picmi objects
# - Constants
from . import constants
# - For general setup
from picmistandard import PICMI_CylindricalGrid as CylindricalGrid
from picmistandard import PICMI_BinomialSmoother as BinomialSmoother
from picmistandard import PICMI_ElectromagneticSolver as ElectromagneticSolver
# - For the species
from picmistandard import PICMI_Species as Species
from picmistandard import PICMI_MultiSpecies as MultiSpecies
MultiSpecies.Species_class = Species # Set the Species class, as required by PICMI
# - For laser initialization
from picmistandard import PICMI_LaserAntenna as LaserAntenna
from picmistandard import PICMI_GaussianLaser as GaussianLaser
# - For particle initialization
from picmistandard import PICMI_GriddedLayout as GriddedLayout
from picmistandard import PICMI_PseudoRandomLayout as PseudoRandomLayout
from picmistandard import PICMI_GaussianBunchDistribution as GaussianBunchDistribution
from picmistandard import PICMI_AnalyticDistribution as AnalyticDistribution
from picmistandard import PICMI_UniformDistribution as UniformDistribution
# - For diagnostics
from picmistandard import PICMI_FieldDiagnostic as FieldDiagnostic
from picmistandard import PICMI_ParticleDiagnostic as ParticleDiagnostic

# Import the PICMI Simulation object redefined in FBPIC
from .simulation import Simulation

__all__ = [ 'codename', 'Simulation', 'CylindricalGrid', 'BinomialSmoother',
    'ElectromagneticSolver', 'Species', 'MultiSpecies', 'LaserAntenna',
    'GaussianLaser', 'GriddedLayout', 'PseudoRandomLayout',
    'GaussianBunchDistribution', 'AnalyticDistribution', 'UniformDistribution',
    'FieldDiagnostic', 'ParticleDiagnostic', 'constants' ]
