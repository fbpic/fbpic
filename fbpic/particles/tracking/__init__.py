"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It imports the ParticleTracker object, which is used in order to tag
particles with ids.
"""

from .tracking import ParticleTracker
__all__ = ['ParticleTracker']
