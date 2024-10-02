"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It imports the ParticleSpin object, for evolving particle spins in a
simulation.
"""
from .spin_tracker import SpinTracker
__all__ = ['SpinTracker']
