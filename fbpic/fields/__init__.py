"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It imports the Fields object from the fields package, so that
this object can be used at a higher level.
"""
from .fields import Fields, cuda_installed
__all__ = [ 'Fields', 'cuda_installed' ]
