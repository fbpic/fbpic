"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It imports functions which are useful when initializing a laser pulse
"""
from .laser import add_laser, add_laser_pulse
from .laser_profiles import GaussianLaser, LaguerreGaussLaser, \
              DonutLikeLaguerreGaussLaser, FlattenedGaussianLaser, \
              FewCycleLaser

__all__ = ['add_laser', 'add_laser_pulse',
            'GaussianLaser', 'LaguerreGaussLaser',
            'DonutLikeLaguerreGaussLaser', 'FlattenedGaussianLaser',
            'FewCycleLaser']
