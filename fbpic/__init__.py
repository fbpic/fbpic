__version__ = '0.8.0'
__doc__ = """
Fourier-Bessel Particle-In-Cell code (FBPIC)

Usage
-----
See the fbpic.main.Simulation class to set up a simulation.
"""

# Change the default formatting for warnings within fbpic
import warnings
warnings.showwarning = \
    lambda message, category, filename, lineno, *args, **kwargs: \
    print('\nWarning: %s:%s:\n%s\n' %(filename, lineno, message))
