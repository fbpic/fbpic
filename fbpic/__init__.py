__version__ = '0.19.1'
__doc__ = """
Fourier-Bessel Particle-In-Cell code (FBPIC)

Usage
-----
See the fbpic.main.Simulation class to set up a simulation.
"""

# Change the default formatting for warnings within fbpic
import warnings
def modified_formatting(message, category, filename, lineno, line=None):
    """Format a warning so that the code line `line` is not shown`."""
    return('\n%s: %s:%s:\n%s\n'%(category.__name__, filename, lineno, message))
warnings.formatwarning = modified_formatting
