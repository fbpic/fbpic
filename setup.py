from setuptools import setup
import fbpic # In order to extract the version number

# Obtain the long description from README.md
with open('README.md') as f :
    long_description = f.read()

setup(
    name='fbpic',
    install_requires=['numpy', 'scipy', 'matplotlib',
                      'pyfftw', 'h5py', 'datetime' ],
    version=fbpic.__version__,
    author='Remi Lehe',
    author_email='remi.lehe@normalesup.org',
    packages=['fbpic', 'fbpic.particles', 'fbpic.fields', 'fbpic.openpmd_diag'],
    description='Fourier-Bessel Particle-In-Cell code',
    long_description = long_description,
    platforms='Linux, MacOSX',
    url='http://bitbucket.org/remilehe/fbpic'
)
