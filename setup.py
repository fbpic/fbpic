from setuptools import setup, find_packages
import fbpic # In order to extract the version number

# Obtain the long description from README.md
with open('README.md') as f :
    long_description = f.read()

# Get the package requirements from the requirements.txt file
with open('requirements.txt') as f:
    install_requires = [ line.strip('\n') for line in f.readlines() ]
# pyfftw is not included in the requirements.txt since it causes
# a bug when doing conda install --file requirements.txt
install_requires.append('pyfftw')

setup(
    name='fbpic',
    install_requires=install_requires,
    version=fbpic.__version__,
    author='Remi Lehe',
    author_email='remi.lehe@normalesup.org',
    packages = find_packages('./'),
    description='Fourier-Bessel Particle-In-Cell code',
    long_description = long_description,
    platforms='Linux, MacOSX',
    url='http://bitbucket.org/remilehe/fbpic',
    tests_require=['pytest'],
    setup_requires=['pytest-runner']
)
