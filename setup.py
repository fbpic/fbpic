from setuptools import setup, find_packages
import fbpic # In order to extract the version number

# Obtain the long description from README.md
with open('README.md') as f :
    long_description = f.read()
# Get the package requirements from the requirements.txt file
with open('requirements.txt') as f:
    install_requires = [ line.strip('\n') for line in f.readlines() ]

setup(
    name='fbpic',
    version=fbpic.__version__,
    description='Fourier-Bessel Particle-In-Cell code',
    long_description=long_description,
    maintainer='Remi Lehe',
    maintainer_email='remi.lehe@normalesup.org',
    license='BSD-3-Clause-LBNL',
    packages=find_packages('./'),
    tests_require=['pytest', 'openpmd_viewer'],
    setup_requires=['pytest-runner'],
    install_requires=install_requires,
    platforms='any',
    url='http://github.com/FBPIC/FBPIC',
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'],
    )
