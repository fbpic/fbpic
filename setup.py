#!/bin/env python

# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters, Soeren Jalas
# License: 3-Clause-BSD-LBNL
from setuptools import setup, find_packages
import fbpic # In order to extract the version number

# Obtain the long description from README.md
with open('./README.md') as f:
    long_description = f.read()
# Get the package requirements from the requirements.txt file
with open('requirements.txt') as f:
    install_requires = [ line.strip('\n') for line in f.readlines() ]

setup(
    name='fbpic',
    version=fbpic.__version__,
    description='Spectral, quasi-3D Particle-In-Cell for CPU and GPU',
    long_description=long_description,
    long_description_content_type='text/markdown',
    maintainer='Remi Lehe',
    maintainer_email='remi.lehe@normalesup.org',
    license='BSD-3-Clause-LBNL',
    packages=find_packages('.'),
    install_requires=install_requires,
    extras_require = {
        'picmi':  ["picmistandard", "numexpr", "periodictable"],
    },
    include_package_data=True,
    platforms='any',
    url='http://github.com/fbpic/fbpic',
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'],
    )
