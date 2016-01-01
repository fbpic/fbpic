"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It makes sure that the example input script in `docs/example_input`
runs without crashing. It does not actually check the validity of the
physics involved.

Usage:
This file is meant to be run from the root directory of fbpic,
by any of the following commands
$ python tests/test_example_docs_script.py
$ py.test --ignore=tests/unautomated
$ python setup.py test
"""
import os
import shutil

def test_example_input():
    "Test the example input script in `docs/example_input`"

    temporary_dir = './tests/tmp_test_dir'

    # Create a temporary directory for the simulation
    # and copy the example script into this directory
    if os.path.exists( temporary_dir ):
        shutil.rmtree( temporary_dir )
    os.mkdir( temporary_dir )
    shutil.copy('./docs/example_input/lpa_sim.py', temporary_dir )

    # Enter the temporary directory and run the script
    os.chdir( temporary_dir )
    # The globals command make sure that the package which
    # are imported within the script can be used here.
    execfile( 'lpa_sim.py', globals(), globals() )

    # Exit the temporary directory and suppress it
    os.chdir('../../')
    shutil.rmtree( temporary_dir )

if __name__ == '__main__':
    test_example_input()
