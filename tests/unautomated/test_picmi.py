# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It tests the picmi implementation, by downloading the standard PICMI
example file, and checking that fbpic can run it **without crashing**.
(i.e. this test does not check the accuracy of the results)

Usage:
This file is meant to be run from the top directory of fbpic,
by any of the following commands
$ python tests/test_picmi.py
$ py.test -q tests/test_picmi.py
$ python setup.py test
"""
import os
import re
import shutil

def test_picmi_script():
    """
    Download the picmi script and run it, making sure that it does not crash
    """
    n_MPI = 1
    temporary_dir = os.path.abspath('./tmp_test_dir')
    script_name = '../fbpic_script.py'

    # Create a temporary directory for the simulation
    # and copy the example script into this directory
    if os.path.exists( temporary_dir ):
        shutil.rmtree( temporary_dir )
    os.mkdir( temporary_dir )
    os.chdir( temporary_dir )

    # Shortcut for the script file, which is repeatedly changed
    script_filename = os.path.join( temporary_dir, script_name )

    # Read the script and modify it to use fbpic
    with open(script_filename) as f:
        script = f.read()
    script = replace_string( script, 'from .* import picmi',
                             'from fbpic import picmi')
    script = replace_string( script, 'step\(.*\)', 'step(10)')
    with open(script_filename, 'w') as f:
        f.write(script)

    # Launch the script from the OS
    command_line = 'cd %s' %temporary_dir
    if n_MPI == 1:
        command_line += '; python %s' %script_name
    else:
        # Use only one thread for multiple MPI
        command_line += '; NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 '
        command_line += 'mpirun -np %d python %s' %(n_MPI,script_name)
    response = os.system( command_line )
    assert response==0

    # Suppress the temporary directory
    shutil.rmtree( temporary_dir )

def replace_string( text, old_string, new_string ):
    """
    Check that `old_string` is in `text`, and replace it by `new_string`
    (`old_string` and `new_string` use regex syntax)
    """
    # Check that the target line is present
    if re.findall(old_string, text) == []:
        raise RuntimeError('Did not find expected string: %s' %old_string)
    # Return the modified text
    return re.sub( old_string, new_string, text )

def get_string( regex, text ):
    """
    Match `regex` in `text` and returns the corresponding group
    (`regex` should be a regex string that contains a captured group)
    """
    match = re.search( regex, text )
    return( match.groups(1)[0] )


if __name__ == '__main__':
    test_picmi_script()
