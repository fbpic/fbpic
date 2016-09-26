# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It makes sure that the example input scripts in `docs/source/example_input`
runs **without crashing**. It runs the following scripts:
- lwfa_script.py with a single proc
- boosted_frame_script.py with a single proc
- lwfa_script.py with two proc
**It does not actually check the validity of the physics involved.**

Usage:
This file is meant to be run from the top directory of fbpic,
by any of the following commands
$ python tests/test_example_docs_script.py
$ py.test -q tests/test_example_docs_script.py
$ python setup.py test
"""
import os
import shutil

def test_lpa_sim_singleproc():
    "Test the example input script with one proc in `docs/source/example_input`"

    temporary_dir = './tests/tmp_test_dir'

    # Create a temporary directory for the simulation
    # and copy the example script into this directory
    if os.path.exists( temporary_dir ):
        shutil.rmtree( temporary_dir )
    os.mkdir( temporary_dir )
    shutil.copy('./docs/source/example_input/lwfa_script.py',
                    temporary_dir )

    # Enter the temporary directory and run the script
    os.chdir( temporary_dir )
    # Launch the script from the OS
    response = os.system( 'python lwfa_script.py' )
    assert response==0

    # Exit the temporary directory and suppress it
    os.chdir('../../')
    shutil.rmtree( temporary_dir )

def test_boosted_frame_sim_singleproc():
    "Test the example input script with one proc in `docs/source/example_input`"

    temporary_dir = './tests/tmp_test_dir'

    # Create a temporary directory for the simulation
    # and copy the example script into this directory
    if os.path.exists( temporary_dir ):
        shutil.rmtree( temporary_dir )
    os.mkdir( temporary_dir )
    shutil.copy('./docs/source/example_input/boosted_frame_script.py',
                    temporary_dir )

    # Enter the temporary directory and run the script
    os.chdir( temporary_dir )
    # Launch the script from the OS
    response = os.system( 'python boosted_frame_script.py' )
    assert response==0

    # Exit the temporary directory and suppress it
    os.chdir('../../')
    shutil.rmtree( temporary_dir )

def test_lpa_sim_twoproc():
    "Test the example input script with two proc in `docs/source/example_input`"
    temporary_dir = './tests/tmp_test_dir'

    # Create a temporary directory for the simulation
    # and copy the example script into this directory
    if os.path.exists( temporary_dir ):
        shutil.rmtree( temporary_dir )
    os.mkdir( temporary_dir )
    shutil.copy('./docs/source/example_input/lwfa_script.py', temporary_dir )

    # Enter the temporary directory and run the script
    os.chdir( temporary_dir )
    # Launch the script from the OS, with 2 proc
    response = os.system( 'mpirun -np 2 python lwfa_script.py' )
    assert response==0

    # Exit the temporary directory and suppress it
    os.chdir('../../')
    shutil.rmtree( temporary_dir )

def test_lpa_sim_twoproc_restart():
    "Test the checkpoint/restart mechanism with two proc"
    temporary_dir = './tests/tmp_test_dir'

    # Create a temporary directory for the simulation
    # and copy the example script into this directory
    if os.path.exists( temporary_dir ):
        shutil.rmtree( temporary_dir )
    os.mkdir( temporary_dir )
    shutil.copy('./docs/source/example_input/lwfa_script.py', temporary_dir )

    # Enter the temporary directory
    os.chdir( temporary_dir )

    # Read the script and check that the targeted lines are present
    with open('lwfa_script.py') as f:
        script = f.read()
        # Check that the targeted lines are present
        if script.find('save_checkpoints = False') == -1 \
            or script.find('use_restart = False') == -1 \
            or script.find('N_step = 200') == -1:
            raise RuntimeError('Did not find expected lines in lwfa_script.py')

    # Modify the script so as to enable checkpoints
    script = script.replace('save_checkpoints = False',
                                'save_checkpoints = True')
    script = script.replace('N_step = 200', 'N_step = 101')
    with open('lwfa_script.py', 'w') as f: 
        f.write(script)
    # Launch the modified script from the OS, with 2 proc
    response = os.system( 'mpirun -np 2 python lwfa_script.py' )
    assert response==0

    # Modify the script so as to enable restarts
    script = script.replace('use_restart = False',
                                'use_restart = True')
    script = script.replace('save_checkpoints = True',
                                'save_checkpoints = False')
    with open('lwfa_script.py', 'w') as f: 
        f.write(script)
    # Launch the modified script from the OS, with 2 proc
    response = os.system( 'mpirun -np 2 python lwfa_script.py' )
    assert response==0

   # Exit the temporary directory and suppress it
    os.chdir('../../')
    shutil.rmtree( temporary_dir )
    
if __name__ == '__main__':
    test_lpa_sim_singleproc()
    test_boosted_frame_sim_singleproc()
    test_lpa_sim_twoproc()
    test_lpa_sim_twoproc_restart()
