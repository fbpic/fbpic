"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It makes sure that the example input scripts in `docs/example_input`
runs **without crashing**. It runs the following scripts:
- lpa_sim.py with a single proc
- boosted_frame_sim.py with a single proc
- lpa_sim.py with two proc
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
    "Test the example input script with one proc in `docs/example_input`"

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
    exec( open('lpa_sim.py').read(), globals(), globals() )

    # Exit the temporary directory and suppress it
    os.chdir('../../')
    shutil.rmtree( temporary_dir )

def test_boosted_frame_sim_singleproc():
    "Test the example input script with one proc in `docs/example_input`"

    temporary_dir = './tests/tmp_test_dir'

    # Create a temporary directory for the simulation
    # and copy the example script into this directory
    if os.path.exists( temporary_dir ):
        shutil.rmtree( temporary_dir )
    os.mkdir( temporary_dir )
    shutil.copy('./docs/example_input/boosted_frame_sim.py', temporary_dir )

    # Enter the temporary directory and run the script
    os.chdir( temporary_dir )
    # The globals command make sure that the package which
    # are imported within the script can be used here.
    exec( open('boosted_frame_sim.py').read(), globals(), globals() )

    # Exit the temporary directory and suppress it
    os.chdir('../../')
    shutil.rmtree( temporary_dir )

def test_lpa_sim_twoproc():
    "Test the example input script with two proc in `docs/example_input`"
    temporary_dir = './tests/tmp_test_dir'

    # Create a temporary directory for the simulation
    # and copy the example script into this directory
    if os.path.exists( temporary_dir ):
        shutil.rmtree( temporary_dir )
    os.mkdir( temporary_dir )
    shutil.copy('./docs/example_input/lpa_sim.py', temporary_dir )

    # Enter the temporary directory and run the script
    os.chdir( temporary_dir )
    # Launch the script from the OS, with 2 proc
    response = os.system( 'mpirun -np 2 python lpa_sim.py' )
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
    shutil.copy('./docs/example_input/lpa_sim.py', temporary_dir )

    # Enter the temporary directory
    os.chdir( temporary_dir )

    # Read the script and check that the targeted lines are present
    with open('lpa_sim.py') as f:
        script = f.read()
        # Check that the targeted lines are present
        if script.find('save_checkpoints = False') == -1 \
            or script.find('use_restart = False') == -1 \
            or script.find('N_step = 200') == -1:
            raise RuntimeError('Did not find expected lines in lpa_sim.py')

    # Modify the script so as to enable checkpoints
    script = script.replace('save_checkpoints = False',
                                'save_checkpoints = True')
    script = script.replace('N_step = 200', 'N_step = 101')
    with open('lpa_sim.py', 'w') as f: 
        f.write(script)
    # Launch the modified script from the OS, with 2 proc
    response = os.system( 'mpirun -np 2 python lpa_sim.py' )
    assert response==0

    # Modify the script so as to enable restarts
    script = script.replace('use_restart = False',
                                'use_restart = True')
    script = script.replace('save_checkpoints = True',
                                'save_checkpoints = False')
    with open('lpa_sim.py', 'w') as f: 
        f.write(script)
    # Launch the modified script from the OS, with 2 proc
    response = os.system( 'mpirun -np 2 python lpa_sim.py' )
    assert response==0

   # Exit the temporary directory and suppress it
    os.chdir('../../')
    shutil.rmtree( temporary_dir )
    
if __name__ == '__main__':
    test_lpa_sim_singleproc()
    test_boosted_frame_sim_singleproc()
    test_lpa_sim_twoproc()
    test_lpa_sim_twoproc_restart()
