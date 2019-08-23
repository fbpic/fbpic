# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It makes sure that the example input scripts in `docs/source/example_input`
runs **without crashing**. It runs the following scripts:
- lwfa_script.py with a single proc and two proc, using checkpoint and restart
- boosted_frame_script.py with two procs
- parametric_script.py with two procs
**It does not actually check the validity of the physics involved.**

Usage:
This file is meant to be run from the top directory of fbpic,
by any of the following commands
$ python tests/test_example_docs_scripts.py
$ py.test -q tests/test_example_docs_scripts.py
$ python setup.py test
"""
import time
import os
import re
import shutil
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from openpmd_viewer.addons import LpaDiagnostics

def test_lpa_sim_singleproc_restart():
    "Test the example input script with one proc in `docs/source/example_input`"
    # In the tuples below, the float number indicates the relative tolerance
    # with which the fields are checked.
    # Fields are checked with a finite tolerance, because the
    # continuous injection does not occur exactly at the same time
    # in the original and restarted simulation, and results in small
    # differences in the simulations
    checked_fields = [ ('E', 'x', 2.e-5), ('E', 'z', 2.e-5),
                        ('B', 'y', 2.e-5), ('rho', None, 1.e-2) ]
    run_sim( 'lwfa_script.py', n_MPI=1, checked_fields=checked_fields )

def test_lpa_sim_twoproc_restart():
    "Test the example input script with two proc in `docs/source/example_input`"
    # In the tuples below, the float number indicates the relative tolerance
    # with which the fields are checked.
    # Fields are checked with a finite tolerance, because the
    # continuous injection does not occur exactly at the same time
    # in the original and restarted simulation, and results in small
    # differences in the simulations
    checked_fields = [ ('E', 'x', 2.e-5), ('E', 'z', 2.e-5),
                        ('B', 'y', 2.e-5), ('rho', None, 1.e-2) ]
    run_sim( 'lwfa_script.py', n_MPI=2, checked_fields=checked_fields,
             test_checkpoint_dir=True )

def test_ionization_script_twoproc():
    "Test the example script with two proc in `docs/source/example_input`"
    # In the tuples below, the float number indicates the relative tolerance
    # with which the fields are checked.
    # Ionization involves random events, which are not controlled by
    # numpy's seed ; therefore the tolerance (when checking the fields)
    # is lower than the previous cases.
    checked_fields = [ ('E', 'x', 1.e-3), ('E', 'z', 1.e-2),
                        ('B', 'y', 1.e-3), ('rho', None, 0.3) ]
    run_sim( 'ionization_script.py', n_MPI=2, checked_fields=checked_fields )

def run_sim( script_name, n_MPI, checked_fields, test_checkpoint_dir=False ):
    """
    Runs the script `script_name` from the folder docs/source/example_input,
    with `n_MPI` MPI processes. The simulation is then restarted with
    the same number of processes ; the code checks that the restarted results
    are identical.

    More precisely:
    - The first simulation is run for N_step, then the random seed is reset
        (for reproducibility) and the code runs for N_step more steps.
    - Then a second simulation is launched, which reruns the last N_step.
    """
    temporary_dir = './tests/tmp_test_dir'

    # Create a temporary directory for the simulation
    # and copy the example script into this directory
    if os.path.exists( temporary_dir ):
        shutil.rmtree( temporary_dir )
    os.mkdir( temporary_dir )
    shutil.copy('./docs/source/example_input/%s' %script_name,
                    temporary_dir )
    # Shortcut for the script file, which is repeatedly changed
    script_filename = os.path.join( temporary_dir, script_name )

    # Read the script and check
    with open(script_filename) as f:
        script = f.read()

    # Change default N_step, diag_period and checkpoint_period
    script = replace_string( script, 'N_step = .*', 'N_step = 200')
    script = replace_string( script,
        'diag_period = 50', 'diag_period = 10')
    script = replace_string( script,
        'checkpoint_period = 100', 'checkpoint_period = 50')

    # For MPI simulations: modify the script to use finite-order
    if n_MPI > 1:
        script = replace_string( script, 'n_order = -1', 'n_order = 16')
    # Modify the script so as to enable checkpoints
    script = replace_string( script, 'save_checkpoints = False',
                                'save_checkpoints = True')
    if test_checkpoint_dir:
        # Try to change the name of the checkpoint directory
        checkpoint_dir = './test_chkpt'
        script = replace_string( script,
            'set_periodic_checkpoint\( sim, checkpoint_period \)',
            'set_periodic_checkpoint( sim, checkpoint_period, checkpoint_dir="%s" )'%checkpoint_dir)
        script = replace_string( script, 'restart_from_checkpoint\( sim \)',
         'restart_from_checkpoint( sim, checkpoint_dir="%s" )'%checkpoint_dir)
    else:
        checkpoint_dir = './checkpoints'

    script = replace_string( script, 'track_electrons = False',
                                'track_electrons = True')
    # Modify the script to perform N_step, enforce the random seed
    # (should be the same when restarting, for exact comparison),
    # and perform again N_step.
    script = replace_string( script, 'sim.step\( N_step \)',
           'sim.step( N_step ); np.random.seed(0); sim.step( N_step )' )
    with open(script_filename, 'w') as f:
        f.write(script)

    # Launch the script from the OS
    command_line = 'cd %s' %temporary_dir
    if n_MPI == 1:
        command_line += '; python %s' %script_name
    else:
        # Use only one thread for multiple MPI
        command_line += '; NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 '
        command_line += 'mpirun -np %d python %s' %(n_MPI, script_name)
    response = os.system( command_line )
    assert response==0

    # Move diagnostics (for later comparison with the restarted simulation)
    shutil.move( os.path.join( temporary_dir, 'diags'),
                 os.path.join( temporary_dir, 'original_diags') )
    # Keep only the checkpoints from the first N_step
    N_step = int( get_string( 'N_step = (\d+)', script ) )
    period = int( get_string( 'checkpoint_period = (\d+)', script ) )
    for i_MPI in range(n_MPI):
        for step in range( N_step + period, 2*N_step + period, period ):
            os.remove( os.path.join( temporary_dir,
                     '%s/proc%d/hdf5/data%08d.h5' %(checkpoint_dir,i_MPI,step) ))

    # Modify the script so as to enable restarts
    script = replace_string( script, 'use_restart = False',
                                'use_restart = True')
    # Redo only the last N_step
    script = replace_string( script,
           'sim.step\( N_step \); np.random.seed\(0\); sim.step\( N_step \)',
           'np.random.seed(0); sim.step( N_step )',)
    with open(script_filename, 'w') as f:
        f.write(script)

    # Launch the modified script from the OS, with 2 proc
    response = os.system( command_line )
    assert response==0

    # Check that restarted simulation gives the same results
    # as the original simulation
    print('Checking restarted simulation...')
    start_time = time.time()
    ts1 = OpenPMDTimeSeries(
        os.path.join( temporary_dir, 'diags/hdf5') )
    ts2 = OpenPMDTimeSeries(
        os.path.join( temporary_dir, 'original_diags/hdf5') )
    compare_simulations( ts1, ts2, checked_fields )
    end_time = time.time()
    print( "%.2f seconds" %(end_time-start_time))

    # Check that the particle IDs are unique
    print('Checking particle ids...')
    start_time = time.time()
    for iteration in ts1.iterations:
        pid, = ts1.get_particle(["id"], iteration=iteration, species="electrons")
        assert len(np.unique(pid)) == len(pid)
    end_time = time.time()
    print( "%.2f seconds" %(end_time-start_time))

    # Suppress the temporary directory
    shutil.rmtree( temporary_dir )

def test_boosted_frame_sim_twoproc():
    "Test the example input script with two procs in `docs/source/example_input`"

    temporary_dir = './tests/tmp_test_dir'

    # Create a temporary directory for the simulation
    # and copy the example script into this directory
    if os.path.exists( temporary_dir ):
        shutil.rmtree( temporary_dir )
    os.mkdir( temporary_dir )
    shutil.copy('./docs/source/example_input/boosted_frame_script.py',
                    temporary_dir )
    # Shortcut for the script file, which is repeatedly changed
    script_filename = os.path.join( temporary_dir, 'boosted_frame_script.py' )

    # Read the script
    with open(script_filename) as f:
        script = f.read()

    # Change default N_step
    script = replace_string( script, 'N_step = .*', 'N_step = 101')

    # Modify the script so as to enable finite order
    script = replace_string( script, 'n_order = -1', 'n_order = 16')
    script = replace_string(script, 'track_bunch = False', 'track_bunch = True')
    with open(script_filename, 'w') as f:
        f.write(script)

    # Launch the script from the OS
    command_line = 'cd %s; NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 '%temporary_dir
    command_line += 'mpirun -np 2 python boosted_frame_script.py'
    response = os.system( command_line )
    assert response==0

    # Check that the particle ids are unique at each iterations
    ts = OpenPMDTimeSeries( os.path.join( temporary_dir, 'lab_diags/hdf5') )
    print('Checking particle ids...')
    start_time = time.time()
    for iteration in ts.iterations:
        pid, = ts.get_particle(["id"], iteration=iteration )
        assert len(np.unique(pid)) == len(pid)
    end_time = time.time()
    print( "%.2f seconds" %(end_time-start_time))

    # Suppress the temporary directory
    shutil.rmtree( temporary_dir )

def test_parametric_sim_twoproc():
    "Test the example input script with two proc in `docs/source/example_input`"

    temporary_dir = './tests/tmp_test_dir'

    # Create a temporary directory for the simulation
    # and copy the example script into this directory
    if os.path.exists( temporary_dir ):
        shutil.rmtree( temporary_dir )
    os.mkdir( temporary_dir )
    shutil.copy(
        './docs/source/example_input/parametric_script.py', temporary_dir )
    # Shortcut for the script file, which is repeatedly changed
    script_filename = os.path.join( temporary_dir, 'parametric_script.py' )

    # Read the script
    with open(script_filename) as f:
        script = f.read()

    # Change default N_step, diag_period and checkpoint_period
    script = replace_string( script, 'N_step = .*', 'N_step = 200')
    script = replace_string( script,
        'diag_period = 50', 'diag_period = 10')
    script = replace_string( script,
        'checkpoint_period = 100', 'checkpoint_period = 50')

    # Modify the script so as to enable checkpoints
    script = replace_string( script, 'save_checkpoints = False',
                                'save_checkpoints = True')
    with open(script_filename, 'w') as f:
        f.write(script)

    # Launch the modified script from the OS, with 2 proc
    command_line = 'cd %s; NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 '%temporary_dir
    command_line += 'mpirun -np 2 python parametric_script.py'
    response = os.system( command_line )
    assert response==0

    # Modify the script so as to enable restarts
    script = replace_string( script, 'use_restart = False',
                                'use_restart = True')
    script = replace_string( script, 'save_checkpoints = True',
                                'save_checkpoints = False')
    with open(script_filename, 'w') as f:
        f.write(script)
    # Launch the modified script from the OS, with 2 proc
    response = os.system( command_line )
    assert response==0

    # Check that the simulation produced two output directories
    # and that these directories correspond to different values of
    # a0 (this is done by checking the a0 of the laser, with openPMD-viewer)
    for a0 in [ 2.0, 4.0 ]:
        # Open the diagnotics
        diag_folder = os.path.join( temporary_dir, 'diags_a0_%.2f/hdf5' %a0 )
        ts = LpaDiagnostics( diag_folder )
        # Check that the value of a0 in the diagnostics is the
        # expected one.
        a0_in_diag = ts.get_a0( iteration=80, pol='x' )
        assert abs( (a0 - a0_in_diag)/a0 ) < 1.e-2

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

def compare_simulations( ts1, ts2, checked_fields ):
    """
    Compare the fields of the simulations `ts1` and `ts2` and
    make sure that they agree within a given precision
    """
    for iteration in ts1.iterations:
        for field, coord, tolerance in checked_fields:
            print( field, coord )
            F1, info1 = ts1.get_field( field, coord, iteration=iteration )
            F2, info2 = ts2.get_field( field, coord, iteration=iteration )
            # Fields may be shifted by one cell, because, due to round-off
            # errors, the moving window may not be called at the same time
            # in the original and restarted simulation.
            n_diff = int(round( (info2.zmin-info1.zmin)/info1.dz ))
            # Round-off errors in the moving window
            # can never create shifts of more than 1 cell
            assert abs(n_diff) <= 1
            # Restrict the fields to the overlapping part
            if n_diff > 0:
                F1 = F1[:,n_diff:]
                F2 = F2[:,:-n_diff]
            elif n_diff < 0:
                F2 = F2[:,abs(n_diff):]
                F1 = F1[:,:-abs(n_diff)]
            assert np.allclose( F1, F2, atol=tolerance*abs(F1).max() )


if __name__ == '__main__':
    test_boosted_frame_sim_twoproc()
    test_lpa_sim_twoproc_restart()
    test_ionization_script_twoproc()
    test_lpa_sim_singleproc_restart()
    test_parametric_sim_twoproc()
