# Copyright 2017, FBPIC contributors
# Authors: Kristjan Poder
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It tests the restart of the spin-tracker, with and without
ionizer. Without ionization, the spin vectors are checked to be the
same within some precision. For ionization, we check that the simulations run.
"""
import numpy as np
import shutil
import time
import os
import re
from openpmd_viewer import OpenPMDTimeSeries


def run_sim(script_name, n_MPI=1):
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
    if os.path.exists(temporary_dir):
        shutil.rmtree(temporary_dir)
    os.mkdir(temporary_dir)
    shutil.copy('./docs/source/example_input/%s' %script_name,
                temporary_dir)
    # Shortcut for the script file, which is repeatedly changed
    script_filename = os.path.join(temporary_dir, script_name)

    # Read the script and check
    with open(script_filename) as f:
        script = f.read()

    # Change default N_step, diag_period and checkpoint_period
    script = replace_string(script, 'N_step = .*', 'N_step = 200')
    script = replace_string(script,
        'diag_period = 50', 'diag_period = 10')
    script = replace_string(script,
        'checkpoint_period = 100', 'checkpoint_period = 50')

    # For MPI simulations: modify the script to use finite-order
    if n_MPI > 1:
        script = replace_string(script, 'n_order = -1', 'n_order = 16')
    # Modify the script so as to enable checkpoints
    script = replace_string(script, 'save_checkpoints = False',
                                'save_checkpoints = True')

    checkpoint_dir = './checkpoints'

    script = replace_string( script, 'track_electrons = False',
                                'track_electrons = True')
    # Modify the script to perform N_step, enforce the random seed
    # (should be the same when restarting, for exact comparison),
    # and perform again N_step.
    script = replace_string( script, 'sim.step\( N_step \)',
           'sim.step( N_step ); np.random.seed(0); sim.step( N_step )' )

    # Activate the spin tracking
    if script_name == 'lwfa_script.py':
        script = replace_string(script, '# Load initial fields',
                      'elec.activate_spin_tracking(sz_m=1., anom=0.)')
    elif script_name == 'ionization_script.py':
        script = re.sub('(\s*)(\S*) = sim.add_new_species\(([\s\S\n]*?)\)',
               '\g<1>\g<2> = sim.add_new_species(\g<3>)\g<1>\g<2>.activate_spin_tracking(sz_m=1., anom=0.)',
               script)
    else:
        raise ValueError('File %s unknown!' % script_name)

    with open(script_filename, 'w') as f:
        f.write(script)

    # Launch the script from the OS
    command_line = 'cd %s' % temporary_dir
    if n_MPI == 1:
        command_line += '; python %s' % script_name
    else:
        # Use only one thread for multiple MPI
        command_line += '; NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 '
        command_line += 'mpirun -np %d python %s' %(n_MPI, script_name)
    response = os.system(command_line)
    assert response == 0

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
    assert response == 0

    # Check that restarted simulation gives the same results
    # as the original simulation
    print('Checking restarted simulation...')
    start_time = time.time()
    ts1 = OpenPMDTimeSeries(
        os.path.join( temporary_dir, 'diags/hdf5') )
    ts2 = OpenPMDTimeSeries(
        os.path.join( temporary_dir, 'original_diags/hdf5') )

    checked_species = [('electrons', 1e-3)]
    if script_name == 'lwfa_script.py':
        compare_spin_vectors( ts1, ts2, checked_species)
    elif script_name == 'ionization_script.py':
        checked_species += [('electrons from N', 0.2)]
        compare_polarisations(ts1, ts2, checked_species, 'z')

    end_time = time.time()
    print( "%.2f seconds" %(end_time-start_time))

    # Check that the particle IDs are unique
    print('Checking particle ids...')
    start_time = time.time()
    for iteration in ts1.iterations:
        pid, = ts1.get_particle(["id"], iteration=iteration,
                                species="electrons")
        assert len(np.unique(pid)) == len(pid)
    end_time = time.time()
    print( "%.2f seconds" %(end_time-start_time))

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


def test_spin_restart():
    run_sim('lwfa_script.py', n_MPI=1)


def test_spin_restart_twoproc():
    run_sim('lwfa_script.py', n_MPI=2)


def test_spin_ionization_restart():
    """Function that is run by pytest"""
    run_sim('ionization_script.py', n_MPI=1)


def test_spin_ionization_restart_twoproc():
    run_sim('ionization_script.py', n_MPI=2)


def compare_spin_vectors(ts1, ts2, checked_species):
    """
    Compare the spin vectors of the simulations `ts1` and `ts2`
    and make sure that they agree within a given precision
    """
    for iteration in ts1.iterations:
        for species, tolerance in checked_species:
            for var in ['spin/x', 'spin/y', 'spin/z']:
                s1 = ts1.get_particle([var, 'id'], species=species,
                                      iteration=iteration)
                s2 = ts2.get_particle([var, 'id'], species=species,
                                      iteration=iteration)

                # Get the same IDs
                ids = s1[1] if len(s1[0]) < len(s2[0]) else s2[1]

                s1_common = s1[0][np.in1d(s1[1], ids)]
                s2_common = s2[0][np.in1d(s2[1], ids)]
                assert np.allclose(s1_common, s2_common, atol=tolerance)


def compare_polarisations(ts1, ts2, checked_species, pol_component):
    """ Check the polarisations of species in the two simulations. """
    for iteration in ts1.iterations:
        for species, tolerance in checked_species:
            var = 'spin/'+pol_component
            s1 = ts1.get_particle([var, 'w'], species=species,
                                  iteration=iteration)
            s2 = ts2.get_particle([var, 'w'], species=species,
                                  iteration=iteration)

            p1 = np.sum(s1[0] * s1[1]) / s1[1].sum()
            p2 = np.sum(s2[0] * s2[1]) / s2[1].sum()
            print(species, iteration, pol_component, p1, p2, p1 / p2)
            assert np.isclose(p1, p2, rtol=tolerance)


if __name__ == '__main__':
    test_spin_restart()
    test_spin_restart_twoproc()
    test_spin_ionization_restart()
    test_spin_ionization_restart_twoproc()
