# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It tests the space charge initialization routines, and makes sure that these
routines produce identical results for both serial and parallel simulations.

Usage:
This file is meant to be run from the top directory of fbpic,
by any of the following commands
$ python tests/test_example_docs_scripts.py
$ py.test -q tests/test_example_docs_scripts.py
$ python setup.py test
"""
import os
import shutil
import numpy as np
from opmd_viewer import OpenPMDTimeSeries

def run_sim_serial_and_parallel( script_file, data_file=None ):
    """Copy the script `script_file` from the `unautomated directory`
    and run the simulation both in serial and paralllel.

    Then compare the results."""

    temporary_dir = './tests/tmp_test_dir'
    origin_dir = './tests/unautomated'

    # Create a temporary directory for the simulation
    # and copy the testing script into this directory
    if os.path.exists( temporary_dir ):
        shutil.rmtree( temporary_dir )
    os.mkdir( temporary_dir )
    shutil.copy( os.path.join(origin_dir, script_file),
                os.path.join(temporary_dir, script_file) )
    if data_file is not None:
        shutil.copy( os.path.join(origin_dir, data_file),
                os.path.join(temporary_dir, data_file) )

    # Launch the script from the OS, in serial
    response = os.system(
        'cd %s; python %s' %(temporary_dir, script_file) )
    assert response==0
    # Rename the diags folder
    shutil.move( os.path.join(temporary_dir, 'diags/'),
                os.path.join(temporary_dir, 'diags_serial/') )

    # Launch the script from the OS, in parallel
    response = os.system(
        'cd %s; mpirun -np 2 python %s' %(temporary_dir, script_file) )
    assert response==0
    # Rename the diags folder
    shutil.move( os.path.join(temporary_dir, 'diags/'),
                os.path.join(temporary_dir, 'diags_parallel/') )

    # Check that the results of the initial space charge
    # calculation are identical
    check_identical_fields(
        os.path.join(temporary_dir, 'diags_serial/hdf5/'),
        os.path.join(temporary_dir, 'diags_parallel/hdf5/') )

    # Suppress the temporary directory
    shutil.rmtree( temporary_dir )

def check_identical_fields( folder1, folder2 ):
    ts1 = OpenPMDTimeSeries( folder1 )
    ts2 = OpenPMDTimeSeries( folder2 )
    # Check the vector fields
    for field in ["J", "E", "B"]:
        for coord in ["r", "t", "z"]:
            print("Checking %s%s" %(field, coord))
            field1, info = ts1.get_field(field, coord, iteration=0)
            field2, info = ts2.get_field(field, coord, iteration=0)
            # For 0 fields, do not use allclose
            if abs(field1).max() == 0:
                assert abs(field2).max() == 0
            else:
                assert np.allclose(
                    field1/abs(field1).max(), field2/abs(field2).max() )
    # Check the rho field
    print("Checking rho")
    field1, info = ts1.get_field("rho", iteration=0)
    field2, info = ts2.get_field("rho", iteration=0)
    assert np.allclose( field1/abs(field1).max(), field2/abs(field2).max() )

def test_bunch_from_file():
    run_sim_serial_and_parallel( 'test_space_charge_file.py',
                                'test_space_charge_file_data.txt')

def test_bunch_gaussian():
    run_sim_serial_and_parallel( 'test_space_charge_gaussian.py')

if __name__ == '__main__':
    test_bunch_gaussian()
    test_bunch_from_file()
