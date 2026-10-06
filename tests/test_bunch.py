# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Tests for Gaussian bunch initialization and saved beam distributions."""
import numpy as np
import pytest
from scipy.constants import c, e, m_e

from fbpic.main import Simulation
from fbpic.lpa_utils.bunch import add_particle_bunch_gaussian


@pytest.mark.parametrize('symmetrize', [False, True])
@pytest.mark.parametrize('sig_gamma', [0., 1.])
def test_gaussian_save_beam(tmp_path, symmetrize, sig_gamma):
    """Saved quantities must describe the same particles and momenta."""
    sim = Simulation(32, 20.e-6, 8, 10.e-6, 2, 1.e-15,
                     zmin=-20.e-6, use_cuda=False,
                     boundaries={'z': 'periodic', 'r': 'reflective'})
    filename = str(tmp_path / 'beam.npz')
    bunch = add_particle_bunch_gaussian(
        sim, q=-e, m=m_e, sig_r=1.e-6, sig_z=1.e-6, n_emit=1.e-6,
        gamma0=100., sig_gamma=sig_gamma, n_physical_particles=1.e6,
        n_macroparticles=32, tf=1.e-6/c, save_beam=filename,
        symmetrize=symmetrize, initialize_self_field=False)

    with np.load(filename) as beam:
        for name in ['x', 'y', 'z', 'ux', 'uy', 'uz', 'inv_gamma', 'w']:
            assert beam[name].shape == (32,)
            np.testing.assert_allclose(beam[name], getattr(bunch, name))
        np.testing.assert_allclose(
            beam['inv_gamma'],
            1. / np.sqrt(1. + beam['ux']**2 + beam['uy']**2 + beam['uz']**2))
        np.testing.assert_allclose(beam['w'].sum(), 1.e6)
