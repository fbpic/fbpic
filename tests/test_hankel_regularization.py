"""Regression checks for the experimental matrix-Hankel SVD cutoff."""
import numpy as np
import pytest

from fbpic.fields.spectral_transform.hankel import DHT


def test_resolved_low_mode_profile_at_fine_resolution():
    """Blocking regression: a 50-um Gaussian is resolved by 400 radial cells."""
    for cutoff in [None, 1.e-3]:
        dht = DHT(0, 1, 1200, 1, 150.e-6, rcond=1.e-15 if cutoff is None else cutoff,
                  rcond_mminus=cutoff)
        profile = np.exp(-(dht.r/50.e-6)**2)[None, :].astype(complex)
        spectral, reconstructed = np.empty_like(profile), np.empty_like(profile)
        dht.transform(profile, spectral)
        dht.inverse_transform(spectral, reconstructed)
        error = np.sqrt(np.sum(dht.r*abs(reconstructed-profile)**2) /
                        np.sum(dht.r*abs(profile)**2))
        assert error < 1.e-6, f'Resolved m=1 field error {error:g}, cutoff={cutoff}'


@pytest.mark.parametrize('m', [0, 1, 5, 11])
@pytest.mark.parametrize('offset', [-1, 0, 1])
def test_disabled_matches_upstream_arithmetic(m, offset):
    dht = DHT(m+offset, m, 32, 1, 150.e-6, rcond=1.e-15, rcond_mminus=None)
    if m == 0 or offset == -1:
        expected = np.linalg.inv(dht.invM)
    else:
        expected = np.zeros_like(dht.M)
        expected[:, 1:] = np.linalg.pinv(dht.invM[1:], rcond=1.e-15)
    np.testing.assert_array_equal(dht.M, expected)


@pytest.mark.parametrize('cutoff', [-1., 1., float('nan'), float('inf')])
@pytest.mark.parametrize('name', ['rcond', 'rcond_mminus'])
def test_invalid_cutoff_rejected(name, cutoff):
    with pytest.raises(ValueError, match='cutoffs'):
        DHT(0, 1, 32, 1, 1., **{name: cutoff})


@pytest.mark.parametrize('m', [0, 1])
@pytest.mark.parametrize('offset', [-1, 0, 1])
def test_low_modes_unchanged(m, offset):
    regularized = DHT(m+offset, m, 64, 1, 1., rcond=1.e-3, rcond_mminus=1.e-3)
    original = DHT(m+offset, m, 64, 1, 1., rcond=1.e-15, rcond_mminus=None)
    np.testing.assert_array_equal(regularized.M, original.M)


@pytest.mark.parametrize('m', [4, 8, 11])
@pytest.mark.parametrize('offset', [0, 1])
def test_high_mode_pseudoinverse_gain_is_bounded(m, offset):
    dht = DHT(m+offset, m, 64, 1, 1., rcond=1.e-3)
    gain = np.linalg.norm(dht.M, 2) * np.linalg.norm(dht.invM[1:], 2)
    assert gain <= 1.e3 * (1.+1.e-10)
    np.testing.assert_array_equal(dht.M[:, 0], 0.)


@pytest.mark.parametrize('m', [1, 4, 8, 11])
def test_extra_zero_mode_is_unchanged(m):
    regularized = DHT(m-1, m, 64, 1, 1., rcond=1.e-3, rcond_mminus=None)
    original = DHT(m-1, m, 64, 1, 1., rcond=1.e-15, rcond_mminus=None)
    np.testing.assert_array_equal(regularized.M, original.M)
    np.testing.assert_array_equal(regularized.invM, original.invM)


@pytest.mark.parametrize('m', range(1, 12))
@pytest.mark.parametrize('offset', [-1, 0, 1])
def test_smooth_regular_profile(m, offset):
    p = m+offset
    dht = DHT(p, m, 64, 1, 1., rcond=1.e-3, rcond_mminus=1.e-3)
    sigma = 0.08
    profile = (dht.r/sigma)**p * np.exp(-0.5*(dht.r/sigma)**2)
    scale = profile.max()
    profile = (profile/scale)[np.newaxis, :].astype(complex)
    analytic = (2*np.pi*sigma**2 * (2*np.pi*sigma*dht.nu)**p *
                np.exp(-0.5*(2*np.pi*sigma*dht.nu)**2)/scale)
    spectral = np.empty_like(profile)
    reconstructed = np.empty_like(profile)
    dht.transform(profile, spectral)
    dht.inverse_transform(spectral, reconstructed)
    assert np.max(np.abs(reconstructed-profile)) < 0.01
    # For p=m-1, index 0 is the extra algebraic mode, not a Hankel sample.
    assert np.max(np.abs(spectral[0, 1:]-analytic[1:]))/analytic.max() < 0.03


@pytest.mark.parametrize('m', range(4))
def test_constrained_inverse_leaves_resolved_low_modes_unchanged(m):
    original = DHT(m-1, m, 75, 1, 150.e-6, rcond=1.e-15, rcond_mminus=None)
    constrained = DHT(m-1, m, 75, 1, 150.e-6, rcond_mminus=1.e-3)
    np.testing.assert_array_equal(constrained.M, original.M)


@pytest.mark.parametrize('nr', [32, 75, 128])
@pytest.mark.parametrize('m', [4, 5, 8, 11])
def test_constrained_inverse_preserves_zero_mode_and_is_a_projection(nr, m):
    original = DHT(m-1, m, nr, 1, 150.e-6, rcond=1.e-15, rcond_mminus=None)
    constrained = DHT(m-1, m, nr, 1, 150.e-6, rcond_mminus=1.e-3)
    # Keeping the backward basis preserves its analytic curl/div relations.
    np.testing.assert_array_equal(constrained.invM, original.invM)
    np.testing.assert_allclose(constrained.invM[0] @ constrained.M,
                               np.eye(nr)[0], atol=1.e-12)
    projection = constrained.M @ constrained.invM
    np.testing.assert_allclose(projection @ projection, projection, atol=1.e-12)
    # Row-vector projector: self-adjoint in cylindrical volume, not plain L2.
    scale = np.sqrt(2*np.pi*constrained.r*(constrained.rmax/nr))
    normalized = projection*scale[None, :]/scale[:, None]
    np.testing.assert_allclose(normalized.T, normalized, atol=1.e-12)
    np.testing.assert_allclose(normalized @ normalized, normalized, atol=1.e-12)
    assert np.linalg.norm(normalized, 2) <= 1.+1.e-12


@pytest.mark.parametrize('nr', [32, 75, 128])
def test_filter_no_longer_amplifies_unresolved_axis_current(nr):
    from fbpic.fields.smoothing import BinomialSmoother
    original = DHT(4, 5, nr, 1, 150.e-6, rcond=1.e-15, rcond_mminus=None)
    constrained = DHT(4, 5, nr, 1, 150.e-6, rcond_mminus=1.e-3)
    _, radial_filter = BinomialSmoother().get_filter_array(
        np.array([0.]), 2*np.pi*original.nu, 1., 150.e-6/nr)
    source = np.eye(nr)[0]
    old = (source @ original.M * radial_filter) @ original.invM
    new = (source @ constrained.M * radial_filter) @ constrained.invM
    assert np.linalg.norm(old) > 10.
    assert np.linalg.norm(new) < 0.01
