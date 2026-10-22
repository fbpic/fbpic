"""The opt-in reaches evolution, bunch space charge and direct laser injection."""
import numpy as np
import pytest
from scipy.constants import e, m_e

from fbpic.main import Simulation
from fbpic.fields.spectral_transform.hankel import DHT
from fbpic.lpa_utils.bunch import add_particle_bunch
from fbpic.lpa_utils.laser import add_laser_pulse
from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser
import fbpic.lpa_utils.bunch as bunch_module
import fbpic.lpa_utils.laser.direct_injection as laser_module


def make_sim(cutoff):
    return Simulation(16,4.e-6,32,8.e-6,6,1.e-16,use_cuda=False,
                      hankel_rcond=cutoff,verbose_level=0)


def test_public_option_off_is_upstream():
    sim = make_sim(None)
    assert sim.fld.hankel_rcond is None
    for m, trans in enumerate(sim.fld.trans):
        for p,d in [(m,trans.dht0),(m+1,trans.dhtp),(m-1,trans.dhtm)]:
            old = DHT(p,m,32,16,8.e-6,rcond=1.e-15,rcond_mminus=None)
            np.testing.assert_array_equal(d.M,old.M)


def test_public_option_and_initialization_helpers(monkeypatch):
    sim = make_sim(1.e-3)
    for m,trans in enumerate(sim.fld.trans):
        for p,d in [(m,trans.dht0),(m+1,trans.dhtp),(m-1,trans.dhtm)]:
            expected = DHT(p,m,32,16,8.e-6,rcond=1.e-3,rcond_mminus=1.e-3)
            np.testing.assert_array_equal(d.M,expected.M)
    seen = []
    for module in [bunch_module,laser_module]:
        original = module.Fields
        def record(*args, _original=original, **kwargs):
            seen.append(kwargs['hankel_rcond'])
            return _original(*args,**kwargs)
        monkeypatch.setattr(module,'Fields',record)
    add_particle_bunch(sim,-e,m_e,10.,1.e20,1.e-6,2.e-6,0.,2.e-6,
                       p_nz=2,p_nr=2,p_nt=12)
    pulse = GaussianLaser(.01,2.e-6,2.e-15,1.e-6)
    add_laser_pulse(sim,pulse,method='direct')
    assert seen == [1.e-3,1.e-3]
    assert all(np.isfinite(g.Ez).all() for g in sim.fld.interp)


@pytest.mark.parametrize('cutoff',[-1.,1.,float('nan')])
def test_invalid_public_option(cutoff):
    with pytest.raises(ValueError,match='cutoffs'):
        make_sim(cutoff)
