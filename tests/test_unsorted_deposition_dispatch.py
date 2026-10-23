from types import SimpleNamespace

import numpy as np

import fbpic.particles.particles as particles_module
from fbpic.particles.particles import Particles


class FakeKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, launch_config):
        def launch(*args):
            self.calls.append((launch_config, args))
        return launch


def make_species(prefix_sum_shift=0):
    species = object.__new__(Particles)
    species.q = -1.0
    species.particle_shape = "linear"
    species.use_cuda = True
    species.sorted = False
    species.ionizer = None
    species.Ntot = 2
    species.deposit_tpb = 64
    species.unsorted_j_tpb = 64
    species.unsorted_rho_tpb = 64
    species.prefix_sum_shift = prefix_sum_shift
    species.prefix_sum = np.zeros(3, dtype=np.int64)
    species.cell_idx = np.zeros(2, dtype=np.int64)
    for name in ("x", "y", "z", "w", "ux", "uy", "uz", "inv_gamma"):
        setattr(species, name, np.ones(2))
    species.sort_calls = 0

    def sort_particles(fld):
        species.sort_calls += 1

    species.sort_particles = sort_particles
    return species


def make_fields():
    array = np.zeros((2, 2), dtype=np.complex128)
    grid = SimpleNamespace(
        invdz=1.0, zmin=0.0, Nz=2,
        invdr=1.0, rmin=0.0, Nr=2,
        Jr=array.copy(), Jt=array.copy(), Jz=array.copy(),
        rho=array.copy(), d_ruyten_linear_coef=np.ones(3))
    return SimpleNamespace(interp=[grid])


def install_fake_cuda(monkeypatch):
    kernels = {
        name: FakeKernel() for name in (
            "deposit_J_gpu_unsorted_momentum",
            "deposit_rho_gpu_unsorted",
            "deposit_J_gpu_linear_one_mode",
            "deposit_rho_gpu_linear_one_mode")
    }
    for name, kernel in kernels.items():
        monkeypatch.setattr(particles_module, name, kernel, raising=False)
    monkeypatch.setattr(
        particles_module, "cuda_tpb_bpg_1d",
        lambda work, TPB=None: (1, TPB or 64), raising=False)
    return kernels


def test_unsorted_current_bypasses_sort(monkeypatch):
    kernels = install_fake_cuda(monkeypatch)
    monkeypatch.setenv("FBPIC_EXPERIMENT_UNSORTED_J", "1")
    species = make_species()

    species.deposit(make_fields(), "J")

    assert species.sort_calls == 0
    assert len(kernels["deposit_J_gpu_unsorted_momentum"].calls) == 1
    assert kernels["deposit_J_gpu_unsorted_momentum"].calls[0][0] == (1, 64)
    assert not kernels["deposit_J_gpu_linear_one_mode"].calls


def test_unsorted_current_keeps_ionizer_sort(monkeypatch):
    kernels = install_fake_cuda(monkeypatch)
    monkeypatch.setenv("FBPIC_EXPERIMENT_UNSORTED_J", "1")
    species = make_species()
    species.ionizer = SimpleNamespace(w_times_level=np.ones(2))

    species.deposit(make_fields(), "J")

    assert species.sort_calls == 1
    assert len(kernels["deposit_J_gpu_unsorted_momentum"].calls) == 1
    assert not kernels["deposit_J_gpu_linear_one_mode"].calls


def test_unsorted_rho_requires_fresh_cell_map(monkeypatch):
    kernels = install_fake_cuda(monkeypatch)
    monkeypatch.setenv("FBPIC_EXPERIMENT_UNSORTED_RHO", "1")
    species = make_species(prefix_sum_shift=1)

    species.deposit(make_fields(), "rho")

    assert species.sort_calls == 1
    assert not kernels["deposit_rho_gpu_unsorted"].calls
    assert len(kernels["deposit_rho_gpu_linear_one_mode"].calls) == 1


def test_unsorted_rho_uses_fast_kernel_with_fresh_cell_map(monkeypatch):
    kernels = install_fake_cuda(monkeypatch)
    monkeypatch.setenv("FBPIC_EXPERIMENT_UNSORTED_RHO", "1")
    species = make_species(prefix_sum_shift=0)

    species.deposit(make_fields(), "rho")

    assert species.sort_calls == 1
    assert len(kernels["deposit_rho_gpu_unsorted"].calls) == 1
    assert kernels["deposit_rho_gpu_unsorted"].calls[0][0] == (1, 64)
    assert not kernels["deposit_rho_gpu_linear_one_mode"].calls
