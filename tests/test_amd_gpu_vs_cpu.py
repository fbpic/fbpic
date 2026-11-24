# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""
Acceptance tests for the FBPIC AMD ROCm / HIP backend (§6.1 - §6.5).
"""
import os
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from numba import njit
try:
    import numba.cuda.random as ncr
except ImportError:
    ncr = None
from scipy.constants import c, e, m_e
from scipy.stats import ks_2samp

from fbpic.main import Simulation
from fbpic.fields.spectral_transform.fourier import FFT
from fbpic.fields.spectral_transform.hankel import DHT
from fbpic.lpa_utils.bunch import add_elec_bunch
from fbpic.lpa_utils.external_fields import ExternalField
from fbpic.lpa_utils.laser import add_laser_pulse, GaussianLaser
from fbpic.utils.cuda import (
    cuda, cuda_installed, gpu_backend, cupy, thrust,
    cuda_tpb_bpg_1d, cuda_tpb_bpg_2d,
    get_copy_tpb_2d, get_spectral_tpb_2d,
    get_deposit_tpb, get_gather_tpb,
    compile_cupy,
)
from fbpic.utils.random_gpu import (
    create_xoroshiro128p_states,
    xoroshiro128p_uniform_float64,
    xoroshiro128p_uniform_float64_cpu,
    xoroshiro128p_uniform_float32,
    xoroshiro128p_uniform_float32_cpu,
    xoroshiro128p_normal_float64,
    xoroshiro128p_normal_float64_cpu,
)
from fbpic.utils.random_seed import set_random_seed


def _require_gpu(hip_only=False):
    """Require GPU when explicitly requested, or skip on CPU-only CI."""
    explicit = os.environ.get('FBPIC_GPU_BACKEND', '').strip().lower()
    if not cuda_installed:
        if explicit in ('hip', 'cuda'):
            pytest.fail("GPU backend was requested but is unavailable")
        pytest.skip("No GPU backend installed (CPU-only environment)")
    if hip_only and gpu_backend != 'hip':
        if explicit == 'hip':
            pytest.fail("HIP backend was requested but is unavailable")
        pytest.skip("HIP-specific test skipped on non-HIP backend")


def _rel_l2(a, b):
    denom = np.linalg.norm(b)
    if denom == 0.0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


def _run_linear_wakefield(use_cuda, particle_shape, Nm, seed=1234, n_steps=40):
    set_random_seed(seed)
    Nz = 128
    zmin, zmax = -40.e-6, 40.e-6
    Nr = 48
    rmax = 40.e-6
    dt = (zmax - zmin) / Nz / c

    sim = Simulation(
        Nz, zmax, Nr, rmax, Nm, dt,
        zmin=zmin, n_order=-1,
        particle_shape=particle_shape,
        boundaries={'z': 'periodic', 'r': 'reflective'},
        initialize_ions=False,
        use_cuda=use_cuda,
        verbose_level=0,
    )
    elec = sim.add_new_species(
        q=-e, m=m_e, n=1.e24,
        p_zmin=-15.e-6, p_zmax=35.e-6, p_rmax=30.e-6,
        p_nz=2, p_nr=2, p_nt=4 * max(Nm, 2),
    )
    elec.track(sim.comm)
    laser = GaussianLaser(
        a0=0.3, waist=12.e-6, tau=18.e-15, z0=-15.e-6,
    )
    add_laser_pulse(sim, laser)

    sim.step(n_steps, show_progress=False)

    fields = {}
    for m in range(Nm):
        fields[m] = {
            'Ez': sim.fld.interp[m].Ez.copy(),
            'Er': sim.fld.interp[m].Er.copy(),
            'Bt': sim.fld.interp[m].Bt.copy(),
            'rho': sim.fld.interp[m].rho.copy(),
        }
    order = np.argsort(elec.tracker.id)
    ptcl = {
        k: getattr(elec, k)[order].copy()
        for k in ('x', 'y', 'z', 'ux', 'uy', 'uz')
    }
    ptcl['id'] = elec.tracker.id[order].copy()
    return fields, ptcl


# ---------------------------------------------------------------------------
# §6.1 CPU vs HIP linear wakefield (linear & cubic, Nm >= 2 + Nm=1 bunch)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    'particle_shape,Nm',
    [('linear', 2), ('linear', 3), ('cubic', 2), ('cubic', 3)],
)
def test_wakefield_cpu_vs_hip(particle_shape, Nm):
    """Compare CPU vs GPU linear-wakefield fields and sorted particles."""
    _require_gpu()

    fld_cpu, ptcl_cpu = _run_linear_wakefield(False, particle_shape, Nm)
    fld_gpu, ptcl_gpu = _run_linear_wakefield(True, particle_shape, Nm)

    max_fld_err = 0.0
    for m in range(Nm):
        for comp in ('Ez', 'Er', 'Bt', 'rho'):
            err = _rel_l2(fld_gpu[m][comp], fld_cpu[m][comp])
            max_fld_err = max(max_fld_err, err)
            assert err <= 1.e-5, (
                f"shape={particle_shape}, Nm={Nm}, mode={m}, field={comp}: "
                f"rel L2={err:.3e} > 1e-5"
            )

    assert np.array_equal(ptcl_gpu['id'], ptcl_cpu['id'])
    max_ptcl_err = 0.0
    for attr in ('x', 'y', 'z', 'ux', 'uy', 'uz'):
        err = _rel_l2(ptcl_gpu[attr], ptcl_cpu[attr])
        max_ptcl_err = max(max_ptcl_err, err)
        assert err <= 1.e-5, (
            f"shape={particle_shape}, Nm={Nm}, ptcl={attr}: "
            f"rel L2={err:.3e} > 1e-5"
        )
    print(
        f"\n[§6.1 {particle_shape}, Nm={Nm}] "
        f"max_fld_rel_L2={max_fld_err:.3e}, max_ptcl_rel_L2={max_ptcl_err:.3e}"
    )


def _run_bunch_wakefield_nm1(use_cuda, particle_shape, seed=1234, n_steps=40):
    set_random_seed(seed)
    Nz = 128
    zmin, zmax = -40.e-6, 40.e-6
    Nr = 48
    rmax = 40.e-6
    dt = (zmax - zmin) / Nz / c

    sim = Simulation(
        Nz, zmax, Nr, rmax, 1, dt,
        zmin=zmin, n_order=-1,
        particle_shape=particle_shape,
        boundaries={'z': 'periodic', 'r': 'reflective'},
        initialize_ions=False,
        use_cuda=use_cuda,
        verbose_level=0,
    )
    elec = sim.add_new_species(
        q=-e, m=m_e, n=1.e24,
        p_zmin=-15.e-6, p_zmax=35.e-6, p_rmax=30.e-6,
        p_nz=2, p_nr=2, p_nt=8,
    )
    elec.track(sim.comm)
    add_elec_bunch(
        sim, gamma0=100.0, n_e=5.e23,
        p_zmin=-15.e-6, p_zmax=-5.e-6, p_rmin=0.0, p_rmax=8.e-6,
        p_nz=2, p_nr=2, p_nt=8,
    )

    sim.step(n_steps, show_progress=False)

    fields = {
        'Ez': sim.fld.interp[0].Ez.copy(),
        'Er': sim.fld.interp[0].Er.copy(),
        'Bt': sim.fld.interp[0].Bt.copy(),
        'rho': sim.fld.interp[0].rho.copy(),
    }
    order = np.argsort(elec.tracker.id)
    ptcl = {
        k: getattr(elec, k)[order].copy()
        for k in ('x', 'y', 'z', 'ux', 'uy', 'uz')
    }
    ptcl['id'] = elec.tracker.id[order].copy()
    return fields, ptcl


@pytest.mark.parametrize('particle_shape', ['linear', 'cubic'])
def test_wakefield_axisymmetric_bunch_nm1_cpu_vs_hip(particle_shape):
    """Exercise ``Nm=1`` (``_one_mode`` kernels) on an axisymmetric bunch."""
    _require_gpu()

    fld_cpu, ptcl_cpu = _run_bunch_wakefield_nm1(False, particle_shape)
    fld_gpu, ptcl_gpu = _run_bunch_wakefield_nm1(True, particle_shape)

    max_fld_err = 0.0
    for comp in ('Ez', 'Er', 'Bt', 'rho'):
        err = _rel_l2(fld_gpu[comp], fld_cpu[comp])
        max_fld_err = max(max_fld_err, err)
        assert err <= 1.e-5, (
            f"bunch Nm=1, shape={particle_shape}, field={comp}: "
            f"rel L2={err:.3e} > 1e-5"
        )

    assert np.array_equal(ptcl_gpu['id'], ptcl_cpu['id'])
    max_ptcl_err = 0.0
    for attr in ('x', 'y', 'z', 'ux', 'uy', 'uz'):
        err = _rel_l2(ptcl_gpu[attr], ptcl_cpu[attr])
        max_ptcl_err = max(max_ptcl_err, err)
        assert err <= 1.e-5, (
            f"bunch Nm=1, shape={particle_shape}, ptcl={attr}: "
            f"rel L2={err:.3e} > 1e-5"
        )
    print(
        f"\n[§6.1 Nm=1 bunch {particle_shape}] "
        f"max_fld_rel_L2={max_fld_err:.3e}, max_ptcl_rel_L2={max_ptcl_err:.3e}"
    )


# ---------------------------------------------------------------------------
# §6.2 HIP repeatability
# ---------------------------------------------------------------------------
def test_hip_repeatability():
    """Verify two fixed-seed GPU runs agree within atomic-order noise."""
    _require_gpu()

    fld_a, ptcl_a = _run_linear_wakefield(True, 'linear', 2, seed=2026)
    fld_b, ptcl_b = _run_linear_wakefield(True, 'linear', 2, seed=2026)

    max_abs_fld = 0.0
    max_rel_fld = 0.0
    for m in range(2):
        for comp in ('Ez', 'Er', 'Bt', 'rho'):
            diff = np.max(np.abs(fld_a[m][comp] - fld_b[m][comp]))
            rel = _rel_l2(fld_a[m][comp], fld_b[m][comp])
            max_abs_fld = max(max_abs_fld, float(diff))
            max_rel_fld = max(max_rel_fld, rel)
            assert rel <= 1.e-12, (
                f"Repeatability drift on m={m} {comp}: {rel:.3e}"
            )

    max_abs_ptcl = 0.0
    max_rel_ptcl = 0.0
    for attr in ('x', 'y', 'z', 'ux', 'uy', 'uz'):
        diff = np.max(np.abs(ptcl_a[attr] - ptcl_b[attr]))
        rel = _rel_l2(ptcl_a[attr], ptcl_b[attr])
        max_abs_ptcl = max(max_abs_ptcl, float(diff))
        max_rel_ptcl = max(max_rel_ptcl, rel)
        assert rel <= 1.e-12, f"Repeatability drift on {attr}: {rel:.3e}"

    print(
        f"\n[§6.2 Repeatability] "
        f"fld(max_abs={max_abs_fld:.3e}, max_rel={max_rel_fld:.3e}), "
        f"ptcl(max_abs={max_abs_ptcl:.3e}, max_rel={max_rel_ptcl:.3e})"
    )


# ---------------------------------------------------------------------------
# §6.3 Vendored xoroshiro128+ vs Numba reference + Statistical MC tests
# ---------------------------------------------------------------------------
def test_xoroshiro128p_matches_numba_reference():
    """Verify vendored xoroshiro128+ states and uniform/normal draws."""
    n = 16
    seed = 42
    subseq = 7
    n_draws = 12

    cpu_states = create_xoroshiro128p_states(
        n, seed, subsequence_start=subseq, on_gpu=False
    )

    @njit
    def _our_cpu_run(states, out_u64, out_u32, out_norm):
        for i in range(states.shape[0]):
            for k in range(out_u64.shape[1]):
                out_u64[i, k] = xoroshiro128p_uniform_float64_cpu(states, i)
                out_u32[i, k] = xoroshiro128p_uniform_float32_cpu(states, i)
                out_norm[i, k] = xoroshiro128p_normal_float64_cpu(states, i)

    our_u64 = np.zeros((n, n_draws), dtype=np.float64)
    our_u32 = np.zeros((n, n_draws), dtype=np.float32)
    our_norm = np.zeros((n, n_draws), dtype=np.float64)
    _our_cpu_run(cpu_states, our_u64, our_u32, our_norm)

    # Verify deterministic repeatability and valid output invariants on CPU
    cpu_states_rep = create_xoroshiro128p_states(
        n, seed, subsequence_start=subseq, on_gpu=False
    )
    rep_u64 = np.zeros((n, n_draws), dtype=np.float64)
    rep_u32 = np.zeros((n, n_draws), dtype=np.float32)
    rep_norm = np.zeros((n, n_draws), dtype=np.float64)
    _our_cpu_run(cpu_states_rep, rep_u64, rep_u32, rep_norm)
    assert np.array_equal(our_u64, rep_u64)
    assert np.array_equal(our_u32, rep_u32)
    assert np.array_equal(our_norm, rep_norm)
    assert np.all((our_u64 >= 0.0) & (our_u64 < 1.0))
    assert np.all((our_u32 >= 0.0) & (our_u32 < 1.0))
    assert np.all(np.isfinite(our_norm))
    assert len(np.unique(our_u64[:, 0])) == n

    if ncr is not None:
        ref_states = np.zeros(n, dtype=ncr.xoroshiro128p_dtype)

        @njit
        def _ref_run(states, s, sub, out_u64, out_u32, out_norm):
            ncr.init_xoroshiro128p_state(states, 0, s)
            for _ in range(sub):
                ncr.xoroshiro128p_jump(states, 0)
            for i in range(1, states.shape[0]):
                states[i] = states[i - 1]
                ncr.xoroshiro128p_jump(states, i)
            for i in range(states.shape[0]):
                for k in range(out_u64.shape[1]):
                    out_u64[i, k] = ncr.xoroshiro128p_uniform_float64(
                        states, i
                    )
                    out_u32[i, k] = ncr.xoroshiro128p_uniform_float32(
                        states, i
                    )
                    out_norm[i, k] = ncr.xoroshiro128p_normal_float64(
                        states, i
                    )

        ref_u64 = np.zeros((n, n_draws), dtype=np.float64)
        ref_u32 = np.zeros((n, n_draws), dtype=np.float32)
        ref_norm = np.zeros((n, n_draws), dtype=np.float64)
        _ref_run(ref_states, seed, subseq, ref_u64, ref_u32, ref_norm)
        assert np.array_equal(our_u64, ref_u64)
        assert np.array_equal(our_u32, ref_u32)
        # Normal uses float64 log(float64(u1)) vs numba.cuda float32 logf(u1)
        assert np.allclose(our_norm, ref_norm, rtol=1.e-6, atol=1.e-6)

    if cuda_installed:
        gpu_states = create_xoroshiro128p_states(
            n, seed, subsequence_start=subseq, on_gpu=True
        )

        @compile_cupy
        def _our_gpu_run(states, out_u64, out_u32, out_norm):
            i = cuda.grid(1)
            if i < states.shape[0]:
                for k in range(out_u64.shape[1]):
                    out_u64[i, k] = xoroshiro128p_uniform_float64(states, i)
                    out_u32[i, k] = xoroshiro128p_uniform_float32(states, i)
                    out_norm[i, k] = xoroshiro128p_normal_float64(states, i)

        g_u64 = cupy.zeros((n, n_draws), dtype=np.float64)
        g_u32 = cupy.zeros((n, n_draws), dtype=np.float32)
        g_norm = cupy.zeros((n, n_draws), dtype=np.float64)
        _our_gpu_run[(1,), (32,)](gpu_states, g_u64, g_u32, g_norm)
        cuda.synchronize()
        assert np.array_equal(g_u64.get(), our_u64)
        assert np.array_equal(g_u32.get(), our_u32)
        # Host libm vs AMDGPU OCML log/cos/sqrt can differ by ~1 ULP (~1e-16)
        assert np.allclose(g_norm.get(), our_norm, rtol=1.e-14, atol=1.e-14)

        @compile_cupy
        def _our_gpu_norm_only(states, out_norm):
            i = cuda.grid(1)
            if i < states.shape[0]:
                for k in range(out_norm.shape[1]):
                    out_norm[i, k] = xoroshiro128p_normal_float64(states, i)

        cpu_states_2 = create_xoroshiro128p_states(
            n, seed, subsequence_start=subseq, on_gpu=False
        )
        gpu_states_2 = create_xoroshiro128p_states(
            n, seed, subsequence_start=subseq, on_gpu=True
        )

        @njit
        def _our_cpu_norm_only(states, out_norm):
            for i in range(states.shape[0]):
                for k in range(out_norm.shape[1]):
                    out_norm[i, k] = xoroshiro128p_normal_float64_cpu(
                        states, i
                    )

        c_norm2 = np.zeros((n, n_draws), dtype=np.float64)
        g_norm2 = cupy.zeros((n, n_draws), dtype=np.float64)
        _our_cpu_norm_only(cpu_states_2, c_norm2)
        _our_gpu_norm_only[(1,), (32,)](gpu_states_2, g_norm2)
        cuda.synchronize()
        assert np.allclose(g_norm2.get(), c_norm2, rtol=1.e-14, atol=1.e-14)


def test_ionization_cpu_vs_hip_statistical():
    """Compare ADK ionization yields and electron energy spectra CPU vs GPU."""
    _require_gpu()

    def _run_ioniz(use_cuda):
        set_random_seed(11)
        Nz, Nr, Nm = 256, 32, 2
        zmin, zmax = -20.e-6, 20.e-6
        rmax = 25.e-6
        dt = (zmax - zmin) / Nz / c
        sim = Simulation(
            Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin, n_order=-1,
            boundaries={'z': 'periodic', 'r': 'reflective'},
            initialize_ions=False, use_cuda=use_cuda, verbose_level=0,
        )
        ions = sim.add_new_species(
            q=0, m=4. * 1836 * m_e, n=5.e23,
            p_zmin=-4.e-6, p_zmax=14.e-6, p_rmax=12.e-6,
            p_nz=2, p_nr=2, p_nt=8,
        )
        elec = sim.add_new_species(q=-e, m=m_e)
        ions.make_ionizable(
            'He', target_species=elec, level_start=0, level_max=2
        )
        add_laser_pulse(
            sim, GaussianLaser(a0=2.5, waist=12.e-6, tau=15.e-15, z0=-6.e-6)
        )
        sim.step(40, show_progress=False)
        ke = np.sqrt(1. + elec.ux ** 2 + elec.uy ** 2 + elec.uz ** 2) - 1.
        return elec.Ntot, elec.w.copy(), ke

    n_cpu, w_cpu, ke_cpu = _run_ioniz(False)
    n_gpu, w_gpu, ke_gpu = _run_ioniz(True)
    assert n_cpu > 500 and n_gpu > 500, (
        f"Too few ionized electrons: cpu={n_cpu}, gpu={n_gpu}"
    )
    sigma = np.sqrt(n_cpu + n_gpu)
    assert abs(n_gpu - n_cpu) <= 3.0 * sigma, (
        f"Ionization count mismatch: cpu={n_cpu}, gpu={n_gpu}, "
        f"3sigma={3 * sigma:.1f}"
    )
    ks_stat, pval = ks_2samp(ke_cpu, ke_gpu)
    assert pval > 0.01, (
        f"Ionization energy KS failed: stat={ks_stat:.4f}, p={pval:.4g}"
    )
    print(f"\n[§6.3 Ionization] N_cpu={n_cpu}, N_gpu={n_gpu}, KS_p={pval:.4f}")


def test_compton_cpu_vs_hip_statistical():
    """Compare Compton photon counts and energy spectra CPU vs GPU."""
    _require_gpu()

    def _run_compton(use_cuda):
        set_random_seed(22)
        Nz, Nr, Nm = 64, 24, 2
        zmin, zmax = -20.e-6, 20.e-6
        rmax = 20.e-6
        dt = (zmax - zmin) / Nz / c
        sim = Simulation(
            Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin, n_order=-1,
            boundaries={'z': 'periodic', 'r': 'reflective'},
            initialize_ions=False, use_cuda=use_cuda, verbose_level=0,
        )
        elec = add_elec_bunch(
            sim, gamma0=80.0, n_e=5.e22,
            p_zmin=-5.e-6, p_zmax=5.e-6, p_rmin=0.0, p_rmax=6.e-6,
            p_nr=4, p_nz=8, p_nt=12,
        )
        photons = sim.add_new_species(q=0.0, m=0.0)
        elec.activate_compton(
            target_species=photons,
            laser_energy=20.0,
            laser_wavelength=0.8e-6,
            laser_waist=10.e-6,
            laser_ctau=10.e-6,
            laser_initial_z0=8.e-6,
            ratio_w_electron_photon=10.0,
        )
        sim.step(25, show_progress=False)
        energy_mc = np.sqrt(
            photons.ux ** 2 + photons.uy ** 2 + photons.uz ** 2
        )
        return photons.Ntot, energy_mc

    n_cpu, e_cpu = _run_compton(False)
    n_gpu, e_gpu = _run_compton(True)
    assert n_cpu > 300 and n_gpu > 300
    sigma = np.sqrt(n_cpu + n_gpu)
    assert abs(n_gpu - n_cpu) <= 3.0 * sigma, (
        f"Compton count mismatch: cpu={n_cpu}, gpu={n_gpu}, "
        f"3sigma={3 * sigma:.1f}"
    )
    ks_stat, pval = ks_2samp(e_cpu, e_gpu)
    assert pval > 0.01, (
        f"Compton spectrum KS failed: stat={ks_stat:.4f}, p={pval:.4g}"
    )
    print(f"\n[§6.3 Compton] N_cpu={n_cpu}, N_gpu={n_gpu}, KS_p={pval:.4f}")


def test_synchrotron_cpu_vs_hip_statistical():
    """Compare synchrotron spectra and recoil gamma distributions."""
    _require_gpu()

    def bfield_y(F, x, y, z, t, amplitude, length_scale):
        return F + amplitude

    def _run_synchrotron(use_cuda):
        set_random_seed(33)
        Nz, Nr, Nm = 64, 24, 2
        zmin, zmax = -20.e-6, 20.e-6
        rmax = 20.e-6
        dt = (zmax - zmin) / Nz / c
        sim = Simulation(
            Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin, n_order=-1,
            boundaries={'z': 'periodic', 'r': 'reflective'},
            initialize_ions=False, use_cuda=use_cuda, verbose_level=0,
        )
        elec = add_elec_bunch(
            sim, gamma0=1500.0, n_e=1.e21,
            p_zmin=-4.e-6, p_zmax=4.e-6, p_rmin=0.0, p_rmax=6.e-6,
            p_nr=3, p_nz=6, p_nt=8,
        )
        sim.external_fields = [
            ExternalField(bfield_y, 'By', 800.0, 1.0, species=elec)
        ]
        elec.activate_synchrotron(
            photon_energy_axis=(1.6e-16, 1.6e-13, 64),
            theta_x_axis=(-0.05, 0.05, 16),
            theta_y_axis=(-0.05, 0.05, 16),
            gamma_cutoff=10.0,
            radiation_reaction=True,
        )
        sim.step(10, show_progress=False)
        elec.synchrotron_radiator.receive_from_gpu()
        rad = elec.synchrotron_radiator.radiation_data.copy()
        omega_ax = elec.synchrotron_radiator.omega_ax.copy()
        gamma_final = 1.0 / elec.inv_gamma
        return rad, omega_ax, gamma_final

    rad_cpu, omega_ax, gam_cpu = _run_synchrotron(False)
    rad_gpu, _, gam_gpu = _run_synchrotron(True)

    tot_cpu = float(np.sum(rad_cpu))
    tot_gpu = float(np.sum(rad_gpu))
    assert tot_cpu > 0.0 and tot_gpu > 0.0
    rel_tot = abs(tot_gpu - tot_cpu) / tot_cpu
    assert rel_tot < 0.05, f"Synchrotron total energy diff {rel_tot:.3e}"

    # Sample photon frequencies from CPU vs HIP angularly integrated spectra
    rng = np.random.default_rng(77)
    pdf_cpu = np.sum(rad_cpu, axis=(0, 1))
    pdf_cpu /= np.sum(pdf_cpu)
    pdf_gpu = np.sum(rad_gpu, axis=(0, 1))
    pdf_gpu /= np.sum(pdf_gpu)
    s_cpu = rng.choice(omega_ax, size=4000, p=pdf_cpu)
    s_gpu = rng.choice(omega_ax, size=4000, p=pdf_gpu)
    ks_spec, pval_spec = ks_2samp(s_cpu, s_gpu)
    ks_gam, pval_gam = ks_2samp(gam_cpu, gam_gpu)
    assert pval_spec > 0.01, (
        f"Synchrotron spectrum KS failed: p={pval_spec:.4g}"
    )
    assert pval_gam > 0.01, (
        f"Synchrotron recoil gamma KS failed: p={pval_gam:.4g}"
    )
    print(
        f"\n[§6.3 Synchrotron] rel_tot={rel_tot:.3e}, "
        f"spec_KS_p={pval_spec:.4f}, recoil_KS_p={pval_gam:.4f}"
    )


# ---------------------------------------------------------------------------
# §6.4 Vendor math unit tests (FFT, DHT, thrust.argsort) vs NumPy
# ---------------------------------------------------------------------------
def test_vendor_math_fft_hankel_sort_vs_numpy():
    """Verify GPU FFT, Hankel transform, and thrust.argsort vs NumPy."""
    _require_gpu()

    rng = np.random.default_rng(123)
    Nz, Nr = 256, 64
    rmax = 30.e-6

    # 1. FFT forward & inverse
    h_in = rng.normal(size=(Nz, Nr)) + 1j * rng.normal(size=(Nz, Nr))
    fft_gpu = FFT(Nr, Nz, use_cuda=True)
    fft_cpu = FFT(Nr, Nz, use_cuda=False)
    h_spec_cpu = np.empty_like(h_in)
    h_back_cpu = np.empty_like(h_in)
    fft_cpu.transform(h_in, h_spec_cpu)
    fft_cpu.inverse_transform(h_spec_cpu, h_back_cpu)

    d_in = cupy.asarray(h_in)
    d_spec = cupy.empty_like(d_in)
    d_back = cupy.empty_like(d_in)
    fft_gpu.transform(d_in, d_spec)
    fft_gpu.inverse_transform(d_spec, d_back)
    cuda.synchronize()
    assert _rel_l2(d_spec.get(), h_spec_cpu) <= 1.e-12
    assert _rel_l2(d_back.get(), h_back_cpu) <= 1.e-12

    # 2. DHT (Hankel transform) forward & inverse
    dht_gpu = DHT(0, 0, Nr, Nz, rmax, use_cuda=True)
    dht_cpu = DHT(0, 0, Nr, Nz, rmax, use_cuda=False)
    h_dht_cpu = np.empty_like(h_in)
    dht_cpu.transform(h_spec_cpu, h_dht_cpu)
    d_dht = cupy.empty_like(d_in)
    dht_gpu.transform(d_spec, d_dht)
    cuda.synchronize()
    assert _rel_l2(d_dht.get(), h_dht_cpu) <= 1.e-12

    # 3. thrust.argsort vs np.argsort + index dtype check
    keys_h = rng.integers(0, 500, size=20000, dtype=np.int32)
    keys_d = cupy.asarray(keys_h.copy())
    idx_d = cupy.empty(keys_h.shape[0], dtype=np.intp)
    thrust.argsort(
        dtype=keys_d.dtype,
        idx_start=idx_d.data.ptr,
        data_start=keys_d.data.ptr,
        keys_start=0,
        shape=keys_d.shape,
    )
    cuda.synchronize()
    idx_h = idx_d.get()
    sorted_keys_d = keys_d.get()
    assert idx_d.dtype == np.dtype(np.intp)
    assert np.array_equal(sorted_keys_d, np.sort(keys_h))
    assert np.array_equal(keys_h[idx_h], sorted_keys_d)


# ---------------------------------------------------------------------------
# §6.5 MPI 2-rank vs 1-rank + shared-GPU oversubscription warning
# ---------------------------------------------------------------------------
def test_mpi_two_ranks_shared_gpu(tmp_path):
    """Compare 2-rank vs 1-rank HIP runs and check oversubscription warning."""
    _require_gpu(hip_only=True)

    driver = tmp_path / 'mpi_wake_driver.py'
    out_1 = tmp_path / 'rank1.npz'
    out_2 = tmp_path / 'rank2.npz'
    out_2_cpu = tmp_path / 'rank2_cpu.npz'
    driver.write_text(
        "import sys\n"
        "import numpy as np\n"
        "from scipy.constants import c, e, m_e\n"
        "from fbpic.main import Simulation\n"
        "from fbpic.lpa_utils.laser import add_laser_pulse, GaussianLaser\n"
        "from fbpic.utils.random_seed import set_random_seed\n"
        "use_cuda = bool(int(sys.argv[2]))\n"
        "set_random_seed(77)\n"
        "Nz, Nr, Nm = 256, 32, 2\n"
        "zmin, zmax = -30.e-6, 30.e-6\n"
        "dt = (zmax - zmin) / Nz / c\n"
        "sim = Simulation(\n"
        "    Nz, zmax, Nr, 25.e-6, Nm, dt, zmin=zmin, n_order=16,\n"
        "    boundaries={'z': 'periodic', 'r': 'reflective'},\n"
        "    initialize_ions=False, use_cuda=use_cuda, verbose_level=0,\n"
        ")\n"
        "sim.add_new_species(\n"
        "    q=-e, m=m_e, n=1.e24, p_zmin=-10.e-6, p_zmax=25.e-6,\n"
        "    p_rmax=20.e-6, p_nz=2, p_nr=2, p_nt=8,\n"
        ")\n"
        "add_laser_pulse(\n"
        "    sim,\n"
        "    GaussianLaser(a0=0.3, waist=10.e-6, tau=15.e-15, z0=-10.e-6),\n"
        ")\n"
        "sim.step(20, show_progress=False)\n"
        "ez0 = sim.comm.gather_grid_array(sim.fld.interp[0].Ez, root=0)\n"
        "er0 = sim.comm.gather_grid_array(sim.fld.interp[0].Er, root=0)\n"
        "bt1 = sim.comm.gather_grid_array(sim.fld.interp[1].Bt, root=0)\n"
        "if sim.comm.rank == 0:\n"
        "    np.savez(sys.argv[1], ez0=ez0, er0=er0, bt1=bt1)\n"
    )

    env = os.environ.copy()
    env['FBPIC_GPU_BACKEND'] = 'hip'
    env.pop('HIP_VISIBLE_DEVICES', None)
    env.setdefault('ROCR_VISIBLE_DEVICES', '1')
    env['OMP_WAIT_POLICY'] = 'PASSIVE'
    env['OMP_NUM_THREADS'] = '4'
    env['NUMBA_NUM_THREADS'] = '4'
    env['OPENBLAS_NUM_THREADS'] = '1'

    res1 = subprocess.run(
        ['mpiexec', '-n', '1', sys.executable, str(driver), str(out_1), '1'],
        env=env, capture_output=True, text=True, check=False,
    )
    assert res1.returncode == 0, (
        f"1-rank MPI failed:\n{res1.stderr}\n{res1.stdout}"
    )
    res2 = subprocess.run(
        ['mpiexec', '-n', '2', sys.executable, str(driver), str(out_2), '1'],
        env=env, capture_output=True, text=True, check=False,
    )
    assert res2.returncode == 0, (
        f"2-rank MPI failed:\n{res2.stderr}\n{res2.stdout}"
    )
    assert "GPUs have been oversubscribed by MPI ranks." in res2.stderr, (
        f"Expected shared-GPU oversubscription warning, got:\n{res2.stderr}\n"
        f"(1-rank stderr:\n{res1.stderr})"
    )

    env_cpu = env.copy()
    env_cpu['FBPIC_GPU_BACKEND'] = 'cpu'
    env_cpu['ROCR_VISIBLE_DEVICES'] = ''
    res2_cpu = subprocess.run(
        [
            'mpiexec', '-n', '2', sys.executable,
            str(driver), str(out_2_cpu), '0',
        ],
        env=env_cpu, capture_output=True, text=True, check=False,
    )
    assert res2_cpu.returncode == 0, (
        f"2-rank CPU MPI failed:\n{res2_cpu.stderr}\n{res2_cpu.stdout}"
    )

    d1 = np.load(out_1)
    d2 = np.load(out_2)
    d2_cpu = np.load(out_2_cpu)

    # 2-rank HIP vs 2-rank CPU must match to machine precision
    cpu_hip_ez = _rel_l2(d2['ez0'], d2_cpu['ez0'])
    cpu_hip_er = _rel_l2(d2['er0'], d2_cpu['er0'])
    cpu_hip_bt = _rel_l2(d2['bt1'], d2_cpu['bt1'])
    assert cpu_hip_ez <= 1.e-10, (
        f"2-rank HIP vs CPU Ez rel_L2={cpu_hip_ez:.3e}"
    )
    assert cpu_hip_er <= 1.e-10, (
        f"2-rank HIP vs CPU Er rel_L2={cpu_hip_er:.3e}"
    )
    assert cpu_hip_bt <= 1.e-10, (
        f"2-rank HIP vs CPU Bt1 rel_L2={cpu_hip_bt:.3e}"
    )

    # n_order=16 stencil agrees across ranks within guard-cell truncation
    rel_ez = _rel_l2(d2['ez0'], d1['ez0'])
    rel_er = _rel_l2(d2['er0'], d1['er0'])
    rel_bt = _rel_l2(d2['bt1'], d1['bt1'])
    assert rel_ez <= 2.5e-3, f"2-rank vs 1-rank Ez rel_L2={rel_ez:.3e}"
    assert rel_er <= 2.5e-3, f"2-rank vs 1-rank Er rel_L2={rel_er:.3e}"
    assert rel_bt <= 2.5e-3, f"2-rank vs 1-rank Bt1 rel_L2={rel_bt:.3e}"
    print(
        f"\n[§6.5 MPI] 2rank_vs_1rank="
        f"(Ez={rel_ez:.3e}, Er={rel_er:.3e}, Bt1={rel_bt:.3e}), "
        f"2rank_HIP_vs_CPU="
        f"(Ez={cpu_hip_ez:.3e}, Er={cpu_hip_er:.3e}, Bt1={cpu_hip_bt:.3e})"
    )


# ---------------------------------------------------------------------------
# Phase 6 helper unit test: RDNA wave32 & CDNA wave64 launch-config selection
# ---------------------------------------------------------------------------
def test_wavefront_alignment_helpers():
    """Verify HIP block-size helpers align cleanly to wave32 and wave64."""
    rdna_mock = {
        'vendor': 'amd', 'arch': 'gfx1201', 'warp_size': 32, 'family': 'rdna',
    }
    cdna_mock = {
        'vendor': 'amd', 'arch': 'gfx942', 'warp_size': 64, 'family': 'cdna',
    }

    with patch('fbpic.utils.gpu_backend.gpu_backend', 'hip'):
        for info in (rdna_mock, cdna_mock):
            w = info['warp_size']
            with patch(
                'fbpic.utils.gpu_backend.get_device_arch_info',
                return_value=info,
            ):
                bpg1, tpb1 = cuda_tpb_bpg_1d(1000, TPB=8)
                assert tpb1 % w == 0 and bpg1 >= 1
                _, (tx, ty) = cuda_tpb_bpg_2d(100, 64, TPBx=1, TPBy=16)
                assert (tx * ty) % w == 0
                assert get_deposit_tpb('linear', arch_info=info) % w == 0
                assert get_deposit_tpb('cubic', arch_info=info) % w == 0
                assert get_gather_tpb('linear', arch_info=info) % w == 0
                assert get_gather_tpb('cubic', arch_info=info) % w == 0
                c_tpb = get_copy_tpb_2d(arch_info=info)
                s_tpb = get_spectral_tpb_2d(arch_info=info)
                assert (c_tpb[0] * c_tpb[1]) % w == 0
                assert (s_tpb[0] * s_tpb[1]) % w == 0
