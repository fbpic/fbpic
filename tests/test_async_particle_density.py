import os
import sys
from types import MethodType, SimpleNamespace

# The default Numba backend aborts while launching parallel workers on macOS.
if sys.platform == "darwin":
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import h5py
import numpy as np
import pytest
from scipy.constants import c, e, m_e

from fbpic.main import Simulation
from fbpic.openpmd_diag import (
    ParticleChargeDensityDiagnostic,
    asynchronous_diagnostics as make_async_diagnostics,
)
from fbpic.openpmd_diag.async_diag import (
    AsyncDiagnosticGroup, SnapshotPool,
    _AsyncParticleChargeDensityDiagnostic)


def test_async_group_accepts_exact_particle_density_diagnostic():
    diagnostic = object.__new__(ParticleChargeDensityDiagnostic)
    diagnostic.comm = None
    group = AsyncDiagnosticGroup([diagnostic])

    assert type(group.diagnostics[0]).__name__ == (
        "_AsyncParticleChargeDensityDiagnostic")
    group.close()


def test_duplicate_density_diagnostics_use_independent_pools():
    first = object.__new__(ParticleChargeDensityDiagnostic)
    first.comm = None
    second = object.__new__(ParticleChargeDensityDiagnostic)
    second.comm = None
    group = AsyncDiagnosticGroup([first, second])

    assert (group.diagnostics[0].snapshot_pool
            is not group.diagnostics[1].snapshot_pool)
    group.close()


def test_async_group_rejects_particle_density_subclass():
    class DerivedDensityDiagnostic(ParticleChargeDensityDiagnostic):
        pass

    diagnostic = object.__new__(DerivedDensityDiagnostic)
    diagnostic.comm = None
    with pytest.raises(TypeError, match="ParticleChargeDensityDiagnostic"):
        AsyncDiagnosticGroup([diagnostic])


def test_synchronous_density_preserves_deposit_transform_write_order():
    events = []
    first, second = object(), object()
    diagnostic = object.__new__(ParticleChargeDensityDiagnostic)
    diagnostic.species = {"first": first, "second": second}
    diagnostic.comm = SimpleNamespace(
        get_zmin_zmax=lambda **kwargs: (0.0, 1.0),
        get_Nz_and_iz=lambda **kwargs: (2, 0),
        get_Nr=lambda **kwargs: 3)
    diagnostic.write_dir = "."
    diagnostic.fld = SimpleNamespace(
        dt=1.0, interp=[SimpleNamespace(dz=1.0)], use_cuda=False,
        spect2interp=lambda quantity: events.append(("transform", quantity)))
    diagnostic.sim = SimpleNamespace(
        comm=None,
        deposit=lambda fieldtype, species_list, update_spectral, exchange:
            events.append(("deposit", fieldtype, species_list[0])))
    diagnostic.create_file_empty_meshes = MethodType(
        lambda self, *args: events.append(("legacy_create", 7)), diagnostic)
    diagnostic.open_file = MethodType(lambda self, path: None, diagnostic)
    diagnostic.write_dataset = MethodType(
        lambda self, group, name, quantity:
            events.append(("legacy_write", name)), diagnostic)
    diagnostic._create_density_file = MethodType(
        lambda self, iteration:
            events.append(("create", iteration)) or "file.h5", diagnostic)
    diagnostic._write_density_dataset = MethodType(
        lambda self, path, iteration, name:
            events.append(("write", path, iteration, name)), diagnostic)

    diagnostic.write_hdf5(7)

    assert events == [
        ("create", 7),
        ("deposit", "rho_next", first),
        ("transform", "rho_next"),
        ("write", "file.h5", 7, "first"),
        ("deposit", "rho_next", second),
        ("transform", "rho_next"),
        ("write", "file.h5", 7, "second"),
    ]


class RecordingWriter:
    def __init__(self):
        self.begun = []
        self.submitted = []

    def begin_iteration(self, iteration):
        self.begun.append(iteration)

    def submit(self, diagnostic, iteration):
        self.submitted.append((diagnostic, iteration))


def test_density_snapshot_is_detached_from_live_rho():
    events = []
    grids = [
        SimpleNamespace(rho=np.full((2, 3), 1.0), use_cuda=False),
        SimpleNamespace(rho=np.full((2, 3), 2.0), use_cuda=False),
    ]
    field = SimpleNamespace(
        interp=grids, use_cuda=False, Nm=2, dt=1.0,
        spect2interp=lambda quantity: None)

    class FakeDensityDiagnostic:
        period = 5
        iteration_min = 0
        iteration_max = 100

        def __init__(self):
            self.species = {"electrons": object()}
            self.fld = field
            self.sim = SimpleNamespace(
                deposit=lambda *args, **kwargs: events.append("deposit"))

        def _create_density_file(self, iteration):
            return "density.h5"

        def _write_density_dataset(self, path, iteration, name):
            events.append([grid.rho.copy() for grid in self.fld.interp])

    diagnostic = FakeDensityDiagnostic()
    writer = RecordingWriter()
    wrapper = _AsyncParticleChargeDensityDiagnostic(
        diagnostic, writer, SnapshotPool())

    wrapper.write(10)
    grids[0].rho[:] = 9.0
    grids[1].rho[:] = 9.0
    snapshot, iteration = writer.submitted[0]
    snapshot.write_hdf5(iteration)

    assert writer.begun == [10]
    assert events[0] == "deposit"
    np.testing.assert_array_equal(events[1][0], np.full((2, 3), 1.0))
    np.testing.assert_array_equal(events[1][1], np.full((2, 3), 2.0))
    assert snapshot.diagnostic.sim is None


def test_density_snapshot_freezes_communicator_geometry_and_guard_cropping():
    class MovingCommunicator:
        rank = 0
        size = 1

        def __init__(self):
            self.zmin = 4.0

        def get_zmin_zmax(self, local, with_damp, with_guard, rank=None):
            assert (local, with_damp, with_guard, rank) == (
                False, False, False, None)
            return self.zmin, self.zmin + 2.0

        def get_Nz_and_iz(self, local, with_damp, with_guard, rank=None):
            if (local, with_damp, with_guard, rank) == (
                    False, False, False, None):
                return 2, 0
            if (local, with_damp, with_guard, rank) == (
                    True, False, False, 0):
                return 2, 0
            if (local, with_damp, with_guard, rank) == (
                    True, True, True, 0):
                return 4, -1
            raise AssertionError("unexpected geometry request")

        def get_Nr(self, with_damp):
            assert with_damp is False
            return 3

        def gather_grid_array(self, array):
            return array[1:3, :3]

    events = []
    communicator = MovingCommunicator()
    diagnostic = object.__new__(ParticleChargeDensityDiagnostic)
    diagnostic.period = 1
    diagnostic.iteration_min = 0
    diagnostic.iteration_max = 10
    diagnostic.species = {"electrons": object()}
    diagnostic.comm = communicator
    diagnostic.write_dir = "."
    diagnostic.fld = SimpleNamespace(
        dt=0.25, use_cuda=False, Nm=1,
        interp=[SimpleNamespace(
            rho=np.array([[99., 99., 99.], [1., 1., 1.],
                          [2., 2., 2.], [99., 99., 99.]]),
            use_cuda=False, dz=0.5)],
        spect2interp=lambda quantity: None)
    diagnostic.sim = SimpleNamespace(
        deposit=lambda *args, **kwargs: None)
    diagnostic.create_file_empty_meshes = MethodType(
        lambda self, path, iteration, time, Nr, Nz, zmin, dz, dt:
            events.append(("geometry", Nr, Nz, zmin, dz, dt)), diagnostic)
    diagnostic.open_file = MethodType(lambda self, path: None, diagnostic)
    diagnostic.write_dataset = MethodType(
        lambda self, group, name, quantity:
            events.append(("density", name, self.get_dataset(quantity, 0))),
        diagnostic)
    writer = RecordingWriter()
    wrapper = _AsyncParticleChargeDensityDiagnostic(
        diagnostic, writer, SnapshotPool())

    wrapper.write(4)
    communicator.zmin = 104.0
    snapshot, iteration = writer.submitted[0]
    snapshot.write_hdf5(iteration)

    assert events[0] == ("geometry", 3, 2, 4.0, 0.5, 0.25)
    np.testing.assert_array_equal(
        events[1][2], np.array([[1., 1., 1.], [2., 2., 2.]]))


@pytest.mark.parametrize("iteration", [0, 5, 11, 20])
def test_density_snapshot_respects_iteration_bounds(iteration):
    diagnostic = SimpleNamespace(
        period=5, iteration_min=10, iteration_max=20,
        species={"electrons": object()},
        fld=SimpleNamespace(interp=[], spect2interp=lambda quantity: None),
        sim=SimpleNamespace(
            deposit=lambda *args, **kwargs:
                pytest.fail("deposit called outside output bounds")))
    writer = RecordingWriter()
    wrapper = _AsyncParticleChargeDensityDiagnostic(
        diagnostic, writer, SnapshotPool())

    wrapper.write(iteration)

    assert writer.begun == []
    assert writer.submitted == []


def test_density_snapshot_copy_failure_stays_on_caller_thread():
    class FailingPool:
        buffers = {}

        def copy(self, array, key):
            raise RuntimeError("density copy failed")

    diagnostic = SimpleNamespace(
        period=1, iteration_min=0, iteration_max=10,
        species={"electrons": object()},
        fld=SimpleNamespace(
            interp=[SimpleNamespace(rho=np.ones((2, 3)))],
            spect2interp=lambda quantity: None),
        sim=SimpleNamespace(deposit=lambda *args, **kwargs: None))
    writer = RecordingWriter()
    wrapper = _AsyncParticleChargeDensityDiagnostic(
        diagnostic, writer, FailingPool())

    with pytest.raises(RuntimeError, match="density copy failed"):
        wrapper.write(0)

    assert writer.submitted == []


def test_density_species_use_distinct_snapshot_keys():
    class RecordingPool:
        buffers = {}

        def __init__(self):
            self.keys = []

        def copy(self, array, key):
            self.keys.append(key)
            return np.array(array, copy=True)

    pool = RecordingPool()
    diagnostic = SimpleNamespace(
        period=1, iteration_min=0, iteration_max=10,
        species={"first": object(), "second": object()},
        fld=SimpleNamespace(
            interp=[SimpleNamespace(rho=np.ones((2, 3)))],
            spect2interp=lambda quantity: None, use_cuda=False),
        sim=SimpleNamespace(deposit=lambda *args, **kwargs: None))
    writer = RecordingWriter()
    wrapper = _AsyncParticleChargeDensityDiagnostic(
        diagnostic, writer, pool)

    wrapper.write(0)

    assert pool.keys == [
        ("particle_density", "first", 0),
        ("particle_density", "second", 0),
    ]


def collect_hdf5(path):
    result = {"datasets": {}, "attributes": {}}
    with h5py.File(path, "r") as handle:
        def collect(name, obj):
            result["attributes"][name] = dict(obj.attrs)
            if isinstance(obj, h5py.Dataset):
                result["datasets"][name] = obj[...]
        handle.visititems(collect)
        result["attributes"][""] = dict(handle.attrs)
    return result


def run_density_case(output_dir, asynchronous):
    output_dir.mkdir()
    np.random.seed(0)
    zmax = 2.0e-6
    nz = 16
    sim = Simulation(
        nz, zmax, 8, 2.0e-6, 2, zmax / nz / c,
        zmin=0.0, n_order=-1, use_cuda=False,
        boundaries={"z": "periodic", "r": "reflective"},
        verbose_level=0)
    species = sim.add_new_species(
        q=-e, m=m_e, n=1.0e20,
        p_nz=1, p_nr=1, p_nt=4,
        p_zmin=0.0, p_zmax=zmax, p_rmax=2.0e-6)
    diagnostic = ParticleChargeDensityDiagnostic(
        1, sim, {"electrons": species}, write_dir=str(output_dir))
    if asynchronous:
        sim.diags = make_async_diagnostics(
            [diagnostic], use_pinned_memory=False)
    else:
        sim.diags = [diagnostic]
    sim.step(1, show_progress=False)
    if asynchronous:
        sim.diags[0].close()
    assert all(np.all(np.isfinite(grid.rho)) for grid in sim.fld.interp)
    return collect_hdf5(output_dir / "hdf5/data00000000.h5")


def assert_hdf5_attributes_equal(left, right):
    assert left.keys() == right.keys()
    for path in left:
        assert left[path].keys() == right[path].keys()
        for name in left[path]:
            if path == "" and name == "date":
                assert left[path][name]
                assert right[path][name]
            else:
                np.testing.assert_equal(left[path][name], right[path][name])


def assert_hdf5_datasets_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert left[name].shape == right[name].shape
        assert left[name].dtype == right[name].dtype
        np.testing.assert_array_equal(left[name], right[name])


def test_particle_density_openpmd_matches_synchronous_output(tmp_path):
    synchronous = run_density_case(tmp_path / "sync", asynchronous=False)
    asynchronous = run_density_case(tmp_path / "async", asynchronous=True)

    assert_hdf5_datasets_equal(
        synchronous["datasets"], asynchronous["datasets"])
    assert_hdf5_attributes_equal(
        synchronous["attributes"], asynchronous["attributes"])


@pytest.mark.parametrize("left, right", [
    (np.ones((1, 2), dtype=np.float64), np.ones(2, dtype=np.float64)),
    (np.ones(2, dtype=np.float64), np.ones(2, dtype=np.float32)),
])
def test_hdf5_dataset_comparison_rejects_shape_and_dtype_mismatches(
        left, right):
    with pytest.raises(AssertionError):
        assert_hdf5_datasets_equal({"rho": left}, {"rho": right})


def test_density_write_failure_propagates_and_close_releases_buffers():
    def fail_write(path, iteration, species_name):
        raise RuntimeError("density write failed")

    diagnostic = object.__new__(ParticleChargeDensityDiagnostic)
    diagnostic.comm = None
    diagnostic.period = 1
    diagnostic.iteration_min = 0
    diagnostic.iteration_max = 10
    diagnostic.species = {"electrons": object()}
    diagnostic.fld = SimpleNamespace(
        interp=[SimpleNamespace(rho=np.ones((2, 3)), use_cuda=False)],
        use_cuda=False, Nm=1, spect2interp=lambda quantity: None)
    diagnostic.sim = SimpleNamespace(deposit=lambda *args, **kwargs: None)
    diagnostic._create_density_file = lambda iteration: "density.h5"
    diagnostic._write_density_dataset = fail_write
    group = AsyncDiagnosticGroup([diagnostic], use_pinned_memory=False)

    group.write(0)
    with pytest.raises(RuntimeError, match="density write failed"):
        group.flush()
    group.diagnostics[0].snapshot_pool.buffers["sentinel"] = np.ones(1)
    group.close()

    assert group.diagnostics[0].snapshot_pool.buffers == {}
