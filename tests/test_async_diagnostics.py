import numpy as np
import pytest

from fbpic.openpmd_diag.async_diag import (
    AsyncDiagnosticGroup,
    AsyncDiagnosticWriteError,
    SnapshotPool,
    _SerialDiagnosticWriter,
    asynchronous_diagnostics,
)


class FakeGpuArray:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.dtype = self.values.dtype
        self.ndim = self.values.ndim
        self.shape = self.values.shape
        self.size = self.values.size

    def get(self, out=None):
        if out is None:
            return self.values.copy()
        out[...] = self.values
        return out


def test_snapshot_pool_reuses_pinned_capacity_for_particle_arrays():
    allocations = []

    def allocate(shape, dtype):
        allocations.append((shape, dtype))
        return np.empty(shape, dtype=dtype)

    pool = SnapshotPool(pinned=True, pinned_allocator=allocate)
    first = pool.copy(FakeGpuArray(np.arange(5.0)), ("particle", "x"))
    second = pool.copy(FakeGpuArray(np.arange(3.0)), ("particle", "x"))

    assert allocations == [(8, np.dtype("float64"))]
    np.testing.assert_array_equal(first, np.arange(5.0))
    np.testing.assert_array_equal(second, np.arange(3.0))
    assert np.shares_memory(first, second)


def test_snapshot_pool_returns_owning_copy_for_cpu_arrays():
    source = np.arange(4.0)
    snapshot = SnapshotPool().copy(source, "field")
    source[:] = -1
    np.testing.assert_array_equal(snapshot, np.arange(4.0))


def test_serial_writer_preserves_submission_order():
    events = []

    class Diagnostic:
        def __init__(self, name):
            self.name = name

        def write_hdf5(self, iteration):
            events.append((self.name, iteration))

    writer = _SerialDiagnosticWriter()
    writer.submit(Diagnostic("field"), 10)
    writer.submit(Diagnostic("particle"), 10)
    writer.submit(Diagnostic("density"), 10)
    writer.flush()
    writer.close()

    assert events == [
        ("field", 10), ("particle", 10), ("density", 10)]


def test_serial_writer_propagates_each_pending_failure():
    class FailingDiagnostic:
        def write_hdf5(self, iteration):
            raise RuntimeError("field write failed")

    class SuccessfulDiagnostic:
        def write_hdf5(self, iteration):
            pass

    writer = _SerialDiagnosticWriter()
    writer.submit(FailingDiagnostic(), 10)
    writer.submit(SuccessfulDiagnostic(), 10)

    with pytest.raises(RuntimeError, match="field write failed"):
        writer.flush()
    writer.close()


def test_serial_writer_aggregates_multiple_pending_failures():
    class FailingDiagnostic:
        def __init__(self, message):
            self.message = message

        def write_hdf5(self, iteration):
            raise RuntimeError(self.message)

    writer = _SerialDiagnosticWriter()
    writer.submit(FailingDiagnostic("field failed"), 10)
    writer.submit(FailingDiagnostic("particle failed"), 10)

    with pytest.raises(AsyncDiagnosticWriteError) as exc_info:
        writer.close()
    assert [str(error) for error in exc_info.value.errors] == [
        "field failed", "particle failed"]
    with pytest.raises(RuntimeError, match="cannot schedule new futures"):
        writer.submit(FailingDiagnostic("closed"), 20)


def test_async_diagnostics_requires_supported_diagnostics():
    with pytest.raises(ValueError, match="at least one"):
        asynchronous_diagnostics([])
    with pytest.raises(
            TypeError,
            match="FieldDiagnostic, ParticleDiagnostic, or "
                  "ParticleChargeDensityDiagnostic"):
        asynchronous_diagnostics([object()])


def test_async_group_rejects_multi_rank_output():
    from fbpic.openpmd_diag import FieldDiagnostic

    diagnostic = object.__new__(FieldDiagnostic)
    diagnostic.comm = type("Comm", (), {"size": 2})()
    with pytest.raises(NotImplementedError, match="single-rank"):
        AsyncDiagnosticGroup([diagnostic])


def test_async_group_rejects_world_mpi_with_comm_none(monkeypatch):
    import fbpic.openpmd_diag.async_diag as async_diag
    from fbpic.openpmd_diag import FieldDiagnostic

    diagnostic = object.__new__(FieldDiagnostic)
    diagnostic.comm = None
    monkeypatch.setattr(
        async_diag, "world_comm", type("Comm", (), {"size": 2})())

    with pytest.raises(NotImplementedError, match="single-rank"):
        AsyncDiagnosticGroup([diagnostic])


def test_async_group_rejects_particle_subsampling():
    from fbpic.openpmd_diag import ParticleDiagnostic

    diagnostic = object.__new__(ParticleDiagnostic)
    diagnostic.comm = None
    diagnostic.subsampling_fraction = 0.5
    with pytest.raises(NotImplementedError, match="subsampling"):
        AsyncDiagnosticGroup([diagnostic])


def test_duplicate_diagnostics_use_independent_snapshot_pools():
    from fbpic.openpmd_diag import FieldDiagnostic

    first = object.__new__(FieldDiagnostic)
    first.comm = None
    second = object.__new__(FieldDiagnostic)
    second.comm = None
    group = AsyncDiagnosticGroup([first, second])

    assert (group.diagnostics[0].snapshot_pool
            is not group.diagnostics[1].snapshot_pool)
    group.close()
