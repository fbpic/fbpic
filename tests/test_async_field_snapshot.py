"""Field values and geometry must describe the same output iteration."""

import threading

import h5py
import numpy as np
import pytest
from scipy.constants import c

from fbpic.main import Simulation
from fbpic.openpmd_diag import FieldDiagnostic
from fbpic.openpmd_diag.async_diag import AsyncDiagnosticGroup


@pytest.mark.parametrize("with_comm", [True, False])
@pytest.mark.parametrize("boundary", ["periodic", "open"])
def test_delayed_field_write_preserves_values_and_geometry(
        tmp_path, with_comm, boundary):
    sim = Simulation(
        16, 2.e-6, 8, 2.e-6, 2, 2.e-6 / 16 / c,
        zmin=0., use_cuda=False, n_order=8, n_guard=16,
        boundaries={"z": boundary, "r": "reflective"}, verbose_level=0)
    for mode, grid in enumerate(sim.fld.interp):
        for component in ("Er", "Et", "Ez", "Br", "Bt", "Bz"):
            values = np.arange(grid.Er.size).reshape(grid.Er.shape) + 1.
            getattr(grid, component)[:] = values * (1. + 0.5j * mode)

    diagnostics = [FieldDiagnostic(
        1, sim.fld, comm=sim.comm if with_comm else None,
        write_dir=str(tmp_path / name)) for name in ("sync", "async")]
    diagnostics[0].write(0)
    group = AsyncDiagnosticGroup([diagnostics[1]], use_pinned_memory=False)
    started, release = threading.Event(), threading.Event()

    def block_worker():
        started.set()
        if not release.wait(10):
            raise RuntimeError("diagnostic test worker was not released")

    blocker = group.writer.executor.submit(block_worker)
    try:
        assert started.wait(5)
        group.write(0)
        shift = 5 * sim.fld.interp[0].dz
        sim.comm.shift_global_domain_positions(shift)
        for grid in sim.fld.interp:
            grid.zmin += shift
            grid.zmax += shift
            grid.Er[:] = -999.
    finally:
        release.set()
        blocker.result(timeout=5)
        group.close()

    with h5py.File(tmp_path / "sync/hdf5/data00000000.h5") as reference:
        with h5py.File(tmp_path / "async/hdf5/data00000000.h5") as candidate:
            def compare(name, obj):
                other = candidate[name]
                assert set(obj.attrs) == set(other.attrs)
                for key in obj.attrs:
                    np.testing.assert_array_equal(
                        obj.attrs[key], other.attrs[key])
                if isinstance(obj, h5py.Dataset):
                    np.testing.assert_array_equal(obj[...], other[...])
            reference.visititems(compare)
