# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Asynchronous, snapshot-based diagnostics for single-GPU simulations."""

import copy
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np

from fbpic.utils.mpi import comm as world_comm
from .field_diag import FieldDiagnostic
from .particle_diag import ParticleDiagnostic
from .particle_density_diag import ParticleChargeDensityDiagnostic


class AsyncDiagnosticWriteError(RuntimeError):
    """Report multiple failures from one serialized diagnostic batch."""

    def __init__(self, errors):
        self.errors = tuple(errors)
        messages = "; ".join(str(error) for error in self.errors)
        super().__init__(
            "multiple asynchronous diagnostic writes failed: " + messages)


class SnapshotPool:
    """Reuse optional pinned host buffers between diagnostic outputs."""

    def __init__(self, pinned=False, pinned_allocator=None):
        self.pinned = pinned
        self.pinned_allocator = pinned_allocator
        self.buffers = {}

    def _allocate(self, shape, dtype):
        if self.pinned_allocator is not None:
            return self.pinned_allocator(shape, dtype)
        from cupyx import empty_pinned
        return empty_pinned(shape, dtype=dtype)

    def copy(self, array, key):
        """Return a host snapshot whose storage remains valid until reuse."""
        if not hasattr(array, "get"):
            return np.array(array, copy=True)
        if not self.pinned:
            return array.get()
        if array.size == 0:
            return np.empty(array.shape, dtype=array.dtype)

        buffer = self.buffers.get(key)
        if array.ndim == 1:
            required = array.size
            if (buffer is None or buffer.dtype != array.dtype
                    or buffer.size < required):
                capacity = 1 << (required - 1).bit_length()
                buffer = self._allocate(capacity, array.dtype)
                self.buffers[key] = buffer
            output = buffer[:required]
        else:
            if (buffer is None or buffer.dtype != array.dtype
                    or buffer.shape != array.shape):
                buffer = self._allocate(array.shape, array.dtype)
                self.buffers[key] = buffer
            output = buffer
        array.get(out=output)
        return output


class _SerialDiagnosticWriter:
    """Serialize HDF5 writes while the main thread advances the GPU."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending_futures = []
        self.capture_iteration = None
        self.worker_elapsed_s = 0.0

    def begin_iteration(self, iteration):
        """Wait before buffers from an earlier output iteration are reused."""
        if iteration != self.capture_iteration:
            self.flush()
            self.capture_iteration = iteration

    def submit(self, diagnostic, iteration):
        def write_snapshot():
            start = time.perf_counter()
            diagnostic.write_hdf5(iteration)
            self.worker_elapsed_s += time.perf_counter() - start

        self.pending_futures.append(
            self.executor.submit(write_snapshot))

    def flush(self):
        pending_futures = self.pending_futures
        self.pending_futures = []
        errors = []
        for future in pending_futures:
            try:
                future.result()
            except Exception as error:
                errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise AsyncDiagnosticWriteError(errors)

    def close(self):
        error = None
        try:
            self.flush()
        except Exception as caught_error:
            error = caught_error
        finally:
            self.executor.shutdown()
        if error is not None:
            raise error


def _is_output_iteration(diagnostic, iteration):
    return (iteration % diagnostic.period == 0
            and iteration >= diagnostic.iteration_min
            and iteration < diagnostic.iteration_max)


class _AsyncFieldDiagnostic:
    """Capture requested interpolation-grid fields into host snapshots."""

    def __init__(self, diagnostic, writer, snapshot_pool):
        self.diagnostic = diagnostic
        self.writer = writer
        self.snapshot_pool = snapshot_pool

    def write(self, iteration):
        if not _is_output_iteration(self.diagnostic, iteration):
            return
        self.writer.begin_iteration(iteration)
        fld = self.diagnostic.fld

        if "rho" in self.diagnostic.fieldtypes:
            fld.spect2interp("rho_prev")
            if (self.diagnostic.comm is not None
                    and self.diagnostic.comm.size > 1
                    and not fld.exchanged_source["rho_prev"]):
                self.diagnostic.comm.exchange_fields(
                    fld.interp, "rho", "add")
        if "J" in self.diagnostic.fieldtypes:
            fld.spect2interp("J")
            if (self.diagnostic.comm is not None
                    and self.diagnostic.comm.size > 1
                    and not fld.exchanged_source["J"]):
                self.diagnostic.comm.exchange_fields(
                    fld.interp, "J", "add")

        quantities = []
        for fieldtype in self.diagnostic.fieldtypes:
            if fieldtype == "rho" or fieldtype.endswith("_pml"):
                quantities.append(fieldtype)
            elif fieldtype in ("E", "B", "J"):
                quantities.extend(
                    fieldtype + coord for coord in self.diagnostic.coords)

        snapshot_fld = copy.copy(fld)
        snapshot_fld.use_cuda = False
        snapshot_fld.spect2interp = lambda _quantity: None
        snapshot_fld.interp = []
        for grid_index, grid in enumerate(fld.interp):
            snapshot_grid = copy.copy(grid)
            snapshot_grid.use_cuda = False
            for quantity in quantities:
                setattr(snapshot_grid, quantity, self.snapshot_pool.copy(
                    getattr(grid, quantity),
                    ("field", grid_index, quantity)))
            snapshot_fld.interp.append(snapshot_grid)

        snapshot = copy.copy(self.diagnostic)
        snapshot.fld = snapshot_fld
        if self.diagnostic.comm is not None:
            # The moving window can advance before the worker writes metadata.
            # Freeze geometry and guard cropping with the field values.
            snapshot.comm = _SingleRankCommunicatorSnapshot(
                self.diagnostic.comm)
        self.writer.submit(snapshot, iteration)


class _AsyncParticleDiagnostic:
    """Capture requested particle records without moving live GPU state."""

    def __init__(self, diagnostic, writer, snapshot_pool):
        self.diagnostic = diagnostic
        self.writer = writer
        self.snapshot_pool = snapshot_pool

    def write(self, iteration):
        if not _is_output_iteration(self.diagnostic, iteration):
            return
        self.writer.begin_iteration(iteration)
        snapshot = copy.copy(self.diagnostic)
        snapshot.species_dict = {}

        for species_name in self.diagnostic.species_names_list:
            species = self.diagnostic.species_dict[species_name]
            snapshot_species = copy.copy(species)
            snapshot_species.use_cuda = False
            quantities = set(
                self.diagnostic.array_quantities_dict[species_name])
            if self.diagnostic.select is not None:
                quantities.update(self.diagnostic.select)
            if "gamma" in quantities:
                quantities.add("inv_gamma")
            for quantity in quantities:
                if quantity in ("gamma", "id", "charge", "sx", "sy", "sz"):
                    continue
                setattr(snapshot_species, quantity, self.snapshot_pool.copy(
                    getattr(species, quantity),
                    ("particle", species_name, quantity)))
            if species.tracker is not None:
                snapshot_species.tracker = copy.copy(species.tracker)
                snapshot_species.tracker.id = self.snapshot_pool.copy(
                    species.tracker.id, ("particle", species_name, "id"))
            if species.ionizer is not None:
                snapshot_species.ionizer = copy.copy(species.ionizer)
                snapshot_species.ionizer.ionization_level = (
                    self.snapshot_pool.copy(
                        species.ionizer.ionization_level,
                        ("particle", species_name, "charge")))
            if species.spin_tracker is not None:
                snapshot_species.spin_tracker = copy.copy(
                    species.spin_tracker)
                for quantity in ("sx", "sy", "sz"):
                    setattr(snapshot_species.spin_tracker, quantity,
                            self.snapshot_pool.copy(
                                getattr(species.spin_tracker, quantity),
                                ("particle", species_name, quantity)))
            snapshot.species_dict[species_name] = snapshot_species

        self.writer.submit(snapshot, iteration)


class _SingleRankCommunicatorSnapshot:
    """Freeze physical-grid geometry and guard cropping for one output."""

    def __init__(self, comm):
        if comm.size != 1:
            raise NotImplementedError(
                "asynchronous diagnostics currently support single-rank "
                "output only")
        self.rank = comm.rank
        self.size = comm.size
        self._zmin, self._zmax = comm.get_zmin_zmax(
            local=False, with_damp=False, with_guard=False)
        self._Nz, _ = comm.get_Nz_and_iz(
            local=False, with_damp=False, with_guard=False)
        self._Nr = comm.get_Nr(with_damp=False)
        Nz_local, iz_start_local_domain = comm.get_Nz_and_iz(
            local=True, with_damp=False, with_guard=False, rank=comm.rank)
        _, iz_start_local_array = comm.get_Nz_and_iz(
            local=True, with_damp=True, with_guard=True, rank=comm.rank)
        self._iz_in_array = iz_start_local_domain - iz_start_local_array
        self._Nz_local = Nz_local

    def get_zmin_zmax(self, local, with_damp, with_guard, rank=None):
        if local or with_damp or with_guard or rank is not None:
            raise ValueError("only global physical geometry is available")
        return self._zmin, self._zmax

    def get_Nz_and_iz(self, local, with_damp, with_guard, rank=None):
        if local or with_damp or with_guard or rank is not None:
            raise ValueError("only global physical geometry is available")
        return self._Nz, 0

    def get_Nr(self, with_damp):
        if with_damp:
            raise ValueError("only physical radial geometry is available")
        return self._Nr

    def gather_grid_array(self, array, root=0, with_damp=False):
        if root != self.rank or with_damp:
            raise ValueError("only root physical-grid gathering is available")
        return np.ascontiguousarray(array[
            self._iz_in_array:self._iz_in_array + self._Nz_local, :self._Nr])


class _ParticleChargeDensitySnapshot:
    def __init__(self, diagnostic, species_grids):
        self.diagnostic = diagnostic
        self.species_grids = species_grids

    def write_hdf5(self, iteration):
        diagnostic = self.diagnostic
        fullpath = diagnostic._create_density_file(iteration)
        for species_name, grids in self.species_grids:
            diagnostic.fld.interp = grids
            diagnostic._write_density_dataset(
                fullpath, iteration, species_name)


class _AsyncParticleChargeDensityDiagnostic:
    def __init__(self, diagnostic, writer, snapshot_pool):
        self.diagnostic = diagnostic
        self.writer = writer
        self.snapshot_pool = snapshot_pool

    def write(self, iteration):
        if not _is_output_iteration(self.diagnostic, iteration):
            return
        self.writer.begin_iteration(iteration)
        diagnostic = self.diagnostic
        species_grids = []
        for species_name, species in diagnostic.species.items():
            diagnostic.sim.deposit(
                "rho_next", species_list=[species],
                update_spectral=True, exchange=False)
            diagnostic.fld.spect2interp("rho_next")
            grids = []
            for mode, grid in enumerate(diagnostic.fld.interp):
                snapshot_grid = copy.copy(grid)
                snapshot_grid.use_cuda = False
                snapshot_grid.rho = self.snapshot_pool.copy(
                    grid.rho, ("particle_density", species_name, mode))
                grids.append(snapshot_grid)
            species_grids.append((species_name, grids))

        snapshot_diagnostic = copy.copy(diagnostic)
        snapshot_diagnostic.sim = None
        comm = getattr(diagnostic, "comm", None)
        if comm is not None:
            snapshot_diagnostic.comm = _SingleRankCommunicatorSnapshot(
                comm)
        snapshot_diagnostic.fld = copy.copy(diagnostic.fld)
        snapshot_diagnostic.fld.use_cuda = False
        snapshot_diagnostic.fld.interp = species_grids[0][1]
        snapshot = _ParticleChargeDensitySnapshot(
            snapshot_diagnostic, species_grids)
        self.writer.submit(snapshot, iteration)


class AsyncDiagnosticGroup:
    """Serialize field, particle, and charge-density snapshot writes."""

    def __init__(self, diagnostics, use_pinned_memory=True):
        diagnostics = list(diagnostics)
        if not diagnostics:
            raise ValueError(
                "asynchronous diagnostics require at least one diagnostic")
        if world_comm.size > 1:
            raise NotImplementedError(
                "asynchronous diagnostics currently support single-rank "
                "output only")
        supported_types = (
            FieldDiagnostic,
            ParticleDiagnostic,
            ParticleChargeDensityDiagnostic)
        for diagnostic in diagnostics:
            if type(diagnostic) not in supported_types:
                raise TypeError(
                    "asynchronous output supports FieldDiagnostic, "
                    "ParticleDiagnostic, or ParticleChargeDensityDiagnostic")
            if diagnostic.comm is not None and diagnostic.comm.size > 1:
                raise NotImplementedError(
                    "asynchronous diagnostics currently support single-rank "
                    "output only")
            if (type(diagnostic) is ParticleDiagnostic
                    and diagnostic.subsampling_fraction is not None):
                raise NotImplementedError(
                    "asynchronous diagnostics do not yet support particle "
                    "subsampling")

        self.writer = _SerialDiagnosticWriter()
        self._fbpic_async_diagnostic = True
        self.diagnostics = []
        for diagnostic in diagnostics:
            # Each queued diagnostic owns distinct mutable snapshot storage.
            pool = SnapshotPool(pinned=use_pinned_memory)
            if type(diagnostic) is ParticleChargeDensityDiagnostic:
                wrapper = _AsyncParticleChargeDensityDiagnostic(
                    diagnostic, self.writer, pool)
            elif type(diagnostic) is FieldDiagnostic:
                wrapper = _AsyncFieldDiagnostic(diagnostic, self.writer, pool)
            else:
                wrapper = _AsyncParticleDiagnostic(
                    diagnostic, self.writer, pool)
            self.diagnostics.append(wrapper)

    def write(self, iteration):
        for diagnostic in self.diagnostics:
            diagnostic.write(iteration)

    def flush(self):
        self.writer.flush()

    def close(self):
        try:
            self.writer.close()
        finally:
            for diagnostic in self.diagnostics:
                diagnostic.snapshot_pool.buffers.clear()


def asynchronous_diagnostics(diagnostics, use_pinned_memory=True):
    """Return a diagnostic list that snapshots GPU output asynchronously.

    Parameters
    ----------
    diagnostics : iterable
        Standard field, particle, or particle charge-density diagnostics on a
        single MPI rank. Subclasses and particle subsampling are unsupported.
        Keep other diagnostics outside this group.
    use_pinned_memory : bool, optional
        Reuse pinned host buffers for GPU snapshots. Defaults to True.

    Returns
    -------
    list of AsyncDiagnosticGroup
        Assign this list to ``sim.diags``. ``Simulation.step`` flushes pending
        writes before returning. Call the group's ``close()`` when discarding
        it to join the worker and release snapshot buffers.
    """
    return [AsyncDiagnosticGroup(
        diagnostics, use_pinned_memory=use_pinned_memory)]
