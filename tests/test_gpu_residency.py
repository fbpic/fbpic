"""CPU-safe tests for simulation GPU-residency ownership."""

import pytest

from fbpic.main import Simulation, _preserve_gpu_residency
from fbpic.particles.injection.continuous_injection import ContinuousInjector
from fbpic.utils.cuda import GpuMemoryManager


class RecordingData(object):
    """Minimal field/species data owner with observable residency changes."""

    def __init__(self, events, name, on_gpu=False):
        self.events = events
        self.name = name
        self.data_is_on_gpu = on_gpu

    def send_fields_to_gpu(self):
        self.events.append("send-" + self.name)
        self.data_is_on_gpu = True

    def receive_fields_from_gpu(self):
        self.events.append("receive-" + self.name)
        self.data_is_on_gpu = False

    send_particles_to_gpu = send_fields_to_gpu
    receive_particles_from_gpu = receive_fields_from_gpu


class RecordingSimulation(object):
    """Minimal simulation shape consumed by ``GpuMemoryManager``."""

    def __init__(self, field_on_gpu=False, species_on_gpu=False):
        self.events = []
        self.use_cuda = True
        self.fld = RecordingData(self.events, "fields", field_on_gpu)
        self.ptcl = [RecordingData(
            self.events, "species", species_on_gpu)]


def test_gpu_memory_manager_returns_itself_and_restores_cpu_state():
    """Catch a missing context value or an omitted owned-state restoration."""
    sim = RecordingSimulation()
    manager = GpuMemoryManager(sim)

    with manager as entered:
        assert entered is manager
        assert sim.fld.data_is_on_gpu
        assert sim.ptcl[0].data_is_on_gpu

    assert sim.events == [
        "send-fields", "send-species",
        "receive-fields", "receive-species",
    ]


def test_gpu_memory_manager_preserves_preexisting_gpu_state():
    """Catch a nested manager that downloads state owned by an outer scope."""
    sim = RecordingSimulation(field_on_gpu=True, species_on_gpu=True)

    with GpuMemoryManager(sim):
        pass

    assert sim.events == []
    assert sim.fld.data_is_on_gpu
    assert sim.ptcl[0].data_is_on_gpu


def test_gpu_memory_manager_restores_state_after_exception():
    """Catch an exceptional exit that strands originally CPU-owned data."""
    sim = RecordingSimulation()

    with pytest.raises(RuntimeError, match="injected"):
        with GpuMemoryManager(sim):
            raise RuntimeError("injected")

    assert not sim.fld.data_is_on_gpu
    assert not sim.ptcl[0].data_is_on_gpu


class RecordingStepper(RecordingSimulation):
    """Exercise the real residency wrapper without constructing a PIC grid."""

    @_preserve_gpu_residency
    def advance(self, fail=False):
        self.events.append((
            "body", self.fld.data_is_on_gpu,
            self.ptcl[0].data_is_on_gpu))
        if fail:
            raise RuntimeError("injected")


def test_wrapped_cuda_call_retains_default_transfer_boundary():
    """Catch a default CUDA call that stops restoring host-visible state."""
    sim = RecordingStepper()

    sim.advance()

    assert sim.events == [
        "send-fields", "send-species", ("body", True, True),
        "receive-fields", "receive-species",
    ]


def test_public_scope_amortizes_transfers_across_wrapped_calls():
    """Catch per-call transfers that defeat an outer residency scope."""
    sim = RecordingStepper()

    with Simulation.gpu_resident(sim):
        sim.advance()
        sim.advance()

    assert sim.events == [
        "send-fields", "send-species",
        ("body", True, True), ("body", True, True),
        "receive-fields", "receive-species",
    ]


def test_public_scope_is_nested_and_exception_safe():
    """Catch duplicate transfers or stranded state during nested unwinding."""
    sim = RecordingStepper()

    with pytest.raises(RuntimeError, match="injected"):
        with Simulation.gpu_resident(sim):
            with Simulation.gpu_resident(sim):
                sim.advance(fail=True)

    assert sim.events.count("send-fields") == 1
    assert sim.events.count("receive-fields") == 1
    assert not sim.fld.data_is_on_gpu


def test_cpu_execution_bypasses_manager_and_public_scope_rejects_it():
    """
    Catch accidental CUDA-manager use or silent no-op optimization on CPU.
    """
    sim = RecordingStepper()
    sim.use_cuda = False

    sim.advance()

    assert sim.events == [("body", False, False)]
    with pytest.raises(RuntimeError, match="use_cuda=True"):
        Simulation.gpu_resident(sim)


def test_continuous_injector_stores_device_reduction_as_python_scalar():
    """Catch direct arithmetic on the CuPy scalar returned by ``max``."""
    class DeviceScalar(object):
        def item(self):
            return 2.0

        def __add__(self, unused):
            raise AssertionError("device scalar must be converted explicitly")

    class DeviceArray(object):
        def __len__(self):
            return 1

        def max(self):
            return DeviceScalar()

    class Communicator(object):
        rank = 0
        size = 1
        n_inject = 2
        dz = 0.1
        exchange_period = 1

        def get_zmin_zmax(self, local, with_damp, with_guard):
            return 0.0, 1.0

    injector = ContinuousInjector(
        1, 0.0, 1.0, None, 1, 0.0, 1.0, 1, 1.0, None,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    injector.initialize_injection_positions(
        Communicator(), 0.0, DeviceArray(), 0.1)

    assert injector.z_end_plasma == 2.5
    assert isinstance(injector.z_end_plasma, float)
