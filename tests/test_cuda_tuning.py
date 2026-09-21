import pytest

import fbpic.utils.cuda as cuda_utils
from fbpic.utils.cuda import get_cuda_launch_config, legacy_cuda_gpu_model
from fbpic.utils.cuda_tuning import select_cuda_launch_config


@pytest.mark.parametrize(
    "compute_capability, architecture, deposit_tpb, copy_tpb",
    [
        ((6, 0), "pascal", 8, (2, 16)),
        ((7, 0), "volta", 16, (8, 32)),
        ((8, 0), "ampere", 16, (8, 32)),
        ((9, 0), "hopper", 16, (8, 32)),
        ((10, 0), "blackwell", 32, (8, 32)),
        ((10, 3), "blackwell", 32, (8, 32)),
        ((12, 0), "blackwell", 32, (8, 32)),
        ((13, 0), "future", 16, (8, 32)),
    ],
)
def test_selects_safe_policy_from_compute_capability(
        compute_capability, architecture, deposit_tpb, copy_tpb):
    config = select_cuda_launch_config(compute_capability)

    assert config.architecture == architecture
    assert config.deposit_tpb_linear == deposit_tpb
    assert config.copy_tpb == copy_tpb


def test_bounds_modern_policy_by_device_thread_limit():
    config = select_cuda_launch_config(
        (13, 0), max_threads_per_block=128)

    assert config.gather_tpb_cubic == 128
    assert config.copy_tpb == (8, 16)
    assert config.copy_tpb[0] * config.copy_tpb[1] <= 128


@pytest.mark.parametrize(
    "max_threads, expected_copy_tpb",
    [(1, (1, 1)), (4, (4, 1)), (7, (7, 1))],
)
def test_bounds_modern_copy_tile_below_first_dimension(
        max_threads, expected_copy_tpb):
    config = select_cuda_launch_config(
        (13, 0), max_threads_per_block=max_threads, warp_size=1)

    assert config.copy_tpb == expected_copy_tpb
    assert config.copy_tpb[0] * config.copy_tpb[1] <= max_threads


@pytest.mark.parametrize(
    "capability", [(), (13,), (13, 0, 1), ("13", 0), (-1, 0), None])
def test_rejects_malformed_compute_capability(capability):
    with pytest.raises(ValueError):
        select_cuda_launch_config(capability)


@pytest.mark.parametrize("max_threads", [0, -1, 32.0, True])
def test_rejects_malformed_device_thread_limit(max_threads):
    with pytest.raises(ValueError):
        select_cuda_launch_config((13, 0), max_threads_per_block=max_threads)


@pytest.mark.parametrize("warp_size", [0, -1, 32.0, True])
def test_rejects_malformed_warp_size(warp_size):
    with pytest.raises(ValueError):
        select_cuda_launch_config((13, 0), warp_size=warp_size)


def test_valid_environment_overrides_replace_auto_policy():
    config = select_cuda_launch_config(
        (9, 0),
        environ={
            "FBPIC_CUDA_DEPOSIT_TPB_LINEAR": "32",
            "FBPIC_CUDA_COPY_TPB": "16,16",
        },
    )

    assert config.deposit_tpb_linear == 32
    assert config.copy_tpb == (16, 16)


def test_bytes_environment_copy_tile_override_is_supported():
    config = select_cuda_launch_config(
        (9, 0), environ={"FBPIC_CUDA_COPY_TPB": b"16,16"})

    assert config.copy_tpb == (16, 16)


def test_none_environment_reads_process_environment(monkeypatch):
    monkeypatch.setenv("FBPIC_CUDA_DEPOSIT_TPB_LINEAR", "32")

    config = select_cuda_launch_config((9, 0))

    assert config.deposit_tpb_linear == 32


def test_environment_override_accepts_cubic_and_gather_values():
    config = select_cuda_launch_config(
        (9, 0),
        environ={
            "FBPIC_CUDA_DEPOSIT_TPB_CUBIC": "64",
            "FBPIC_CUDA_GATHER_TPB_LINEAR": "64",
            "FBPIC_CUDA_GATHER_TPB_CUBIC": "128",
        },
    )

    assert config.deposit_tpb_cubic == 64
    assert config.gather_tpb_linear == 64
    assert config.gather_tpb_cubic == 128


@pytest.mark.parametrize(
    "name, value", [
        ("FBPIC_CUDA_DEPOSIT_TPB_LINEAR", "not-an-integer"),
        ("FBPIC_CUDA_COPY_TPB", "16"),
        ("FBPIC_CUDA_COPY_TPB", "16,16,16"),
    ])
def test_rejects_malformed_environment_overrides(name, value):
    with pytest.raises(ValueError):
        select_cuda_launch_config((9, 0), environ={name: value})


def test_rejects_environment_block_above_device_limit():
    with pytest.raises(ValueError):
        select_cuda_launch_config(
            (9, 0), max_threads_per_block=128,
            environ={"FBPIC_CUDA_DEPOSIT_TPB_LINEAR": "256"})


def test_rejects_environment_gather_block_not_warp_multiple():
    with pytest.raises(ValueError):
        select_cuda_launch_config(
            (9, 0), environ={"FBPIC_CUDA_GATHER_TPB_LINEAR": "48"})


class HopperDevice:
    compute_capability = (9, 0)
    MAX_THREADS_PER_BLOCK = 1024
    WARP_SIZE = 32


def test_device_adapter_selects_hopper_policy_from_numba_properties():
    config = get_cuda_launch_config(device=HopperDevice(), environ={})

    assert config.architecture == "hopper"
    assert config.compute_capability == (9, 0)
    assert config.deposit_tpb_linear == 16
    assert config.deposit_tpb_cubic == 32
    assert config.gather_tpb_linear == 128
    assert config.gather_tpb_cubic == 256
    assert config.copy_tpb == (8, 32)
    assert config.unsorted_j_tpb == 64
    assert config.unsorted_rho_tpb == 64
    assert config.fft_cuda_graphs is True


def test_blackwell_preserves_validated_transform_policy():
    config = select_cuda_launch_config((10, 0), environ={})

    assert config.architecture == "blackwell"
    assert config.deposit_tpb_linear == 32
    assert config.unsorted_j_tpb == 64
    assert config.unsorted_rho_tpb == 64
    assert config.fft_cuda_graphs is False


def test_future_gpu_uses_conservative_deposition_and_transform_policy():
    config = select_cuda_launch_config((13, 0), environ={})

    assert config.architecture == "future"
    assert config.unsorted_j_tpb == 128
    assert config.unsorted_rho_tpb == 128
    assert config.fft_cuda_graphs is False


def test_architecture_policy_overrides_remain_authoritative():
    config = select_cuda_launch_config(
        (9, 0),
        environ={
            "FBPIC_EXPERIMENT_UNSORTED_J_TPB": "96",
            "FBPIC_EXPERIMENT_UNSORTED_RHO_TPB": "32",
            "FBPIC_CUDA_FFT_GRAPHS": "0",
        },
    )

    assert config.unsorted_j_tpb == 96
    assert config.unsorted_rho_tpb == 32
    assert config.fft_cuda_graphs is False


@pytest.mark.parametrize(
    "name, value",
    [
        ("FBPIC_EXPERIMENT_UNSORTED_J_TPB", "invalid"),
        ("FBPIC_EXPERIMENT_UNSORTED_RHO_TPB", "0"),
        ("FBPIC_CUDA_FFT_GRAPHS", "yes"),
    ],
)
def test_rejects_invalid_architecture_policy_overrides(name, value):
    with pytest.raises(ValueError):
        select_cuda_launch_config((9, 0), environ={name: value})


@pytest.mark.parametrize(
    "device_name, expected_model",
    [
        (b"Tesla P100-PCIE-16GB", "P100"),
        ("Tesla V100-SXM2-32GB", "V100"),
        ("NVIDIA A100-SXM4-80GB", "V100"),
        ("NVIDIA H100 80GB HBM3", "other"),
    ],
)
def test_legacy_model_is_derived_from_selected_device_name(
        device_name, expected_model):
    class NamedDevice(object):
        name = device_name

    assert legacy_cuda_gpu_model(NamedDevice()) == expected_model


class LowercaseHopperDevice:
    compute_capability = (9, 0)
    max_threads_per_block = 128
    warp_size = 32


def test_device_adapter_accepts_lowercase_numba_properties():
    config = get_cuda_launch_config(device=LowercaseHopperDevice(), environ={})

    assert config.compute_capability == (9, 0)
    assert config.gather_tpb_cubic == 128
    assert config.copy_tpb == (8, 16)


class PascalDevice:
    compute_capability = (6, 0)
    MAX_THREADS_PER_BLOCK = 64
    WARP_SIZE = 32


def test_device_adapter_looks_up_the_current_selected_device(monkeypatch):
    monkeypatch.setattr(cuda_utils.cuda, "get_current_device", PascalDevice)

    config = get_cuda_launch_config(environ={})

    assert config.compute_capability == (6, 0)
    assert config.gather_tpb_cubic == 64
    assert config.copy_tpb == (2, 16)


def test_cuda_module_import_does_not_discover_a_device(monkeypatch):
    import importlib

    def unexpected_device_discovery():
        raise AssertionError("module import must not discover a CUDA device")

    with monkeypatch.context() as patch:
        patch.setattr(cuda_utils.cuda, "is_available", lambda: True)
        patch.setattr(
            cuda_utils.cuda, "get_current_device", unexpected_device_discovery)
        importlib.reload(cuda_utils)
        assert cuda_utils.numba_cuda_installed is True
        assert isinstance(cuda_utils.cuda_installed, bool)
    # Restore the real module globals for later GPU CI tests after the patched
    # import-state assertion has completed.
    importlib.reload(cuda_utils)


class DeviceWithBrokenThreadLimit:
    compute_capability = (9, 0)
    WARP_SIZE = 32

    @property
    def MAX_THREADS_PER_BLOCK(self):
        raise RuntimeError("CUDA driver failure")


def test_device_adapter_preserves_device_property_errors():
    with pytest.raises(RuntimeError, match="CUDA driver failure"):
        get_cuda_launch_config(
            device=DeviceWithBrokenThreadLimit(),
            environ={})


def test_legacy_model_resolves_the_selected_device_lazily(monkeypatch):
    class NamedPascalDevice(object):
        name = b"Tesla P100-PCIE-16GB"

    monkeypatch.setattr(
        cuda_utils.cuda, "get_current_device", NamedPascalDevice)

    assert cuda_utils.cuda_gpu_model == "P100"


def test_lazy_legacy_model_delegates_practical_string_behavior(monkeypatch):
    """Early imported proxy references remain usable like legacy strings."""
    class NamedAmpereDevice(object):
        name = "NVIDIA A100-SXM4-80GB"

    monkeypatch.setattr(
        cuda_utils.cuda, "get_current_device", NamedAmpereDevice)
    model = cuda_utils._LazyCudaGpuModel()

    assert "model=%s" % model == "model=V100"
    assert "{:>6}".format(model) == "  V100"
    assert hash(model) == hash("V100")
    assert model.lower() == "v100"
    assert model + "-policy" == "V100-policy"
    assert "legacy-" + model == "legacy-V100"
    assert "100" in model


def test_mpi_selection_refreshes_module_alias_to_an_actual_string(monkeypatch):
    """
    After device selection new imports receive the resolved legacy string.
    """
    class SelectedDevice(object):
        name = "NVIDIA A100-SXM4-80GB"

    class FakeCuda(object):
        gpus = [object()]

        @staticmethod
        def select_device(index):
            assert index == 0

        @staticmethod
        def get_current_device():
            return SelectedDevice()

    class Communicator(object):
        rank = 0
        size = 1

        def barrier(self):
            pass

        def gather(self, value):
            return [value]

    class Mpi(object):
        COMM_WORLD = Communicator()

    monkeypatch.setattr(cuda_utils, "cuda", FakeCuda())
    monkeypatch.setattr(cuda_utils, "pynvml_installed", False)
    monkeypatch.setattr(
        cuda_utils, "check_consecutive_ranks_on_same_nodes", lambda mpi: True)
    monkeypatch.setattr(cuda_utils, "get_uuid", lambda index: "gpu-0")
    monkeypatch.setattr(
        cuda_utils, "cuda_gpu_model", cuda_utils._LazyCudaGpuModel())

    cuda_utils.mpi_select_gpus(Mpi())

    assert type(cuda_utils.cuda_gpu_model) is str
    assert cuda_utils.cuda_gpu_model == "V100"
