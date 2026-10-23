from types import SimpleNamespace

import numpy as np

from fbpic.utils import cuda as cuda_utils


class FakeArray:
    def __init__(self, dtype=np.float64, ndim=2, c=True, f=False):
        self.dtype = np.dtype(dtype)
        self.ndim = ndim
        self.flags = SimpleNamespace(c_contiguous=c, f_contiguous=f)


class FakeModule:
    def __init__(self):
        self.ptx = None

    def load(self, ptx):
        self.ptx = ptx

    def get_function(self, name):
        return name, self.ptx


def install_fake_cuda(monkeypatch, capability=(12, 0)):
    fake_cupy = SimpleNamespace(
        ndarray=FakeArray,
        __version__="14.2.0",
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(runtimeGetVersion=lambda: 13020),
            function=SimpleNamespace(Module=FakeModule)))
    fake_device = SimpleNamespace(compute_capability=capability)
    monkeypatch.setattr(cuda_utils, "cupy", fake_cupy, raising=False)
    monkeypatch.setattr(
        cuda_utils.cuda, "get_current_device", lambda: fake_device)


def test_cuda_cache_path_is_stable_and_architecture_specific(
        monkeypatch, tmp_path):
    install_fake_cuda(monkeypatch)
    monkeypatch.setenv("FBPIC_CUDA_KERNEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        cuda_utils, "_get_cuda_source_digest", lambda: "source-digest")

    def kernel():
        pass

    first = cuda_utils._cuda_cache_path(
        kernel, (FakeArray(), np.float64(1.0)))
    second = cuda_utils._cuda_cache_path(
        kernel, (FakeArray(), np.float64(1.0)))
    install_fake_cuda(monkeypatch, capability=(9, 0))
    hopper = cuda_utils._cuda_cache_path(
        kernel, (FakeArray(), np.float64(1.0)))
    monkeypatch.setattr(
        cuda_utils, "_get_cuda_source_digest", lambda: "new-source")
    changed_source = cuda_utils._cuda_cache_path(
        kernel, (FakeArray(), np.float64(1.0)))

    assert first == second
    assert first.parent == tmp_path
    assert first != hopper
    assert hopper != changed_source


def test_cuda_kernel_cache_round_trip(monkeypatch, tmp_path):
    install_fake_cuda(monkeypatch)
    cache_path = tmp_path / "kernel.json"

    cuda_utils._store_cuda_kernel_cache(
        cache_path, ".version 8.8\n", "cached_kernel")
    loaded = cuda_utils._load_cuda_kernel_cache(cache_path)

    assert loaded == ("cached_kernel", b".version 8.8\n")


def test_cuda_kernel_cache_can_be_disabled(monkeypatch):
    install_fake_cuda(monkeypatch)
    monkeypatch.setenv("FBPIC_DISABLE_CUDA_KERNEL_CACHE", "1")

    assert cuda_utils._cuda_cache_path(lambda: None, ()) is None


def test_corrupt_cuda_kernel_cache_falls_back(monkeypatch, tmp_path):
    install_fake_cuda(monkeypatch)
    cache_path = tmp_path / "kernel.json"
    cache_path.write_text("not json")

    assert cuda_utils._load_cuda_kernel_cache(cache_path) is None


def test_inaccessible_cuda_kernel_cache_falls_back(monkeypatch, tmp_path):
    cache_path = tmp_path / "kernel.json"

    def inaccessible(_path):
        raise PermissionError("cache metadata is inaccessible")

    monkeypatch.setattr(type(cache_path), "is_file", inaccessible)

    assert cuda_utils._load_cuda_kernel_cache(cache_path) is None


def test_cuda_kernel_cache_creates_private_directory(tmp_path):
    cache_path = tmp_path / "private" / "kernel.json"

    cuda_utils._store_cuda_kernel_cache(
        cache_path, ".version 8.8\n", "cached_kernel")

    assert cache_path.parent.stat().st_mode & 0o077 == 0


def test_in_process_signature_distinguishes_array_layout(monkeypatch):
    install_fake_cuda(monkeypatch)

    c_layout = cuda_utils._cuda_argument_descriptors((
        FakeArray(c=True, f=False),))
    f_layout = cuda_utils._cuda_argument_descriptors((
        FakeArray(c=False, f=True),))

    assert c_layout != f_layout


def test_ptx_cache_extraction_failure_is_fail_open(tmp_path):
    class BrokenNumbaKernel:
        @property
        def overloads(self):
            raise RuntimeError("private Numba API changed")

    cuda_utils._store_compiled_cuda_kernel(
        tmp_path / "kernel.json", BrokenNumbaKernel())
