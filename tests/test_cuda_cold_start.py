"""PTX-only compilation must preserve options and retain a safe fallback."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import numpy as np
from numba import cuda
from numba.cuda import dispatcher as cuda_dispatcher

from fbpic.utils import cuda as cuda_utils


def install_compiler(monkeypatch, failure=None):
    events = []
    options = {'debug': False, 'fastmath': False, 'opt': True}

    @contextmanager
    def lock():
        events.append('lock')
        yield
        events.append('unlock')

    def definition(func, types, **received):
        assert events[-1] == 'lock'
        assert types == ('type-of-input',)
        assert received == options
        if failure == 'compiler':
            raise AttributeError('changed private API')
        events.append('compile')
        # No bind() method: extracting PTX must not create a Numba GPU module.
        return SimpleNamespace(
            entry_name='kernel',
            _codelibrary=SimpleNamespace(get_asm_str=lambda: '.version 8.8\n'))

    class Module:
        def load(self, ptx):
            assert events[-1] == 'unlock'
            assert ptx == b'.version 8.8\n'
            if failure == 'load':
                raise RuntimeError('module cannot load')
            events.append('load')

        def get_function(self, name):
            assert name == 'kernel'
            return 'cupy-kernel'

    monkeypatch.setattr(cuda_utils, 'numba_version', (0, 67, 0))
    monkeypatch.setattr(cuda_dispatcher, 'global_compiler_lock', lock())
    monkeypatch.setattr(cuda_dispatcher, '_Kernel', definition)
    monkeypatch.setattr(cuda_utils.cuda, 'jit', lambda: lambda func:
                        SimpleNamespace(
                            typeof_pyval=lambda arg: 'type-of-' + arg,
                            targetoptions=options))
    monkeypatch.setattr(
        cuda_utils, 'cupy', SimpleNamespace(
            cuda=SimpleNamespace(
                function=SimpleNamespace(Module=Module))), raising=False)
    return events


def test_ptx_only_preserves_options_and_skips_binary_binding(monkeypatch):
    events = install_compiler(monkeypatch)
    result = cuda_utils._compile_cupy_ptx(lambda: None, ('input',))
    assert result == ('cupy-kernel', '.version 8.8\n', 'kernel')
    assert events == ['lock', 'compile', 'unlock', 'load']


@pytest.mark.parametrize('failure', ['compiler', 'load'])
def test_optional_ptx_path_returns_to_regular_compiler_on_failure(
        monkeypatch, failure):
    install_compiler(monkeypatch, failure)
    assert cuda_utils._compile_cupy_ptx(lambda: None, ('input',)) is None


@pytest.mark.parametrize('version', [(0, 55, 2), (0, 56, 0)])
def test_unsupported_versions_keep_regular_compiler(monkeypatch, version):
    events = install_compiler(monkeypatch)
    monkeypatch.setattr(cuda_utils, 'numba_version', version)
    assert cuda_utils._compile_cupy_ptx(lambda: None, ('input',)) is None
    assert events == []


def test_ptx_only_can_be_disabled(monkeypatch):
    events = install_compiler(monkeypatch)
    monkeypatch.setenv('FBPIC_DISABLE_CUDA_PTX_ONLY', '1')
    assert cuda_utils._compile_cupy_ptx(lambda: None, ('input',)) is None
    assert events == []


def _arithmetic_kernel(source, destination, n, scale):
    i = cuda.grid(1)
    if i < n:
        destination[i] = scale * source[i] + source[i] * source[i]


@pytest.mark.skipif(not cuda_utils.cuda_installed,
                    reason='requires CUDA and CuPy')
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('stride', [1, 2])
def test_real_ptx_cache_reload_and_regular_fallback(
        monkeypatch, tmp_path, dtype, stride):
    import cupy

    monkeypatch.delenv('FBPIC_DISABLE_CUDA_KERNEL_CACHE', raising=False)
    monkeypatch.delenv('FBPIC_DISABLE_CUDA_PTX_ONLY', raising=False)
    monkeypatch.setenv('FBPIC_CUDA_KERNEL_CACHE_DIR', str(tmp_path))
    original = cuda_utils._compile_cupy_ptx
    attempts = []

    def record(*args):
        result = original(*args)
        attempts.append(result is not None)
        return result

    monkeypatch.setattr(cuda_utils, '_compile_cupy_ptx', record)
    n = 257
    host = np.arange(n * stride, dtype=np.float64).astype(dtype)
    if dtype == np.complex128:
        host += 0.5j * host
    source = cupy.asarray(host)[::stride]
    expected = 2. * host[::stride] + host[::stride] * host[::stride]
    outputs = []
    for mode in ('ptx', 'cached', 'regular'):
        if mode == 'regular':
            monkeypatch.setenv('FBPIC_DISABLE_CUDA_PTX_ONLY', '1')
            monkeypatch.setenv('FBPIC_DISABLE_CUDA_KERNEL_CACHE', '1')
        destination = cupy.zeros(n * stride, dtype=dtype)[::stride]
        kernel = cuda_utils.compile_cupy(_arithmetic_kernel)
        kernel[(n + 127) // 128, 128](source, destination, n, 2.)
        cuda.synchronize()
        outputs.append(destination.get())
    assert attempts == [True, False]  # Cache reload needs no compilation.
    assert len(list(tmp_path.glob('*.json'))) == 1
    for output in outputs:
        np.testing.assert_array_equal(output, expected)
