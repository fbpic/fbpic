import pytest

from fbpic.main import _preserve_gpu_residency


class Diagnostic:
    def __init__(self, marked, error=None):
        self._fbpic_async_diagnostic = marked
        self.error = error
        self.flush_calls = 0

    def flush(self):
        self.flush_calls += 1
        if self.error is not None:
            raise self.error


class Simulation:
    use_cuda = False

    def __init__(self, diagnostics):
        self.diags = diagnostics

    @_preserve_gpu_residency
    def succeed(self):
        return "result"

    @_preserve_gpu_residency
    def fail(self):
        raise ValueError("PIC step failed")


def test_only_marked_async_diagnostics_are_flushed():
    asynchronous = Diagnostic(marked=True)
    third_party = Diagnostic(marked=False)
    simulation = Simulation([asynchronous, third_party])

    assert simulation.succeed() == "result"
    assert asynchronous.flush_calls == 1
    assert third_party.flush_calls == 0


def test_async_diagnostics_flush_when_step_raises():
    asynchronous = Diagnostic(marked=True)
    simulation = Simulation([asynchronous])

    with pytest.raises(ValueError, match="PIC step failed"):
        simulation.fail()
    assert asynchronous.flush_calls == 1


def test_step_and_flush_errors_are_both_preserved():
    asynchronous = Diagnostic(
        marked=True, error=RuntimeError("diagnostic write failed"))
    simulation = Simulation([asynchronous])

    with pytest.raises(ValueError, match="PIC step failed") as exc_info:
        simulation.fail()
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "diagnostic write failed"


def test_later_async_groups_flush_after_an_earlier_group_fails():
    first = Diagnostic(marked=True, error=RuntimeError("first failed"))
    second = Diagnostic(marked=True)
    simulation = Simulation([first, second])

    with pytest.raises(RuntimeError, match="first failed"):
        simulation.succeed()

    assert first.flush_calls == 1
    assert second.flush_calls == 1


def test_multiple_group_flush_errors_are_aggregated():
    first_error = RuntimeError("first failed")
    second_error = ValueError("second failed")
    first = Diagnostic(marked=True, error=first_error)
    second = Diagnostic(marked=True, error=second_error)
    simulation = Simulation([first, second])

    with pytest.raises(
            RuntimeError,
            match="multiple asynchronous diagnostic groups") as exc_info:
        simulation.succeed()

    assert exc_info.value.errors == (first_error, second_error)
    assert first.flush_calls == 1
    assert second.flush_calls == 1
