# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""CUDA launch-policy selection independent of CUDA device discovery.

The policy is selected from the device compute capability.  Device names are
intentionally not consulted here: they are reporting metadata, not a stable
input to launch configuration.
"""

from collections import namedtuple
import numbers
import os


CudaLaunchConfig = namedtuple(
    "CudaLaunchConfig",
    [
        "architecture",
        "compute_capability",
        "deposit_tpb_linear",
        "deposit_tpb_cubic",
        "gather_tpb_linear",
        "gather_tpb_cubic",
        "copy_tpb",
        "unsorted_j_tpb",
        "unsorted_rho_tpb",
        "fft_cuda_graphs",
    ],
)


def _architecture_for_capability(compute_capability):
    """Return the reporting class associated with a capability tuple."""
    major = compute_capability[0]

    if major == 6:
        return "pascal"
    if major == 7:
        return "volta"
    if major == 8:
        return "ampere"
    if major == 9:
        return "hopper"
    if 10 <= major <= 12:
        return "blackwell"
    if major >= 13:
        return "future"
    return "generic"


def _positive_integer(value, name):
    """Return *value* as an integer, rejecting invalid launch limits."""
    if (isinstance(value, bool) or
            not isinstance(value, numbers.Integral) or value <= 0):
        raise ValueError("%s must be a positive integer" % name)
    return int(value)


def _parse_integer_override(value, name):
    """Parse one integer-valued environment override."""
    if (isinstance(value, bool) or
            not isinstance(value, (str, bytes, numbers.Integral))):
        raise ValueError("%s must be an integer" % name)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be an integer" % name)
    return _positive_integer(parsed, name)


def _parse_copy_override(value):
    """Parse the ``first,second`` copy-tile environment override."""
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            raise ValueError("FBPIC_CUDA_COPY_TPB must be a 2D tile")
    try:
        dimensions = value.split(",")
    except AttributeError:
        raise ValueError("FBPIC_CUDA_COPY_TPB must be a 2D tile")
    if len(dimensions) != 2:
        raise ValueError("FBPIC_CUDA_COPY_TPB must be a 2D tile")
    return tuple(
        _parse_integer_override(dimension, "FBPIC_CUDA_COPY_TPB")
        for dimension in dimensions)


def _parse_boolean_override(value, name):
    """Parse a strict ``0`` or ``1`` environment override."""
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            raise ValueError("%s must be 0 or 1" % name)
    if value == "0" or value == 0:
        return False
    if value == "1" or value == 1:
        return True
    raise ValueError("%s must be 0 or 1" % name)


def _bound_copy_tpb(copy_tpb, max_threads_per_block):
    """Fit an automatic 2D copy tile within the device thread limit."""
    first, second = copy_tpb
    first = min(first, max_threads_per_block)
    second = min(second, max_threads_per_block // first)
    return (first, max(second, 1))


def _validate_launch_config_values(
        deposit_tpb_linear, deposit_tpb_cubic, gather_tpb_linear,
        gather_tpb_cubic, copy_tpb, unsorted_j_tpb, unsorted_rho_tpb,
        max_threads_per_block, warp_size):
    """Validate all launch dimensions after defaults and overrides resolve."""
    one_dimensional = {
        "deposit_tpb_linear": deposit_tpb_linear,
        "deposit_tpb_cubic": deposit_tpb_cubic,
        "gather_tpb_linear": gather_tpb_linear,
        "gather_tpb_cubic": gather_tpb_cubic,
        "unsorted_j_tpb": unsorted_j_tpb,
        "unsorted_rho_tpb": unsorted_rho_tpb,
    }
    for name, value in one_dimensional.items():
        value = _positive_integer(value, name)
        if value > max_threads_per_block:
            raise ValueError("%s exceeds max_threads_per_block" % name)

    if (not isinstance(copy_tpb, (tuple, list)) or len(copy_tpb) != 2):
        raise ValueError("copy_tpb must be a 2D tile")
    copy_tpb = tuple(
        _positive_integer(value, "copy_tpb dimension") for value in copy_tpb)
    if copy_tpb[0] * copy_tpb[1] > max_threads_per_block:
        raise ValueError("copy_tpb exceeds max_threads_per_block")

    return copy_tpb


def select_cuda_launch_config(
        compute_capability, max_threads_per_block=1024, warp_size=32,
        environ=None):
    """Select deterministic CUDA launch settings for a device capability.

    Device limits and environment overrides are supplied explicitly so the
    policy can be tested without CUDA. Pascal and older devices use the legacy
    generic settings; Volta through Hopper use the legacy data-center tile.
    Blackwell selects a 32-thread linear-deposition block. Unknown future
    architectures retain the conservative modern default. These defaults do
    not replace measurements on the target device and workload.
    """
    try:
        compute_capability = tuple(compute_capability)
    except TypeError:
        raise ValueError("compute_capability must contain two integers")
    if (len(compute_capability) != 2 or
            any(isinstance(value, bool) or
                not isinstance(value, numbers.Integral) or value < 0
                for value in compute_capability)):
        raise ValueError("compute_capability must contain two integers")
    compute_capability = tuple(int(value) for value in compute_capability)

    max_threads_per_block = _positive_integer(
        max_threads_per_block, "max_threads_per_block")
    warp_size = _positive_integer(warp_size, "warp_size")

    architecture = _architecture_for_capability(compute_capability)

    if architecture == "blackwell":
        deposit_tpb_linear = 32
        copy_tpb = (8, 32)
    elif compute_capability[0] >= 7:
        deposit_tpb_linear = 16
        copy_tpb = (8, 32)
    else:
        deposit_tpb_linear = 8
        copy_tpb = (2, 16)

    if architecture in ("hopper", "blackwell"):
        unsorted_j_tpb = 64
        unsorted_rho_tpb = 64
    else:
        unsorted_j_tpb = 128
        unsorted_rho_tpb = 128
    # CUDA Graph replay is validated on Hopper. Blackwell retains the
    # previously validated transform path until it is measured independently.
    fft_cuda_graphs = architecture == "hopper"

    deposit_tpb_linear = min(deposit_tpb_linear, max_threads_per_block)
    deposit_tpb_cubic = min(32, max_threads_per_block)
    gather_tpb_linear = min(128, max_threads_per_block)
    gather_tpb_cubic = min(256, max_threads_per_block)

    copy_tpb = _bound_copy_tpb(copy_tpb, max_threads_per_block)

    environment = os.environ if environ is None else environ
    override_names = {
        "FBPIC_CUDA_DEPOSIT_TPB_LINEAR": "deposit_tpb_linear",
        "FBPIC_CUDA_DEPOSIT_TPB_CUBIC": "deposit_tpb_cubic",
        "FBPIC_CUDA_GATHER_TPB_LINEAR": "gather_tpb_linear",
        "FBPIC_CUDA_GATHER_TPB_CUBIC": "gather_tpb_cubic",
    }
    resolved = {
        "deposit_tpb_linear": deposit_tpb_linear,
        "deposit_tpb_cubic": deposit_tpb_cubic,
        "gather_tpb_linear": gather_tpb_linear,
        "gather_tpb_cubic": gather_tpb_cubic,
        "unsorted_j_tpb": min(unsorted_j_tpb, max_threads_per_block),
        "unsorted_rho_tpb": min(unsorted_rho_tpb, max_threads_per_block),
    }
    for environment_name, field_name in override_names.items():
        if environment_name in environment:
            value = _parse_integer_override(
                environment[environment_name], environment_name)
            if (field_name.startswith("gather_tpb") and
                    value % warp_size != 0):
                raise ValueError("%s must be a warp multiple" % field_name)
            resolved[field_name] = value
    unsorted_override_names = {
        "FBPIC_EXPERIMENT_UNSORTED_J_TPB": "unsorted_j_tpb",
        "FBPIC_EXPERIMENT_UNSORTED_RHO_TPB": "unsorted_rho_tpb",
    }
    for environment_name, field_name in unsorted_override_names.items():
        if environment_name in environment:
            resolved[field_name] = _parse_integer_override(
                environment[environment_name], environment_name)
    if "FBPIC_CUDA_FFT_GRAPHS" in environment:
        fft_cuda_graphs = _parse_boolean_override(
            environment["FBPIC_CUDA_FFT_GRAPHS"],
            "FBPIC_CUDA_FFT_GRAPHS")
    if "FBPIC_CUDA_COPY_TPB" in environment:
        copy_tpb = _parse_copy_override(environment["FBPIC_CUDA_COPY_TPB"])

    copy_tpb = _validate_launch_config_values(
        resolved["deposit_tpb_linear"], resolved["deposit_tpb_cubic"],
        resolved["gather_tpb_linear"], resolved["gather_tpb_cubic"],
        copy_tpb, resolved["unsorted_j_tpb"],
        resolved["unsorted_rho_tpb"], max_threads_per_block, warp_size)

    return CudaLaunchConfig(
        architecture=architecture,
        compute_capability=compute_capability,
        deposit_tpb_linear=resolved["deposit_tpb_linear"],
        deposit_tpb_cubic=resolved["deposit_tpb_cubic"],
        gather_tpb_linear=resolved["gather_tpb_linear"],
        gather_tpb_cubic=resolved["gather_tpb_cubic"],
        copy_tpb=copy_tpb,
        unsorted_j_tpb=resolved["unsorted_j_tpb"],
        unsorted_rho_tpb=resolved["unsorted_rho_tpb"],
        fft_cuda_graphs=fft_cuda_graphs,
    )
