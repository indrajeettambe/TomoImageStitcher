"""Shared pytest configuration.

The only thing that needs handling here is GPU availability.

``pytest.importorskip("cupy")`` in the test modules skips when CuPy is *not
installed*, which is not the same thing as "no CUDA device present". On a host
that has CuPy installed but no usable GPU — a laptop, a CPU-only CI runner, a
container started without ``--gpus all`` — the import succeeds and the test then
dies at the first kernel launch with::

    cupy_backends.cuda.api.runtime.CUDARuntimeError: cudaErrorNoDevice

so the test *fails* where its docstring promises it is skipped. This hook makes
the skip actually happen, by skipping everything marked ``gpu`` when no device
can be enumerated.
"""
import pytest


def _cuda_device_count() -> int:
    """Number of usable CUDA devices, or 0 if there are none."""
    try:
        import cupy
    except Exception:
        return 0
    try:
        return int(cupy.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def pytest_collection_modifyitems(config, items):
    if _cuda_device_count() > 0:
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA-capable device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
