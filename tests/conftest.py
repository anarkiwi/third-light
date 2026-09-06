"""Test configuration shared by the suite."""

import os

import pytest

# Set before any worker subprocess starts, so each xdist worker inherits a single
# thread. Otherwise every worker opens a full-width prange and BLAS pool on the
# same cores: 66 s wall and 23 min CPU for this suite, against 3.6 s and 33 s.
for _var in (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

JIT_DISABLED = os.environ.get("NUMBA_DISABLE_JIT") == "1"


def _cuda_available():
    """Whether a CUDA device is present, without importing CuPy."""
    try:
        from numba import cuda  # pylint: disable=import-outside-toplevel

        return cuda.is_available()
    except Exception:  # pylint: disable=broad-except
        return False


def pytest_collection_modifyitems(config, items):
    """Skip what this host cannot run: device tests without a device, slow tests interpreted.

    The ``slow`` code paths are covered by the small-size tests, so the coverage
    pass loses nothing by dropping them; ``cuda`` needs hardware.
    """
    del config
    if not _cuda_available():
        needs_device = pytest.mark.skip(reason="no CUDA device")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(needs_device)
    if not JIT_DISABLED:
        return
    skip = pytest.mark.skip(reason="NUMBA_DISABLE_JIT=1: too slow interpreted")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
