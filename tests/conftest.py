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


def pytest_collection_modifyitems(config, items):
    """Skip large-size tests when Numba is interpreted, which is ~10^3x slower.

    Their code paths are covered by the small-size tests, so the coverage pass
    loses nothing by dropping them.
    """
    del config
    if not JIT_DISABLED:
        return
    skip = pytest.mark.skip(reason="NUMBA_DISABLE_JIT=1: too slow interpreted")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
