"""Test configuration shared by the suite."""

import os

import pytest

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
