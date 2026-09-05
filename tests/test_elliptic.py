"""AGM elliptic integrals against scipy.special."""

import numpy as np
import pytest
from scipy import special

from thirdlight.em.elliptic import ellipe, ellipk, ellipke


@pytest.mark.parametrize("m", [0.0, 1e-12, 0.25, 0.5, 0.9, 1 - 1e-6, 1 - 1e-12])
def test_matches_scipy(m):
    k, e = ellipke(m)
    assert k == pytest.approx(special.ellipk(m), rel=1e-14)
    assert e == pytest.approx(special.ellipe(m), rel=1e-14)
    assert ellipk(m) == k
    assert ellipe(m) == e


def test_sweep_relative_error():
    m = np.linspace(0.0, 1 - 1e-9, 4001)
    got = np.array([ellipke(float(v)) for v in m])
    ref = np.stack([special.ellipk(m), special.ellipe(m)], axis=1)
    assert np.abs(got / ref - 1.0).max() < 1e-14


def test_legendre_relation():
    """K(m)E(1-m) + K(1-m)E(m) - K(m)K(1-m) = pi/2."""
    for m in (0.1, 0.37, 0.8):
        k1, e1 = ellipke(m)
        k2, e2 = ellipke(1.0 - m)
        assert k1 * e2 + k2 * e1 - k1 * k2 == pytest.approx(0.5 * np.pi, rel=1e-13)
