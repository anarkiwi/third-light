"""Ring self and mutual inductance, and the section inductance matrix.

Coils are filaments at turn resolution: ring ``i`` carries ``n_i`` turns at
radius ``a_i`` and height ``z_i``. Series-connected turns share a current, so a
sectioned ladder matrix is the block sum of the turn-level matrix; no Nagaoka or
Rosa correction enters anywhere.
"""

import math

import numba
import numpy as np
from scipy.constants import mu_0

from thirdlight.backend import kernel
from thirdlight.em.elliptic import ellipke

_SERIES_M = 0.2
_SERIES_TERMS = 40
_SERIES_TOL = 1e-18


@kernel
def _bracket(m):
    """(2/k - k) K(m) - (2/k) E(m) with k = sqrt(m), for 0 < m < 1.

    The two leading terms of that difference cancel as m -> 0, losing 8/m in
    relative precision. Below ``_SERIES_M`` the identity
    (K - E) - m K / 2 = (pi/2) sum_{n>=2} c_{n-1} (n-1) m^n / (2n),
    c_n = c_{n-1} ((2n-1)/(2n))^2, gives the same quantity as a sum of positive
    terms, so no cancellation occurs at any m.
    """
    if m < _SERIES_M:
        c = 0.25
        p = m * m
        s = 0.0
        for n in range(2, _SERIES_TERMS):
            t = c * (n - 1) * p / (2 * n)
            s += t
            if t < _SERIES_TOL * s:
                break
            r = (2 * n - 1) / (2 * n)
            c *= r * r
            p *= m
        return math.pi * s / math.sqrt(m)
    k = math.sqrt(m)
    ek, ee = ellipke(m)
    return (2.0 / k - k) * ek - (2.0 / k) * ee


@kernel
def mutual_ring(a1, a2, dz):
    """Maxwell mutual inductance of two coaxial filament rings, axial gap ``dz``."""
    m = 4.0 * a1 * a2 / ((a1 + a2) * (a1 + a2) + dz * dz)
    return mu_0 * math.sqrt(a1 * a2) * _bracket(m)


@kernel
def self_ring(a, rw):
    """Self inductance of a round-wire loop carrying a surface (skin) current."""
    return mu_0 * a * (math.log(8.0 * a / rw) - 2.0)


@numba.njit(cache=True, parallel=True)
def _assemble(f):
    """Symmetric turn-level inductance matrix from stacked (a, z, n, rw) rows."""
    a, z, n, rw = f[0], f[1], f[2], f[3]
    count = a.shape[0]
    out = np.empty((count, count))
    for i in numba.prange(count):  # pylint: disable=not-an-iterable
        out[i, i] = n[i] * n[i] * self_ring(a[i], rw[i])
        for j in range(i + 1, count):
            v = n[i] * n[j] * mutual_ring(a[i], a[j], z[i] - z[j])
            out[i, j] = v
            out[j, i] = v
    return out


@numba.njit(cache=True, parallel=True)
def _cross(f, g):
    """Mutual inductance between two disjoint ring sets, shape (Nf, Ng)."""
    out = np.empty((f.shape[1], g.shape[1]))
    for i in numba.prange(f.shape[1]):  # pylint: disable=not-an-iterable
        for j in range(g.shape[1]):
            out[i, j] = (
                f[2, i] * g[2, j] * mutual_ring(f[0, i], g[0, j], f[1, i] - g[1, j])
            )
    return out


def _fields(rings):
    """Stack a ring set as contiguous float64 (a, z, n, rw) rows."""
    return np.ascontiguousarray(
        np.stack([rings.a, rings.z, rings.n, rings.rw]), dtype=np.float64
    )


def inductance_matrix(rings):
    """Dense symmetric inductance matrix of a :class:`~thirdlight.geometry.Rings` set.

    Diagonal terms are n_i^2 times the loop self inductance, exact at one turn
    per ring; off-diagonal terms are n_i n_j times the ring mutual inductance.
    """
    return _assemble(_fields(rings))


def mutual_matrix(rings_a, rings_b):
    """Cross-inductance matrix between two disjoint ring sets, shape (Na, Nb)."""
    return _cross(_fields(rings_a), _fields(rings_b))


def reduce_sections(matrix, groups):
    """Block-sum ``matrix`` over contiguous ascending ``groups`` labels."""
    groups = np.asarray(groups)
    if groups.shape != matrix.shape[:1]:
        raise ValueError(f"groups length {groups.shape} does not index {matrix.shape}")
    starts = np.flatnonzero(np.concatenate(([True], np.diff(groups) != 0)))
    if not np.array_equal(groups[starts], np.arange(starts.size)):
        raise ValueError("groups must label contiguous ascending blocks from 0")
    return np.add.reduceat(np.add.reduceat(matrix, starts, axis=0), starts, axis=1)


def turn_groups(turns, sections):
    """Section index of each turn for ``sections`` equal-turn-count sections."""
    if not 0 < sections <= turns:
        raise ValueError(f"sections must be in 1..{turns}, got {sections}")
    return (np.arange(turns) * sections) // turns


def solenoid_inductance(coil):
    """Total inductance of a solenoid at one-ring-per-turn resolution."""
    return float(inductance_matrix(coil.discretise()).sum())


def section_inductance_matrix(coil, sections):
    """Ladder inductance matrix of a solenoid, reduced from turn resolution."""
    rings = coil.discretise()
    return reduce_sections(inductance_matrix(rings), turn_groups(len(rings), sections))
