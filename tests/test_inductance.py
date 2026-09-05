"""Filament inductance kernels and section matrix assembly."""

import math

import numpy as np
import pytest
from scipy.constants import mu_0
from scipy.special import ellipe, ellipk

from thirdlight.em.inductance import (
    inductance_matrix,
    mutual_matrix,
    mutual_ring,
    reduce_sections,
    section_inductance_matrix,
    self_ring,
    solenoid_inductance,
    turn_groups,
)
from thirdlight.geometry import Primary, Solenoid


def parameter(a1, a2, dz):
    return 4.0 * a1 * a2 / ((a1 + a2) * (a1 + a2) + dz * dz)


def maxwell(a1, a2, dz):
    """Maxwell's coaxial-ring mutual inductance, evaluated with scipy."""
    m = parameter(a1, a2, dz)
    k = math.sqrt(m)
    return (
        mu_0 * math.sqrt(a1 * a2) * ((2.0 / k - k) * ellipk(m) - (2.0 / k) * ellipe(m))
    )


def nagaoka(a, length):
    """Lorenz current-sheet coefficient; -> 1 for a long coil, exact in both limits."""
    m = 4.0 * a * a / (4.0 * a * a + length * length)
    kp = math.sqrt(1.0 - m)
    return (
        4.0
        / (3.0 * math.pi * kp)
        * ((kp * kp / m) * (ellipk(m) - ellipe(m)) + ellipe(m) - math.sqrt(m))
    )


def wheeler(coil):
    """Wheeler 1928, Proc. IRE 16(10) 1398: L[uH] = r^2 N^2 / (9r + 10l), inches."""
    r, length = coil.radius / 0.0254, coil.length / 0.0254
    return 1e-6 * r * r * coil.turns**2 / (9.0 * r + 10.0 * length)


def sheet_inductance(coil):
    """Current-sheet (Nagaoka) inductance of the same winding."""
    return (
        mu_0
        * math.pi
        * coil.radius**2
        * coil.turns**2
        / coil.length
        * nagaoka(coil.radius, coil.length)
    )


SECONDARY = Solenoid(
    radius=0.076, length=0.5, turns=1000, wire_diameter=4e-4, base=0.05
)


@pytest.mark.parametrize(
    "a1,a2,dz",
    [
        (1.0, 1.0, 1e-6),
        (1.0, 1.0, 1e-3),
        (1.0, 1.0, 0.1),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 5.0),
        (0.05, 0.05, 1e-4),
        (1.0, 0.1, 0.0),
        (1.0, 0.1, 0.5),
        (0.1, 1.0, 1.0),
        (0.3, 2.5, 1.0),
    ],
)
def test_mutual_ring_matches_maxwell(a1, a2, dz):
    # scipy's own form cancels catastrophically below m ~ 1e-2, so the cases stay above it.
    assert parameter(a1, a2, dz) > 1e-2
    assert mutual_ring(a1, a2, dz) == pytest.approx(maxwell(a1, a2, dz), rel=1e-12)


@pytest.mark.parametrize("ratio", [1e2, 1e3, 1e4, 1e5])
def test_mutual_ring_far_field_is_dipole(ratio):
    a1, a2, d = 1.0, 0.5, ratio
    dipole = mu_0 * math.pi * a1**2 * a2**2 / (2.0 * d**3)
    # M/M_dipole = 1 + 3m/4 + O(m^2), so the residual must fall below m itself.
    assert abs(mutual_ring(a1, a2, d) / dipole - 1.0) < parameter(a1, a2, d)


def test_self_ring_closed_form_and_thin_gap_limit():
    a, rw = 0.1, 1e-3
    assert self_ring(a, rw) == pytest.approx(mu_0 * a * (math.log(8.0 * a / rw) - 2.0))
    # Two rings a hair apart have the self inductance of one loop of that wire radius.
    assert mutual_ring(a, a, 1e-4 * a) == pytest.approx(
        self_ring(a, 1e-4 * a), rel=1e-7
    )


@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0, 5.0])
def test_solenoid_inductance_against_wheeler_and_nagaoka(ratio):
    coil = Solenoid(
        radius=0.05, length=ratio * 0.1, turns=800, wire_diameter=ratio * 0.1 / 800
    )
    total = solenoid_inductance(coil)
    assert total == pytest.approx(wheeler(coil), rel=0.01)
    assert total == pytest.approx(sheet_inductance(coil), rel=0.01)


@pytest.fixture(name="secondary_total", scope="module")
def fixture_secondary_total():
    return solenoid_inductance(SECONDARY)


@pytest.mark.parametrize("sections", [1, 7, 30, 120, 500])
def test_section_reduction_conserves_total(sections, secondary_total):
    matrix = section_inductance_matrix(SECONDARY, sections)
    assert matrix.shape == (sections, sections)
    assert matrix.sum() == pytest.approx(secondary_total, rel=1e-12)


def test_reduce_sections_block_sums():
    matrix = np.arange(16.0).reshape(4, 4)
    reduced = reduce_sections(matrix, np.array([0, 0, 1, 1]))
    assert np.array_equal(reduced, [[10, 18], [42, 50]])


def test_reduce_sections_rejects_bad_groups():
    matrix = np.zeros((4, 4))
    with pytest.raises(ValueError, match="does not index"):
        reduce_sections(matrix, np.zeros(3, dtype=int))
    with pytest.raises(ValueError, match="contiguous"):
        reduce_sections(matrix, np.array([0, 1, 0, 1]))
    with pytest.raises(ValueError, match="sections must be"):
        turn_groups(4, 5)


def test_inductance_matrix_symmetric_positive_definite():
    rings = Solenoid(0.05, 0.4, 400, 8e-4).discretise()
    matrix = inductance_matrix(rings)
    assert np.array_equal(matrix, matrix.T)
    assert np.linalg.eigvalsh(matrix).min() > 0.0
    assert np.diag(matrix) == pytest.approx(self_ring(rings.a[0], rings.rw[0]))


def test_mutual_matrix_is_reciprocal():
    primary = Primary(inner_radius=0.115, turns=5.5, pitch=0.012).discretise()
    secondary = SECONDARY.discretise(60)
    forward = mutual_matrix(primary, secondary)
    assert forward.shape == (len(primary), len(secondary))
    assert np.array_equal(forward, mutual_matrix(secondary, primary).T)


def test_primary_coupling_rises_as_the_primary_approaches_the_base(secondary_total):
    secondary = SECONDARY.discretise()
    ls = secondary_total
    coupling = []
    for base in (-0.15, -0.05, 0.0, 0.03):
        primary = Primary(0.115, 5.5, pitch=0.012, base=base).discretise()
        lp = inductance_matrix(primary).sum()
        coupling.append(mutual_matrix(primary, secondary).sum() / math.sqrt(lp * ls))
    assert all(0.0 < k < 1.0 for k in coupling)
    assert np.all(np.diff(coupling) > 0.0)
