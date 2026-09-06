"""Filament inductance kernels and section matrix assembly."""

import math

import numpy as np
import pytest
from scipy.constants import mu_0
from scipy.integrate import dblquad, quad
from scipy.special import ellipe, ellipk

from thirdlight.em.inductance import (
    inductance_matrix,
    mutual_matrix,
    mutual_ring,
    reduce_sections,
    section_inductance_matrix,
    self_ring,
    solenoid_inductance,
    strip_radius,
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


def segment_log_gmd(w):
    """(1/w^2) int_0^w int_0^w ln|x - y| dx dy, the log self GMD of a segment."""

    def inner(x):
        return quad(
            lambda y: math.log(abs(x - y)),
            0.0,
            w,
            points=[x],
            limit=200,
            epsabs=0.0,
            epsrel=1e-13,
        )[0]

    return quad(inner, 0.0, w, limit=200, epsabs=0.0, epsrel=1e-13)[0] / (w * w)


def rectangle_gmd(w, t):
    """Self GMD of a w x t rectangle, straight from ln g = <ln|r - r'|>.

    The quadruple integral collapses onto the difference vector (u, v), whose
    density is the triangular autocorrelation (w - |u|)(t - |v|) of the section;
    both integrands are even, so a quadrant carries the whole.
    """
    integral = dblquad(
        lambda v, u: (w - u) * (t - v) * math.log(u * u + v * v),
        0.0,
        w,
        0.0,
        t,
        epsabs=1e-12,
        epsrel=1e-12,
    )[0]
    return math.exp(2.0 * integral / (w * w * t * t))


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


@pytest.mark.parametrize("w", [1.0, 0.02, 250.0])
def test_thin_strip_self_gmd_is_exact(w):
    """A segment's self GMD is w exp(-3/2), which is strip_radius at zero thickness."""
    assert segment_log_gmd(w) == pytest.approx(math.log(w) - 1.5, abs=1e-12)
    assert strip_radius(w) == pytest.approx(w * math.exp(-1.5), rel=1e-15)
    assert strip_radius(w, 0.0) == strip_radius(w)


@pytest.mark.parametrize("ratio", [1.0, 0.5, 0.1])
def test_rectangle_gmd_matches_rosa_to_a_quarter_percent(ratio):
    """exp(-3/2)(w + t) sits 0.18 %, 0.21 % and 0.21 % low at t/w = 1, 1/2, 1/10.

    The approximation is one-signed and worst near t/w = 0.22, where it is 0.25 %
    low; it is exact as t/w -> 0. A quarter percent on rw is parts in 10^4 on a
    loop inductance, which reads rw through a logarithm.
    """
    w, t = 0.08, 0.08 * ratio
    error = strip_radius(w, t) / rectangle_gmd(w, t) - 1.0
    assert -2.5e-3 < error < 0.0


def test_a_wider_band_is_a_lower_inductance_loop():
    """Loop inductance falls as ln(1/rw); a band of matched GMD is its round wire."""
    loop = [
        inductance_matrix(
            Primary(
                inner_radius=0.15, turns=1.0, band_width=w, band_thickness=0.002
            ).discretise()
        ).sum()
        for w in (0.01, 0.02, 0.05, 0.1, 0.2)
    ]
    assert np.all(np.diff(loop) < 0.0)
    diameter = 0.0064
    band = Primary(
        inner_radius=0.15, turns=1.0, band_width=0.5 * diameter * math.exp(1.5)
    )
    round_wire = Primary(inner_radius=0.15, turns=1.0, wire_diameter=diameter)
    assert band.discretise().rw == pytest.approx(0.5 * diameter, rel=1e-14)
    assert inductance_matrix(band.discretise()).sum() == pytest.approx(
        inductance_matrix(round_wire.discretise()).sum(), rel=1e-14
    )


def test_strip_radius_rejects_impossible_sections():
    with pytest.raises(ValueError, match="width must be positive"):
        strip_radius(0.0)
    with pytest.raises(ValueError, match="thickness must be non-negative"):
        strip_radius(0.05, -1e-3)
