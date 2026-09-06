"""Thin-wire filament potential coefficients, mixed ring/segment assembly and the tree reduction."""

import math

import numpy as np
import pytest
from scipy.constants import epsilon_0
from scipy.integrate import quad

from thirdlight.discharge.filament import (
    Tree,
    channel_load,
    filament_potential,
    mixed_charges,
    mixed_potential_matrix,
    potential_segment,
    segment_potential_matrix,
    self_potential_segment,
    series_resistance,
    subtree_charges,
)
from thirdlight.em.capacitance import lumped_capacitance
from thirdlight.geometry import Toroid

FAR = 1000.0
RESISTIVITY = 1.0e4
TOP_LOAD = Toroid(major_radius=0.15, minor_radius=0.05, height=0.62)


def segment_quadrature(p, a, b):
    """Direct quadrature of 1/(4 pi eps0 |x - x'|) over a unit-charge straight segment."""
    p, a, b = (np.asarray(v, dtype=float) for v in (p, a, b))

    def integrand(t):
        return 1.0 / np.linalg.norm(p - (a + t * (b - a)))

    value = quad(integrand, 0.0, 1.0, epsabs=0.0, epsrel=1e-13, limit=200)[0]
    return value / (4.0 * math.pi * epsilon_0)


def chain(count, length, radius, base=0.0, axis=2):
    """Straight channel of ``count`` equal segments, one node per joint."""
    nodes = np.zeros((count + 1, 3))
    nodes[:, axis] = base + length * np.arange(count + 1) / count
    return Tree(nodes, np.arange(-1, count), radius)


def polygon(count, radius, wire):
    """Closed regular polygon of ``count`` chords inscribed in a circle, as segment endpoints."""
    theta = 2.0 * math.pi * np.arange(count) / count
    points = np.stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros(count)])
    return points.T, np.roll(points.T, -1, axis=0), wire


def isolated_capacitance(start, end, wire):
    """Total charge on a segment set held at unit potential, with no ground plane."""
    matrix = segment_potential_matrix(start, end, wire, ground_plane=False)
    return np.linalg.solve(matrix, np.ones(matrix.shape[0])).sum()


@pytest.mark.parametrize(
    "p",
    [
        (0.3, 0.0, 0.0),
        (0.0, 0.0, 2.0),
        (1.5, -0.7, 0.25),
        (1e-3, 1e-3, 0.5),
        (-4.0, 3.0, -2.0),
        (0.0, 0.2, 0.5),
    ],
)
def test_potential_segment_matches_quadrature(p):
    a, b = (0.0, 0.0, 0.0), (0.1, -0.2, 1.0)
    assert potential_segment(*p, *a, *b) == pytest.approx(
        segment_quadrature(p, a, b), rel=1e-10
    )
    assert potential_segment(*p, *b, *a) == pytest.approx(
        potential_segment(*p, *a, *b), rel=1e-15
    )


def test_distant_segment_converges_to_a_point_charge():
    a, b = (0.0, 0.0, -0.5), (0.0, 0.0, 0.5)
    error = []
    for distance in (10.0, 100.0, 1000.0):
        point = 1.0 / (4.0 * math.pi * epsilon_0 * distance)
        error.append(abs(potential_segment(distance, 0.0, 0.0, *a, *b) / point - 1.0))
    assert np.all(np.diff(error) < 0.0)
    assert np.all(np.array(error[1:]) < 0.02 * np.array(error[:-1]))
    assert error[-1] < 1e-6


@pytest.mark.parametrize("ratio", [1e-3, 1e-4, 1e-5])
def test_self_term_is_the_thin_wire_logarithm(ratio):
    """The exact diagonal tends to 2 ln(L / rw), with a relative error of order (rw/L)^2."""
    L = 0.4
    value = self_potential_segment(L, ratio * L) * L * 4.0 * math.pi * epsilon_0
    assert value == pytest.approx(2.0 * math.log(1.0 / ratio), rel=ratio**2)


def test_mixed_matrix_is_symmetric_and_positive_definite():
    rings = TOP_LOAD.discretise(32)
    tree = chain(12, 0.8, 5e-3, base=0.67)
    matrix = mixed_potential_matrix(rings, tree)
    assert matrix.shape == (len(rings) + tree.segments,) * 2
    assert np.abs(matrix - matrix.T).max() <= 1e-12 * np.abs(matrix).max()
    np.linalg.cholesky(matrix)


@pytest.mark.parametrize("p", [(0.4, 0.0, 0.0), (0.1, -0.9, 0.0), (3.0, 2.0, 0.0)])
def test_image_cancels_the_potential_on_the_ground_plane(p):
    a, b = (0.05, 0.02, 0.3), (-0.1, 0.2, 1.1)
    image = (a[0], a[1], -a[2], b[0], b[1], -b[2])
    direct = potential_segment(*p, *a, *b)
    assert abs(direct - potential_segment(*p, *image)) <= 1e-12 * abs(direct)


def test_polygon_converges_to_the_thin_torus_capacitance():
    """A closed N-gon of filaments is a torus in the limit; C = 4 pi^2 eps0 a / ln(8a/rw)."""
    a, ratio = 0.25, 1e-3
    exact = 4.0 * math.pi**2 * epsilon_0 * a / math.log(8.0 / ratio)
    error = np.array(
        [
            abs(isolated_capacitance(*polygon(count, a, ratio * a)) / exact - 1.0)
            for count in (8, 16, 32, 64, 128)
        ]
    )
    assert np.all(error[1:] < 0.4 * error[:-1])
    assert error[-1] < 5e-4


def test_straight_channel_against_the_prolate_spheroid():
    """Loose by design: the closed form is a spheroid, the model a uniform cylinder.

    The 10 % band is that shape difference, not a convergence tolerance.
    """
    length, wire = 2.0, 2e-3
    semi = 0.5 * length
    ecc = math.sqrt(1.0 - (wire / semi) ** 2)
    exact = 4.0 * math.pi * epsilon_0 * ecc * semi / math.atanh(ecc)
    tree = chain(64, length, wire, base=FAR)
    assert isolated_capacitance(*tree.endpoints, wire) == pytest.approx(exact, rel=0.1)


def test_straight_channel_capacitance_is_stable_under_refinement():
    length, wire = 2.0, 2e-3
    coarse, fine = (
        isolated_capacitance(*chain(count, length, wire, base=FAR).endpoints, wire)
        for count in (32, 64)
    )
    assert coarse == pytest.approx(fine, rel=1e-2)


def test_series_resistance_of_a_chain_with_all_charge_at_the_tip():
    """Every segment carries the whole current, so the equivalent is the plain sum."""
    tree = chain(7, 1.4, 3e-3)
    charges = np.zeros(tree.segments)
    charges[-1] = 4.2e-12
    element = RESISTIVITY * tree.lengths / (math.pi * tree.radius**2)
    assert series_resistance(tree, charges, RESISTIVITY) == pytest.approx(
        element.sum(), rel=1e-12
    )


def test_series_resistance_of_a_symmetric_bifurcation():
    """A trunk and two equal branches sharing the charge give R_t + R_b / 2."""
    nodes = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.6, 0.0, 1.8], [-0.6, 0.0, 1.8]]
    )
    tree = Tree(nodes, np.array([-1, 0, 1, 1]), 3e-3)
    element = RESISTIVITY * tree.lengths / (math.pi * tree.radius**2)
    charges = np.array([0.0, 1e-12, 1e-12])
    assert series_resistance(tree, charges, RESISTIVITY) == pytest.approx(
        element[0] + 0.5 * element[1], rel=1e-12
    )


def random_tree(count, seed):
    """Random tree with every parent preceding its child, and an exactly summable charge each."""
    rng = np.random.default_rng(seed)
    parent = np.concatenate(([-1], rng.integers(0, np.arange(1, count))))
    charges = rng.integers(1, 1000, count - 1).astype(float)
    return Tree(rng.normal(size=(count, 3)), parent, 2e-3), charges


def ancestor_sums(tree, charges):
    """Naive reference: add each segment's charge to itself and to every ancestor segment."""
    out = np.zeros(tree.segments)
    for k in range(1, len(tree)):
        node = k
        while node > 0:
            out[node - 1] += charges[k - 1]
            node = tree.parent[node]
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_subtree_charges_match_the_ancestor_walk(seed):
    tree, charges = random_tree(60, seed)
    assert np.array_equal(subtree_charges(tree, charges), ancestor_sums(tree, charges))


def test_channel_load_is_the_added_capacitance_and_a_positive_resistance():
    rings = TOP_LOAD.discretise(32)
    tree = chain(16, 1.0, 5e-3, base=0.67)
    capacitance, resistance = channel_load(rings, tree, RESISTIVITY)
    charges = mixed_charges(rings, tree)
    assert capacitance == pytest.approx(
        charges.sum() - lumped_capacitance(rings), rel=1e-12
    )
    assert capacitance > 0.0
    assert resistance == pytest.approx(
        series_resistance(tree, charges[len(rings) :], RESISTIVITY), rel=1e-12
    )
    longer = channel_load(rings, chain(32, 2.0, 5e-3, base=0.67), RESISTIVITY)
    assert longer[0] > capacitance
    assert longer[1] > resistance


def test_filament_potential_is_the_symmetric_closed_form():
    r1, r2, L = 1.5, 2.5, 1.0
    assert filament_potential(r1, r2, L) == pytest.approx(
        math.log(5.0 / 3.0) / (4.0 * math.pi * epsilon_0), rel=1e-15
    )


def test_tree_geometry_and_validation():
    tree = chain(4, 2.0, 1e-3)
    assert tree.segments == 4
    assert np.allclose(tree.lengths, 0.5)
    assert np.allclose(tree.midpoints[:, 2], [0.25, 0.75, 1.25, 1.75])
    start, end = tree.endpoints
    assert np.allclose(end - start, [[0.0, 0.0, 0.5]] * 4)
    with pytest.raises(ValueError, match=r"nodes must be"):
        Tree(np.zeros((3, 2)), np.array([-1, 0, 1]), 1e-3)
    with pytest.raises(ValueError, match="does not index"):
        Tree(np.zeros((3, 3)), np.array([-1, 0]), 1e-3)
    with pytest.raises(ValueError, match="precede"):
        Tree(np.zeros((3, 3)), np.array([0, 0, 1]), 1e-3)
    with pytest.raises(ValueError, match="precede"):
        Tree(np.zeros((3, 3)), np.array([-1, 2, 1]), 1e-3)
