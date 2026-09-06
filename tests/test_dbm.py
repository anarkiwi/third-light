"""Dielectric breakdown growth: the dimensions it produces and the incremental factor."""

import math

import numpy as np
import pytest
from scipy.linalg import cholesky

from thirdlight.discharge.dbm import (
    BorderedCholesky,
    Discharge,
    Growth,
    cap_directions,
    fractal_dimension,
    grow,
    gyration,
    interior,
)
from thirdlight.discharge.filament import Tree, channel_load, mixed_potential_matrix
from thirdlight.em.capacitance import potential_matrix
from thirdlight.geometry import Sphere, Toroid

STEP = 0.05
CHANNEL = 1.0e-3
RESISTIVITY = 1.0e4
PLANAR, SPATIAL = 13, 16

FREE = Sphere(radius=0.05, height=2.0)
FREE_RINGS = FREE.discretise(12)
FREE_SEED = (0.0, 0.0, 2.05)
UP = (0.0, 0.0, 1.0)

POINT = Sphere(radius=0.01, height=2.0)
POINT_RINGS = POINT.discretise(12)
POINT_SEED = (0.0, 0.0, 2.01)

TOP_LOAD = Toroid(major_radius=0.15, minor_radius=0.05, height=0.62)
LOAD_RINGS = TOP_LOAD.discretise(16)
LOAD_SEED = (0.2, 0.0, 0.62)
OUT = (1.0, 0.0, 0.0)

# Mean and seed-to-seed spread measured over six seeds at CLUSTER segments.
CLUSTER = 500
PLANAR_ETA_ONE = (2.264, 0.170)
PLANAR_EDEN = (2.240, 0.195)


def free_tree(eta, seed, count=CLUSTER, plane=True, **kw):
    """One cluster grown into free space off a small sphere, the radial DBM geometry."""
    growth = Growth(
        step=STEP,
        radius=CHANNEL,
        eta=eta,
        directions=PLANAR if plane else SPATIAL,
        plane=plane,
        **kw,
    )
    return grow(
        FREE_RINGS,
        FREE_SEED,
        UP,
        growth,
        count,
        np.random.default_rng(seed),
        bodies=(FREE,),
        ground=False,
    )


def loaded_tree(count, seed=0, **kw):
    """One cluster grown off the example top load over a ground plane."""
    growth = Growth(step=STEP, radius=CHANNEL, directions=SPATIAL, **kw)
    return grow(
        LOAD_RINGS,
        LOAD_SEED,
        OUT,
        growth,
        count,
        np.random.default_rng(seed),
        bodies=(TOP_LOAD,),
    )


def children(tree):
    """Number of children of every node."""
    return np.bincount(tree.parent[1:], minlength=len(tree))


def test_cap_directions_span_the_cone():
    for count, plane in ((PLANAR, True), (SPATIAL, False)):
        cone = cap_directions(count, 0.4, plane)
        assert np.allclose(np.linalg.norm(cone, axis=1), 1.0)
        assert np.all(cone[:, 2] >= math.cos(0.4) - 1e-12)
        assert np.isclose(cone[:, 2].max(), 1.0)
        assert np.isclose(cone[:, 2].min(), math.cos(0.4))
        assert np.all(cone[:, 1] == 0.0) == plane


def test_interior_of_the_top_load_shapes():
    points = np.array([[0.0, 0.0, 2.0], [0.15, 0.0, 0.62], [0.0, 0.0, 0.0]])
    assert list(interior((FREE,), points)) == [True, False, False]
    assert list(interior((TOP_LOAD,), points)) == [False, True, False]
    with pytest.raises(TypeError):
        interior((object(),), points)


def test_gyration_matches_the_direct_moment():
    nodes = np.random.default_rng(0).normal(size=(20, 3))
    direct = [
        np.sqrt(np.mean(np.sum((nodes[:k] - nodes[:k].mean(0)) ** 2, axis=1)))
        for k in range(1, len(nodes) + 1)
    ]
    assert gyration(nodes) == pytest.approx(direct)


def test_bordered_factor_matches_a_dense_cholesky():
    rng = np.random.default_rng(1)
    root = rng.normal(size=(9, 9))
    matrix = root @ root.T + 9.0 * np.eye(9)
    factor = BorderedCholesky(matrix[:4, :4], 9)
    for size in range(4, 9):
        assert factor.append(matrix[:size, size], matrix[size, size])
        assert factor.factor == pytest.approx(
            cholesky(matrix[: size + 1, : size + 1], lower=True), rel=1e-12
        )
    assert len(factor) == 9
    assert factor.solve(np.ones(9)) == pytest.approx(
        np.linalg.solve(matrix, np.ones(9))
    )
    assert not BorderedCholesky(matrix[:4, :4], 5).append(matrix[:4, 4], -1.0)


def test_incremental_factor_matches_the_assembled_matrix():
    """The optimisation is exact: the bordered factor is the assembly's own, not near it."""
    discharge = Discharge(
        LOAD_RINGS,
        LOAD_SEED,
        OUT,
        Growth(step=STEP, radius=CHANNEL, directions=SPATIAL),
        40,
        np.random.default_rng(2),
        bodies=(TOP_LOAD,),
    )
    for step in range(40):
        assert discharge.step()
        if step in (0, 1, 7, 20, 39):
            dense = cholesky(
                mixed_potential_matrix(LOAD_RINGS, discharge.tree), lower=True
            )
            assert discharge.factor == pytest.approx(dense, rel=1e-10)
    assert discharge.factor.shape == (len(LOAD_RINGS) + 40,) * 2


def test_the_initial_factor_is_the_ring_matrix_alone():
    discharge = Discharge(
        LOAD_RINGS,
        LOAD_SEED,
        OUT,
        Growth(step=STEP, radius=CHANNEL),
        4,
        np.random.default_rng(0),
        bodies=(TOP_LOAD,),
    )
    assert discharge.factor == pytest.approx(
        cholesky(potential_matrix(LOAD_RINGS), lower=True), rel=1e-12
    )
    assert len(discharge) == 1
    assert discharge.sites.shape[1] == 3


def test_grown_tree_is_structurally_valid():
    tree = loaded_tree(40)
    assert len(tree) == 41
    Tree(tree.nodes, tree.parent, tree.radius)
    assert np.all(tree.parent[1:] < np.arange(1, len(tree)))
    assert np.all(tree.nodes[:, 2] >= 0.0)
    assert not np.any(interior((TOP_LOAD,), tree.nodes))
    assert tree.lengths == pytest.approx(STEP, rel=1e-15)


def test_growth_is_reproducible_and_seed_dependent():
    first, again = loaded_tree(25), loaded_tree(25)
    assert np.array_equal(first.nodes, again.nodes)
    assert np.array_equal(first.parent, again.parent)
    other = loaded_tree(25, seed=1)
    assert not np.array_equal(first.nodes, other.nodes)


@pytest.mark.parametrize("critical, segments", [(0.0, 60), (16.0, 7), (18.0, 0)])
def test_critical_field_terminates_growth(critical, segments):
    """A channel off a sharp point outruns the field that started it and stops.

    The head field of a channel at fixed potential falls only as it leaves the
    electrode, so the critical field is what a discharge must be initiated over
    and what ends it; at zero it runs to the step cap.
    """
    growth = Growth(step=STEP, radius=CHANNEL, critical=critical, directions=SPATIAL)
    tree = grow(
        POINT_RINGS,
        POINT_SEED,
        UP,
        growth,
        60,
        np.random.default_rng(0),
        bodies=(POINT,),
        ground=False,
    )
    assert tree.segments == segments


def test_growth_loads_the_electrode():
    tree = loaded_tree(45)
    loads = [
        channel_load(
            LOAD_RINGS, Tree(tree.nodes[:k], tree.parent[:k], tree.radius), RESISTIVITY
        )
        for k in (10, 25, 46)
    ]
    added = [load[0] for load in loads]
    assert added[0] < added[1] < added[2]
    assert all(load[1] > 0.0 for load in loads)


@pytest.mark.slow
def test_planar_dimension_at_eta_one():
    """R_g ~ N^(1/D) at eta = 1 in a plane, against the band six seeds actually span.

    A planar cluster in a three-dimensional kernel is screened by field lines
    that leave the plane, so it is not the two-dimensional universality class
    and [18]'s 1.75 is nowhere in this band; see docs/design.md 3.4c.
    """
    mean, spread = PLANAR_ETA_ONE
    seeds = [fractal_dimension(free_tree(1.0, seed).nodes) for seed in range(5)]
    assert np.mean(seeds) == pytest.approx(mean, abs=4.0 * spread / math.sqrt(5))
    assert np.mean(seeds) > 1.75 + spread


@pytest.mark.slow
def test_dimension_falls_as_eta_rises():
    """The one ordering [18] pins that a single noisy value does not."""
    means = [
        np.mean(
            [
                fractal_dimension(free_tree(eta, s, 300, plane=False).nodes)
                for s in range(4)
            ]
        )
        for eta in (0.0, 3.0, 6.0)
    ]
    assert means[0] > means[1] > means[2]


@pytest.mark.slow
def test_eden_limit_is_the_embedding_dimension():
    """At eta = 0 every admissible candidate is equally likely, so growth is compact."""
    mean, spread = PLANAR_EDEN
    trees = [free_tree(0.0, seed) for seed in range(3)]
    dimensions = [fractal_dimension(tree.nodes) for tree in trees]
    assert np.mean(dimensions) == pytest.approx(mean, abs=4.0 * spread / math.sqrt(3))
    assert np.mean(dimensions) == pytest.approx(2.0, abs=3.0 * spread)
    assert all((children(tree) > 1).mean() > 0.2 for tree in trees)


@pytest.mark.slow
def test_large_eta_collapses_to_a_needle():
    """Deterministic selection of the strongest field leaves one channel and D -> 1."""
    for seed in range(2):
        tree = free_tree(30.0, seed, 200)
        forks = children(tree)
        assert forks.max() == 2
        assert (forks > 1).sum() < 0.1 * tree.segments
        assert fractal_dimension(tree.nodes) == pytest.approx(1.0, abs=0.25)
