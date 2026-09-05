"""Medhurst's Table VIII and the eddy-reaction factor interpolated from it."""

import math

import numpy as np
import pytest

from thirdlight.em import medhurst
from thirdlight.em.medhurst import (
    LENGTH,
    PHI,
    SPACING,
    eddy_reaction,
    proximity_factor,
)

FINITE = LENGTH[:-1]


def test_the_table_is_reproduced_at_every_node():
    """Interpolation is exact at the tabulated d/s and l/D."""
    grid = np.array([[proximity_factor(s, v) for v in LENGTH] for s in SPACING])
    assert grid == pytest.approx(PHI, abs=1e-12)


def test_the_reaction_is_the_measured_share_of_the_uniform_field_excess():
    """phi - 1 over pi^2 (d/s)^2 / 2, the model's own high-frequency excess."""
    excess = 0.5 * math.pi**2 * np.square(SPACING)[:, None]
    grid = np.array([[eddy_reaction(s, v) for v in LENGTH] for s in SPACING])
    assert grid == pytest.approx((PHI - 1.0) / excess, rel=1e-12)
    assert np.all(grid <= 1.02)


def test_the_reaction_falls_as_the_turns_close_up():
    """At l/D = infinity the neighbours' eddy field halves the excess by d/s = 1."""
    tail = np.array([eddy_reaction(s, math.inf) for s in SPACING])
    assert tail[0] == pytest.approx(1.013, abs=0.005)
    assert tail[-1] == pytest.approx(0.489, abs=0.005)
    assert np.all(np.diff(tail) < 0.0)


def test_phi_rises_with_d_over_s_at_every_length():
    """More copper per unit pitch is more proximity loss, at any coil shape."""
    fine = np.linspace(SPACING[0], SPACING[-1], 91)
    for ratio in FINITE:
        assert np.all(np.diff(proximity_factor(fine, ratio)) > 0.0)


def test_interpolation_stays_inside_the_bracketing_cells():
    """pchip is shape preserving, so no interpolated value overshoots its neighbours."""
    for i in range(len(SPACING) - 1):
        for j in range(len(FINITE) - 1):
            cell = PHI[i : i + 2, j : j + 2]
            d = np.linspace(SPACING[i], SPACING[i + 1], 7)[:, None]
            ratio = np.linspace(FINITE[j], FINITE[j + 1], 7)[None, :]
            inner = proximity_factor(np.broadcast_to(d, (7, 7)), ratio)
            assert inner.min() >= cell.min() - 1e-9
            assert inner.max() <= cell.max() + 1e-9


def test_the_axes_are_clamped_outside_the_table():
    """Sparser than d/s = 0.1 holds the edge, where measurement and theory agree to 1 %."""
    assert eddy_reaction(0.02, 4.0) == eddy_reaction(SPACING[0], 4.0)
    assert proximity_factor(0.02, 4.0) < proximity_factor(0.1, 4.0)
    assert eddy_reaction(0.5, 1e9) == pytest.approx(
        eddy_reaction(0.5, math.inf), rel=1e-6
    )


def test_the_length_axis_is_compactified_at_infinity():
    """x/(1 + x) carries the table's last column to 1 without an infinity in the grid."""
    assert medhurst._compact(math.inf) == 1.0  # pylint: disable=protected-access
    assert medhurst._compact(1.0) == 0.5  # pylint: disable=protected-access


def test_calls_broadcast():
    """Array arguments give the same values as the scalar calls they broadcast to."""
    d = np.array([[0.2], [0.7]])
    ratio = np.array([1.0, 6.0, math.inf])
    grid = proximity_factor(d, ratio)
    assert grid.shape == (2, 3)
    assert grid == pytest.approx(
        np.array([[proximity_factor(s, v) for v in ratio] for s in d.ravel()])
    )
