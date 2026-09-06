"""Two coils side by side: electrode reduction, mutual capacitance and the split."""

import math
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest
from scipy.constants import epsilon_0

from thirdlight import pair
from thirdlight.geometry import Design, Primary, Solenoid, Sphere, Toroid
from thirdlight.secondary import resonance

C_SELF, L_SELF = 20.0e-12, 30.0e-3
SEPARATION = 0.8
SECONDARY = Solenoid(radius=0.04, length=0.2, turns=60, wire_diameter=3e-4, base=0.02)
PRIMARY = Primary(inner_radius=0.06, turns=4.0, pitch=0.01)
TOP_LOAD = Toroid(major_radius=0.06, minor_radius=0.02, height=0.24)
LEFT = Design(
    secondary=SECONDARY,
    primary=PRIMARY,
    top_load=TOP_LOAD,
    sections=30,
    top_load_sections=8,
)
RIGHT = replace(LEFT, secondary=replace(SECONDARY, turns=66))


@lru_cache(maxsize=None)
def modes_of(design):
    """Mode-1 resonance of a design, solved once per process."""
    return resonance(design, modes=1)


def built(separation=SEPARATION, a=LEFT, b=RIGHT):
    """A pair of the two module designs, reusing their cached resonances."""
    return pair.Pair(a, b, separation, modes_of(a), modes_of(b))


def sphere_pair(radius_a, radius_b, separation):
    """Mutual capacitance of two isolated spheres reduced to point charges."""
    c_a, c_b = (4.0 * math.pi * epsilon_0 * r for r in (radius_a, radius_b))
    p12 = 1.0 / (pair.FOUR_PI_EPS0 * separation)
    return pair.mutual_capacitance(pair.maxwell_capacitance(c_a, c_b, p12))


def two_level(detune, p12):
    """Split of two coils of equal c detuned in l alone, with their detune and coupling."""
    l_b = L_SELF / (1.0 + detune) ** 2
    matrix = pair.maxwell_capacitance(C_SELF, C_SELF, p12)
    split = pair.coupled_modes(L_SELF, l_b, matrix)
    f = np.array([1.0 / (2.0 * math.pi * math.sqrt(l * C_SELF)) for l in (L_SELF, l_b)])
    return split, float(abs(f[0] - f[1]) / f.mean()), pair.coupling(matrix)


def test_electrode_centroid_is_the_sphere_centre_without_a_ground_plane():
    design = Design(
        secondary=SECONDARY,
        primary=PRIMARY,
        top_load=Sphere(radius=0.05, height=0.5),
        ground_plane=False,
        top_load_sections=16,
    )
    assert pair.electrode_height(design) == pytest.approx(0.5, rel=1e-12)
    assert pair.electrode_radius(design) == 0.05


def test_the_ground_plane_image_pulls_the_electrode_centroid_down():
    free = Design(
        secondary=SECONDARY,
        primary=PRIMARY,
        top_load=Sphere(radius=0.05, height=0.5),
        ground_plane=False,
        top_load_sections=16,
    )
    grounded = replace(free, ground_plane=True)
    shift = pair.electrode_height(free) - pair.electrode_height(grounded)
    assert 0.0 < shift < 0.05


def test_electrode_reduction_falls_back_to_the_top_of_the_winding():
    design = Design(secondary=SECONDARY, primary=PRIMARY)
    assert pair.electrode_height(design) == SECONDARY.base + SECONDARY.length
    assert pair.electrode_radius(design) == SECONDARY.radius


@pytest.mark.parametrize("ratio", [10.0, 100.0, 1000.0])
def test_mutual_capacitance_approaches_the_two_sphere_far_field(ratio):
    radius_a, radius_b = 0.2, 0.3
    separation = ratio * radius_b
    mutual = sphere_pair(radius_a, radius_b, separation)
    reference = 4.0 * math.pi * epsilon_0 * radius_a * radius_b / separation
    # The reduction drops the induced dipoles: second order in a/s, and the
    # residual sits on that order to 1 % at the closest of these spacings.
    order = radius_a * radius_b / separation**2
    assert mutual / reference - 1.0 == pytest.approx(order, rel=1.0e-2)


def test_the_ground_plane_image_screens_the_coupling():
    free = pair.mutual_coefficient(1.0, 0.5, 0.6, ground_plane=False)
    grounded = pair.mutual_coefficient(1.0, 0.5, 0.6, ground_plane=True)
    assert 0.0 < grounded < free


@pytest.mark.parametrize("ratio", [10.0, 100.0, 1000.0])
def test_the_image_stops_screening_far_above_the_plane(ratio):
    separation, height = 1.0, ratio
    free = pair.mutual_coefficient(separation, height, height, ground_plane=False)
    grounded = pair.mutual_coefficient(separation, height, height, ground_plane=True)
    # The image sits 2h away against the direct term's s: first order in s / 2h.
    order = separation / (2.0 * height)
    assert 1.0 - grounded / free == pytest.approx(order, rel=order**2)


def test_identical_coils_split_in_closed_form():
    p12 = pair.mutual_coefficient(2.0, 1.5, 1.5)
    split = pair.coupled_modes(
        L_SELF, L_SELF, pair.maxwell_capacitance(C_SELF, C_SELF, p12)
    )
    f0 = 1.0 / (2.0 * math.pi * math.sqrt(L_SELF * C_SELF))
    assert split.f[1] == pytest.approx(f0 * math.sqrt(1.0 + C_SELF * p12), rel=1e-12)
    assert split.f[0] == pytest.approx(f0 * math.sqrt(1.0 - C_SELF * p12), rel=1e-12)
    assert len(split) == 2
    root = 0.5 * math.sqrt(2.0)
    assert split.v == pytest.approx(np.array([[root, -root], [root, root]]), abs=1e-12)
    assert split.participation == pytest.approx([2.0, 2.0], rel=1e-12)


def test_only_the_antiphase_mode_drives_the_mutual_capacitance():
    p12 = pair.mutual_coefficient(2.0, 1.5, 1.5)
    matrix = pair.maxwell_capacitance(C_SELF, C_SELF, p12)
    split = pair.coupled_modes(L_SELF, L_SELF, matrix)
    mutual = pair.mutual_capacitance(matrix)
    ground = matrix[0, 0] - mutual
    assert split.f[0] == pytest.approx(
        split.f[1] / math.sqrt(1.0 + 2.0 * mutual / ground), rel=1e-12
    )


def test_an_uncoupled_pair_returns_the_isolated_frequencies():
    l_b, c_b = 1.1 * L_SELF, 1.3 * C_SELF
    matrix = pair.maxwell_capacitance(C_SELF, c_b, 0.0)
    assert matrix.tolist() == [[C_SELF, 0.0], [0.0, c_b]]
    isolated = sorted(
        1.0 / (2.0 * math.pi * math.sqrt(l * c))
        for l, c in ((L_SELF, C_SELF), (l_b, c_b))
    )
    # Exact but for the eigen-solve's own sqrt round trip through D C D.
    assert pair.coupled_modes(L_SELF, l_b, matrix).f == pytest.approx(
        isolated, rel=1e-15
    )
    assert built(np.inf).split.f == pytest.approx(
        sorted(built().frequencies.tolist()), rel=1e-15
    )


@pytest.mark.parametrize("p12, tolerance", [(1.0e8, 2.0e-6), (1.0e9, 2.0e-4)])
def test_the_splitting_follows_the_two_level_form(p12, tolerance):
    for detune in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        split, detuned, coupled = two_level(detune, p12)
        exact = (split.f[1] - split.f[0]) / split.f.mean()
        # Second order in the coupling: a tenth of p12 buys a hundredth of this.
        assert exact == pytest.approx(math.hypot(detuned, coupled), rel=tolerance)


@pytest.mark.parametrize("p12", [1.0e8, 1.0e9])
def test_the_modes_delocalise_where_the_locking_criterion_says(p12):
    for detune in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        split, detuned, coupled = two_level(detune, p12)
        if abs(detuned - coupled) < 1.0e-3 * coupled:
            continue
        shared = split.participation > 4.0 / 3.0
        assert shared.tolist() == [pair.locks(coupled, detuned)] * 2


def test_the_pair_reduces_its_designs_to_their_own_resonances():
    pairing = built()
    assert pairing.frequencies == pytest.approx(
        [modes_of(LEFT).f[0], modes_of(RIGHT).f[0]], rel=1e-12
    )
    f = pairing.frequencies
    assert pairing.detune == pytest.approx(abs(f[0] - f[1]) / f.mean(), rel=1e-12)
    assert pairing.coupling < pairing.detune
    assert not pairing.locks
    assert pairing.split.participation.max() < 4.0 / 3.0
    assert pairing.gap == pytest.approx(SEPARATION - 2.0 * TOP_LOAD.outer_radius)


def test_swapping_the_two_coils_leaves_the_split_alone():
    forward, reverse = built(), built(a=RIGHT, b=LEFT)
    assert reverse.split.f == pytest.approx(forward.split.f, rel=1e-12)
    assert reverse.mutual == pytest.approx(forward.mutual, rel=1e-12)
    assert reverse.coupling == pytest.approx(forward.coupling, rel=1e-12)


def test_mutual_capacitance_and_splitting_fall_with_separation():
    pairs = [built(s) for s in (0.5, 1.0, 2.0, 4.0)]
    mutual = [p.mutual for p in pairs]
    splitting = [p.split.f[1] - p.split.f[0] for p in pairs]
    assert np.all(np.diff(mutual) < 0.0)
    assert np.all(np.diff(splitting) < 0.0)
    assert np.all(np.array(mutual) > 0.0)


def test_the_pair_solves_the_resonances_it_is_not_given():
    pairing = pair.Pair(LEFT, LEFT, SEPARATION)
    assert pairing.detune == pytest.approx(0.0, abs=1e-15)
    assert pairing.frequencies[0] == pytest.approx(modes_of(LEFT).f[0], rel=1e-12)
    assert pairing.locks


def test_the_pair_rejects_what_it_cannot_reduce():
    with pytest.raises(ValueError, match="not positive"):
        pair.Pair(LEFT, RIGHT, 0.0)
    with pytest.raises(ValueError, match="ground plane"):
        pair.Pair(LEFT, replace(RIGHT, ground_plane=False), SEPARATION)
    with pytest.raises(ValueError, match="indefinite"):
        pair.maxwell_capacitance(C_SELF, C_SELF, 2.0 / C_SELF)


def test_an_antiphase_pair_bridges_the_sum_of_the_two_reaches():
    assert pair.bridged_gap(1.5, 2.5) == 4.0
    assert pair.bridges(4.0, 1.5, 2.5)
    assert not pair.bridges(4.0 + 1e-9, 1.5, 2.5)
    pairing = built()
    assert pairing.bridges(0.5 * pairing.gap, 0.5 * pairing.gap)
    assert not pairing.bridges(0.1 * pairing.gap, 0.1 * pairing.gap)
