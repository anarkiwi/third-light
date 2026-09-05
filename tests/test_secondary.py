"""Ladder assembly, eigen-solve, modal reduction and their validation."""

import math
import sys
from dataclasses import replace

import numpy as np
import pytest

from thirdlight.em import inductance
from thirdlight.em.capacitance import capacitance_matrix, lumped_capacitance
from thirdlight.geometry import Design, Primary, Solenoid, Sphere, Toroid
from thirdlight.secondary import (
    Ladder,
    coupling,
    eigenmodes,
    incidence_matrix,
    ladder,
    node_map,
    resonance,
    stiffness_matrix,
)

LINE_L, LINE_C = 1e-3, 1e-9
SECONDARY = Solenoid(
    radius=0.076, length=0.5, turns=1000, wire_diameter=4e-4, base=0.05
)
PRIMARY = Primary(inner_radius=0.115, turns=5.5, pitch=0.012, base=0.02)
TOP_LOAD = Toroid(major_radius=0.15, minor_radius=0.05, height=0.62)
DRSSTC = Design(
    secondary=SECONDARY,
    primary=PRIMARY,
    top_load=TOP_LOAD,
    sections=200,
    top_load_sections=32,
)
SMALL = Design(
    secondary=Solenoid(
        radius=0.04, length=0.2, turns=60, wire_diameter=3e-4, base=0.02
    ),
    primary=Primary(inner_radius=0.06, turns=4.0, pitch=0.01),
    top_load=Toroid(major_radius=0.06, minor_radius=0.02, height=0.24),
    sections=30,
    top_load_sections=8,
)

# Medhurst, Wireless Engineer 24, Feb/Mar 1947: C_L[pF] = H * D[cm]; H by l/D.
MEDHURST = {
    1.0: 0.46,
    1.5: 0.47,
    2.0: 0.50,
    2.5: 0.56,
    3.0: 0.61,
    3.5: 0.67,
    4.0: 0.72,
    4.5: 0.77,
    5.0: 0.81,
}


def uniform_ladder(sections):
    """Lossless uniform line: equal per-section L and C, no mutual terms."""
    return Ladder(
        L=np.eye(sections) * LINE_L / sections,
        C=np.eye(sections) * LINE_C / sections,
        z=np.arange(sections) / sections,
    )


def medhurst_coil(ratio, diameter=0.10, turns=300):
    """Close-wound solenoid of the given length/diameter ratio."""
    return Solenoid(
        radius=0.5 * diameter,
        length=ratio * diameter,
        turns=turns,
        wire_diameter=0.5 * ratio * diameter / turns,
    )


def isolated(coil, sections=150):
    """Base-grounded coil remote from other objects, as Medhurst's C_L assumes."""
    return Design(
        secondary=coil,
        primary=PRIMARY,
        top_load=None,
        ground_plane=False,
        sections=sections,
    )


def medhurst_frequency(coil, factor):
    """1 / (2 pi sqrt(L_dc C_L)) with C_L = factor * D, D in cm and C_L in pF.

    H values as transcribed from Medhurst's table in
    pupman.com/listarchives/1997/march/msg00671.html, matching his own regression
    C_L/D = 0.1126(l/D) + 0.08 + 0.27 sqrt(D/l) to 1 % up to l/D = 2.
    """
    capacitance = factor * (2.0 * coil.radius * 100.0) * 1e-12
    return 1.0 / (
        2.0 * math.pi * math.sqrt(inductance.solenoid_inductance(coil) * capacitance)
    )


def sign_changes(profile):
    """Number of interior sign changes of a node profile."""
    return int(np.count_nonzero(np.diff(np.sign(profile))))


def test_incidence_matrix_differences_successive_currents():
    matrix = incidence_matrix(4)
    current = np.array([4.0, 3.0, 2.0, 1.0])
    assert np.array_equal(matrix @ current, [1.0, 1.0, 1.0, 1.0])
    assert np.array_equal(
        matrix.T @ np.array([1.0, 2.0, 4.0, 8.0]), [1.0, 1.0, 2.0, 4.0]
    )


def test_node_map_ties_every_top_load_ring_to_the_top_node():
    groups = np.array([0, 0, 1, 1, 2, 2])
    node = node_map(groups, 3)
    assert node.shape == (9, 3)
    assert np.array_equal(node.sum(axis=1), np.ones(9))
    assert np.array_equal(node[:6].argmax(axis=1), groups)
    assert np.array_equal(node[6:].argmax(axis=1), np.full(3, 2))


def test_merging_a_conductor_is_exact():
    """T.T C T over all rings of one conductor is its lumped capacitance exactly."""
    rings = Sphere(radius=0.4, height=1000.0).discretise(60)
    merged = np.ones(60) @ capacitance_matrix(rings, ground_plane=False) @ np.ones(60)
    assert merged == pytest.approx(
        lumped_capacitance(rings, ground_plane=False), rel=1e-12
    )


def test_uniform_line_matches_the_closed_form_spectrum():
    """A A.T is the Dirichlet-Neumann Laplacian, omega_j = 2N sin((2j-1)pi/(4N+2))/sqrt(LC)."""
    sections = 40
    modes = eigenmodes(uniform_ladder(sections), 6)
    order = np.arange(1, 7)
    exact = (
        2.0
        * sections
        * np.sin((2 * order - 1) * math.pi / (4 * sections + 2))
        / (2.0 * math.pi * math.sqrt(LINE_L * LINE_C))
    )
    assert modes.f == pytest.approx(exact, rel=1e-12)


def test_uniform_line_approaches_the_quarter_wave_ratios():
    """f_m/f_1 -> 1:3:5:7 and f_1 -> 1/(4 sqrt(LC)), the ratio error falling as 1/N^2."""
    odd = np.array([1.0, 3.0, 5.0, 7.0])
    quarter = 1.0 / (4.0 * math.sqrt(LINE_L * LINE_C))
    errors = []
    for sections in (25, 50, 100, 200):
        modes = eigenmodes(uniform_ladder(sections), 4)
        assert modes.f[0] == pytest.approx(quarter, rel=1.0 / sections)
        errors.append(np.abs(modes.f / modes.f[0] / odd - 1.0).max())
    errors = np.array(errors)
    assert np.all(np.diff(errors) < 0.0)
    assert errors[1:] / errors[:-1] == pytest.approx(0.25, rel=0.05)


def test_stiffness_is_symmetric_with_a_positive_c_orthogonal_spectrum():
    rungs = ladder(SMALL)
    stiffness = stiffness_matrix(rungs.L)
    assert np.abs(stiffness - stiffness.T).max() <= (
        sys.float_info.epsilon * np.abs(stiffness).max()
    )
    assert np.linalg.eigvalsh(stiffness).min() > 0.0
    modes = eigenmodes(rungs, 6)
    assert np.all(modes.f > 0.0)
    assert modes.v @ rungs.C @ modes.v.T == pytest.approx(np.eye(6), abs=1e-12)
    assert modes.i @ rungs.L @ modes.i.T == pytest.approx(np.eye(6), abs=1e-8)


def test_ladder_shapes_and_top_referred_equivalents():
    rungs = ladder(SMALL)
    assert len(rungs) == SMALL.sections
    assert rungs.L.shape == rungs.C.shape == (SMALL.sections, SMALL.sections)
    assert rungs.L.sum() == pytest.approx(
        inductance.solenoid_inductance(SMALL.secondary), rel=1e-12
    )
    assert np.all(np.diff(rungs.z) > 0.0)
    modes = eigenmodes(rungs, 3)
    assert len(modes) == 3
    assert modes.c_m == pytest.approx(1.0 / modes.v[:, -1] ** 2, rel=1e-12)
    assert modes.l_m * modes.c_m == pytest.approx(
        1.0 / (2.0 * math.pi * modes.f) ** 2, rel=1e-12
    )


def test_mode_shapes_are_quarter_wave_harmonics():
    modes = resonance(SMALL, modes=3)
    assert [sign_changes(profile) for profile in modes.v] == [0, 1, 2]
    assert np.all(np.diff(modes.v[0]) > 0.0)
    assert np.all(modes.i[0] > 0.0)


def test_resonance_falls_and_coupling_holds_as_sections_are_refined():
    coarse = resonance(SMALL, sections=15, modes=1)
    fine = resonance(SMALL, sections=60, modes=1)
    assert coarse.f[0] < fine.f[0]
    assert coupling(SMALL, coarse)[0] == pytest.approx(
        coupling(SMALL, fine)[0], rel=0.02
    )
    assert 0.0 < coupling(SMALL, fine)[0] < 1.0


@pytest.mark.slow
@pytest.mark.parametrize("ratio", [1.0, 2.0, 3.0, 4.0, 5.0])
def test_medhurst_self_resonance(ratio):
    """Predicted f_res exceeds 1/(2 pi sqrt(L_dc H D)) by 2.6-9.4 % over l/D = 1..5.

    Medhurst's coils were wound on solid polystyrene rods (Knight, hamwaves.com),
    so his C_L carries a former dielectric this air-only model does not; the
    residual is one-signed, and the design's 1-2 % tolerance is not met.
    """
    coil = medhurst_coil(ratio)
    error = resonance(isolated(coil), modes=1).f[0] / medhurst_frequency(
        coil, MEDHURST[ratio]
    )
    assert 1.0 < error < 1.10


@pytest.mark.slow
@pytest.mark.parametrize("turns", [200, 800])
def test_effective_capacitance_per_diameter_is_turn_count_independent(turns):
    """Medhurst's central finding: C_L/D is a function of l/D alone, here to 0.4 %."""
    coil = medhurst_coil(3.0, turns=turns)
    omega = 2.0 * math.pi * resonance(isolated(coil), modes=1).f[0]
    effective = 1.0 / (omega**2 * inductance.solenoid_inductance(coil))
    assert effective * 1e12 / (2.0 * coil.radius * 100.0) == pytest.approx(
        0.5433, rel=0.004
    )


@pytest.mark.slow
def test_f_res_converges_in_sections():
    """f_res at N = 50, 100, 200, 400 is 172.87, 173.71, 174.09, 174.25 kHz."""
    f = np.array(
        [resonance(DRSSTC, sections=n, modes=1).f[0] for n in (50, 100, 200, 400)]
    )
    assert np.all(np.diff(f) > 0.0)
    assert f[3] / f[2] - 1.0 < 0.005
    assert f[3] == pytest.approx(174.25e3, rel=0.001)


@pytest.mark.slow
def test_f_res_converges_in_top_load_sections():
    f = np.array(
        [
            resonance(replace(DRSSTC, top_load_sections=n), modes=1).f[0]
            for n in (8, 32, 128)
        ]
    )
    assert np.all(np.diff(f) > 0.0)
    assert f[2] / f[1] - 1.0 < 0.002


@pytest.mark.slow
def test_top_load_lowers_f_res_towards_the_bare_self_resonance():
    """The unloaded coil sits 6.8 % above its Medhurst self-resonance, as the l/D sweep does."""
    loaded = np.array(
        [
            resonance(replace(DRSSTC, top_load=Toroid(r, 0.05, 0.62)), modes=1).f[0]
            for r in (0.10, 0.14, 0.20)
        ]
    )
    assert np.all(np.diff(loaded) < 0.0)
    ratio = SECONDARY.length / (2.0 * SECONDARY.radius)
    factor = np.interp(ratio, list(MEDHURST), list(MEDHURST.values()))
    bare = resonance(isolated(SECONDARY, sections=200), modes=1).f[0]
    assert bare > loaded.max()
    assert 1.0 < bare / medhurst_frequency(SECONDARY, factor) < 1.10


@pytest.mark.slow
def test_coupling_lands_in_the_drsstc_band_and_tracks_geometry():
    """k rises as the primary climbs toward the secondary base and tightens on it.

    The mode-1 current peaks at the base where the primary sits, so k exceeds the
    uniform-current k_dc by a few per cent.
    """
    modes = resonance(DRSSTC, modes=4)
    k = coupling(DRSSTC, modes)
    assert 0.1 < k[0] < 0.3
    assert np.all(np.abs(k) < 1.0)
    primary = DRSSTC.primary_rings()
    k_dc = inductance.mutual_matrix(primary, SECONDARY.discretise()).sum() / math.sqrt(
        inductance.inductance_matrix(primary).sum()
        * inductance.solenoid_inductance(SECONDARY)
    )
    assert k_dc < k[0] < 1.05 * k_dc
    heights = [
        coupling(replace(DRSSTC, primary=replace(PRIMARY, base=b)), modes)[0]
        for b in (-0.10, -0.05, 0.0, 0.05)
    ]
    radii = [
        coupling(replace(DRSSTC, primary=replace(PRIMARY, inner_radius=r)), modes)[0]
        for r in (0.30, 0.20, 0.13, 0.09)
    ]
    assert np.all(np.diff(heights) > 0.0)
    assert np.all(np.diff(radii) > 0.0)
