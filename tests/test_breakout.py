"""Breakout onset: Peek's threshold and the electrode surface field functional."""

import math

import numpy as np
import pytest
from scipy.constants import epsilon_0

from thirdlight import secondary
from thirdlight.discharge import breakout
from thirdlight.em import capacitance, inductance
from thirdlight.geometry import Design, Primary, Solenoid, Sphere, Toroid

SECONDARY = Solenoid(radius=0.05, length=0.30, turns=60, wire_diameter=3e-4, base=0.02)
PRIMARY = Primary(inner_radius=0.08, turns=5.0, pitch=0.01, base=0.025)
TOP_LOAD = Toroid(major_radius=0.09, minor_radius=0.03, height=0.36)
POINT = Sphere(radius=0.004, height=0.40)


def coil(**changes):
    """Small coil with a toroid and a breakout point."""
    spec = {
        "secondary": SECONDARY,
        "primary": PRIMARY,
        "top_load": TOP_LOAD,
        "breakout": POINT,
        "sections": 20,
        "top_load_sections": 12,
        "breakout_sections": 8,
    }
    spec.update(changes)
    return Design(**spec)


def functional(design, modes=3):
    """Breakout functional and the ladder it came from."""
    rungs = secondary.ladder(design)
    return breakout.from_modes(design, rungs, secondary.eigenmodes(rungs, modes)), rungs


def direct_field(design, rungs, modes, state):
    """Electrode surface field at a modal state, by re-solving the potential problem."""
    turns = design.secondary.discretise()
    groups = inductance.turn_groups(len(turns), len(rungs))
    node = secondary.node_map(groups, len(design.top_load_rings()))
    potential = node @ (breakout.shapes(modes) @ state)
    charges = np.linalg.solve(
        capacitance.potential_matrix(
            rungs.rings, design.ground_plane, design.dielectric()
        ),
        potential,
    )
    field = capacitance.surface_field(rungs.rings, charges)[rungs.electrode]
    return field * breakout.correction(design)


def test_peek_reaches_the_uniform_field_strength_on_a_large_conductor():
    """The curvature term is the ionisation layer's depth; it vanishes as r grows."""
    assert breakout.critical_field(1e4) == pytest.approx(
        breakout.DISRUPTIVE_FIELD, rel=1e-3
    )
    assert breakout.critical_field(0.01) == pytest.approx(3.903e6, rel=1e-3)
    assert breakout.critical_field(0.001) > breakout.critical_field(0.01)


def test_peek_scales_with_air_density():
    """delta multiplies the whole threshold and stiffens the curvature term."""
    assert breakout.relative_density() == pytest.approx(1.0)
    assert breakout.relative_density(pressure=0.5 * breakout.STANDARD_PRESSURE) == 0.5
    assert (
        breakout.relative_density(temperature=2.0 * breakout.STANDARD_TEMPERATURE)
        == 0.5
    )
    dense = breakout.critical_field(0.01)
    thin = breakout.critical_field(0.01, density=0.5)
    # Thin air breaks down sooner, but the ionisation layer deepens as 1/sqrt(delta).
    assert 0.5 * dense < thin < dense


@pytest.mark.parametrize("radius", [0.05, 0.37])
@pytest.mark.parametrize("sections", [8, 17])
def test_the_sphere_correction_makes_an_isolated_sphere_field_uniform(radius, sections):
    """An isolated sphere's field is uniform, and the ratio is scale free."""
    rings = Sphere(radius, 1000.0).discretise(sections)
    charges = capacitance.unit_potential_charges(rings, ground_plane=False)
    field = capacitance.surface_field(rings, charges) * capacitance.field_correction(
        Sphere(radius, 0.0), sections
    )
    exact = charges.sum() / (4.0 * math.pi * epsilon_0 * radius**2)
    assert field == pytest.approx(exact, rel=1e-9)


def test_the_polar_cap_error_does_not_converge_but_the_correction_removes_it():
    """Refining the sphere only makes a smaller cap of the same shape."""
    ratios = [capacitance.sphere_field_correction(n)[0] for n in (8, 32, 128)]
    assert ratios == pytest.approx([1.1183] * 3, rel=1e-3)


def test_the_correction_transfers_to_a_mounted_breakout_point():
    """Corrected, the coarse pole field tracks the refined one; uncorrected it cannot."""
    fine = functional(coil(breakout_sections=48))[0].field[-48, 0]
    coarse = functional(coil(breakout_sections=12))[0].field[-12, 0]
    raw = coarse / capacitance.sphere_field_correction(12)[0]
    assert coarse == pytest.approx(fine, rel=0.03)
    assert raw < 0.92 * fine


def test_the_field_functional_matches_a_direct_potential_solve():
    """One matrix built per design replaces a method of moments solve per step."""
    design = coil()
    rungs = secondary.ladder(design)
    modes = secondary.eigenmodes(rungs, 3)
    hot = breakout.from_modes(design, rungs, modes)
    state = np.array([1.7e5, -2.0e4, 5.0e3])
    assert hot.field @ state == pytest.approx(
        direct_field(design, rungs, modes, state), rel=1e-11
    )
    assert hot.stress(state) == pytest.approx(
        np.max(np.abs(direct_field(design, rungs, modes, state))), rel=1e-11
    )


def test_margin_reaches_one_at_the_breakout_voltage():
    """The reported voltage is where the first mode alone brings the surface to threshold."""
    hot = functional(coil())[0]
    assert hot.margin([hot.voltage, 0.0, 0.0]) == pytest.approx(1.0, rel=1e-12)
    assert 3e4 < hot.voltage < 5e5


def test_margin_is_batched_over_states():
    """A run's whole history is one matrix product."""
    hot = functional(coil())[0]
    states = np.array([[1e5, 0.0, 0.0], [-2e5, 1e4, 0.0], [0.0, 0.0, 0.0]])
    assert hot.margin(states) == pytest.approx([hot.margin(s) for s in states])
    assert hot.margin(states)[2] == 0.0


def test_a_breakout_point_breaks_out_before_the_toroid_it_sits_on():
    """That is what the point is for: a sharper electrode at the same potential."""
    blunt = functional(coil(breakout=None))[0]
    sharp = functional(coil())[0]
    assert sharp.voltage < 0.7 * blunt.voltage
    assert sharp.critical[-1] > blunt.critical[-1]


def test_a_smaller_point_breaks_out_at_a_lower_voltage():
    """Field enhancement beats the rise in Peek's threshold at these radii."""
    small = functional(coil(breakout=Sphere(radius=0.002, height=0.40)))[0]
    large = functional(coil(breakout=Sphere(radius=0.008, height=0.40)))[0]
    assert small.voltage < large.voltage


def test_without_a_top_load_the_electrode_is_the_last_turn():
    """A bare coil still breaks out, from the wire itself."""
    design = coil(top_load=None, breakout=None)
    hot, rungs = functional(design)
    assert rungs.top == 0
    assert hot.field.shape == (1, 3)
    assert hot.critical == pytest.approx(
        breakout.critical_field(0.5 * SECONDARY.wire_diameter)
    )
