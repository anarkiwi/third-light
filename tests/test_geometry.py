"""Geometry discretisation invariants."""

import numpy as np
import pytest

from thirdlight.geometry import Design, Primary, Rings, Solenoid, Sphere, Toroid

EXAMPLE = "examples/sstc.yaml"


def test_solenoid_discretisation_conserves_turns_and_extent():
    coil = Solenoid(radius=0.076, length=0.5, turns=1200, wire_diameter=4e-4, base=0.05)
    rings = coil.discretise(160)
    assert len(rings) == 160
    assert rings.n.sum() == pytest.approx(coil.turns)
    assert rings.w.sum() == pytest.approx(coil.length)
    assert rings.z.min() - 0.5 * rings.w[0] == pytest.approx(coil.base)
    assert rings.z.max() + 0.5 * rings.w[0] == pytest.approx(coil.base + coil.length)
    assert coil.pitch == pytest.approx(coil.length / coil.turns)
    assert coil.wire_length > 2 * np.pi * coil.radius * coil.turns


def test_toroid_rings_lie_on_the_tube():
    top = Toroid(major_radius=0.15, minor_radius=0.05, height=0.6)
    rings = top.discretise(64)
    r = np.hypot(rings.a - top.major_radius, rings.z - top.height)
    assert np.allclose(r, top.minor_radius)
    assert top.outer_radius == pytest.approx(0.2)


def test_sphere_rings_lie_on_the_surface():
    top = Sphere(radius=0.1, height=0.5)
    rings = top.discretise(32)
    assert np.allclose(np.hypot(rings.a, rings.z - top.height), top.radius)
    assert top.outer_radius == pytest.approx(0.1)


def test_primary_spiral_and_helix():
    spiral = Primary(inner_radius=0.115, turns=5.5, pitch=0.012).discretise()
    assert len(spiral) == 6
    assert spiral.n.sum() == pytest.approx(5.5)
    assert np.allclose(spiral.z, spiral.z[0])
    helix = Primary(inner_radius=0.115, turns=4, rise=0.02).discretise()
    assert np.allclose(helix.a, 0.115)
    assert helix.z[-1] > helix.z[0]


def test_rings_concat_and_mirror():
    top = Toroid(0.15, 0.05, 0.6).discretise(8)
    both = Rings.concat(Solenoid(0.076, 0.5, 900, 4e-4).discretise(10), top)
    assert len(both) == 18
    assert np.allclose(both.mirrored().z, -both.z)


def test_rings_reject_ragged_fields():
    with pytest.raises(ValueError):
        Rings(np.zeros(3), np.zeros(2), np.zeros(3), np.zeros(3), np.zeros(3))


def test_design_from_yaml_matches_example():
    design = Design.from_yaml(EXAMPLE)
    assert len(design.secondary_rings()) == design.sections
    assert len(design.top_load_rings()) == design.top_load_sections
    assert len(design.primary_rings()) == 6
    assert design.ground_plane


def test_design_without_top_load_has_no_rings():
    design = Design(Solenoid(0.076, 0.5, 900, 4e-4), Primary(0.1, 5, 0.01))
    assert len(design.top_load_rings()) == 0


def test_design_rejects_unknown_top_load():
    with pytest.raises(ValueError, match="unknown top load"):
        Design.from_dict(
            {
                "secondary": {
                    "radius": 0.1,
                    "length": 0.5,
                    "turns": 900,
                    "wire_diameter": 4e-4,
                },
                "primary": {"inner_radius": 0.1, "turns": 5},
                "top_load": {"kind": "cube", "radius": 0.1, "height": 0.5},
            }
        )
