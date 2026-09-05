"""Ring-charge potential coefficients, capacitance extraction and validation."""

import math
import sys
from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import epsilon_0
from scipy.integrate import quad

from thirdlight.em.capacitance import (
    capacitance_matrix,
    lumped_capacitance,
    potential_matrix,
    potential_ring,
    reduce_sections,
    resonator_capacitance,
    self_potential,
    surface_field,
    unit_potential_charges,
)
from thirdlight.geometry import Design, Primary, Rings, Solenoid, Sphere, Toroid

FAR = 1000.0
SPHERE_R = 0.4
SECONDARY = Solenoid(
    radius=0.076, length=0.5, turns=1000, wire_diameter=4e-4, base=0.05
)
PRIMARY = Primary(inner_radius=0.115, turns=5.5, pitch=0.012)
TOP_LOAD = Toroid(major_radius=0.15, minor_radius=0.05, height=0.62)

# Medhurst, Wireless Engineer 24(281), Feb 1947, table of C[pF] = H * D[cm].


def ring_quadrature(r, z_field, a, z_source):
    """Direct quadrature of 1/(4 pi eps0 |x - x'|) over a unit-charge ring."""
    dz = z_field - z_source

    def integrand(t):
        return 1.0 / math.sqrt(r * r + a * a - 2.0 * r * a * math.cos(t) + dz * dz)

    value = quad(integrand, 0.0, 2.0 * math.pi, epsabs=0.0, epsrel=1e-13, limit=200)[0]
    return value / (8.0 * math.pi**2 * epsilon_0)


def bert_pool(d1, d2):
    """Bert Pool toroid equation, d1 outside and d2 tube diameter in inches, C in pF.

    C = 2.8 (1.2781 - d2/d1) sqrt(2 pi^2 (d1 - d2) (d2/2) / (4 pi)); see the toroid
    page of the Kaizer Power Electronics DRSSTC design guide.
    """
    return 1.4 * (1.2781 - d2 / d1) * math.sqrt(math.pi * (d1 - d2) * d2)


def toroid_of(d1, d2, height=FAR):
    """Toroid from outside and tube diameters in inches."""
    return Toroid(0.5 * (d1 - d2) * 0.0254, 0.5 * d2 * 0.0254, height)


def kelvin_ratio(hr):
    """Sphere above a grounded plane: C / (4 pi eps0 R) at height ratio ``hr``."""
    alpha = math.acosh(hr)
    total, n = 0.0, 1
    while True:
        term = 1.0 / math.sinh(n * alpha)
        total += term
        if term < 1e-15:
            return math.sinh(alpha) * total
        n += 1


def full_rings(spec):
    """Secondary and top load as one ring set."""
    return Rings.concat(spec.secondary_rings(), spec.top_load_rings())


def design(sections=40, top_load=TOP_LOAD, ground_plane=True):
    """Realistic secondary, optionally with top load and ground plane."""
    return Design(
        secondary=SECONDARY,
        primary=PRIMARY,
        top_load=top_load,
        ground_plane=ground_plane,
        sections=sections,
        top_load_sections=32,
    )


@pytest.mark.parametrize(
    "r,z_field,a,z_source",
    [
        (1.0, 0.0, 1.0, 0.5),
        (1.0, 0.0, 0.01, 0.0),
        (0.05, 0.1, 0.2, 0.0),
        (2.5, 0.0, 0.3, 1.0),
        (1e-3, 0.0, 1.0, 0.0),
        (1.0, 3.0, 1.0, 0.0),
        (1.0, 0.0, 1.0, 1e-3),
    ],
)
def test_potential_ring_matches_quadrature(r, z_field, a, z_source):
    value = potential_ring(r, z_field, a, z_source)
    assert value == pytest.approx(ring_quadrature(r, z_field, a, z_source), rel=1e-10)
    swapped = potential_ring(r=a, z_field=z_source, a=r, z_source=z_field)
    assert value == pytest.approx(swapped, rel=1e-15)


def test_potential_ring_on_axis_is_a_point_charge_at_the_slant_distance():
    value = potential_ring(0.0, 0.0, 0.5, 3.0)
    distance = math.hypot(0.5, 3.0)
    assert value == pytest.approx(
        1.0 / (4.0 * math.pi * epsilon_0 * distance), rel=1e-15
    )


def test_self_potential_is_the_thin_torus_capacitance():
    a, rw = 0.2, 2e-3
    exact = 4.0 * math.pi**2 * epsilon_0 * a / math.log(8.0 * a / rw)
    assert 1.0 / self_potential(a, rw) == pytest.approx(exact, rel=1e-15)


@pytest.mark.parametrize("sections", [25, 50, 100, 200])
def test_isolated_sphere_capacitance(sections):
    """Converges as 0.226 / N from above, so N = 100 is inside 0.5 %."""
    rings = Sphere(SPHERE_R, 1.0).discretise(sections)
    capacitance = lumped_capacitance(rings, ground_plane=False)
    error = capacitance / (4.0 * math.pi * epsilon_0 * SPHERE_R) - 1.0
    assert 0.0 < error < 0.25 / sections


def test_isolated_sphere_surface_field():
    rings = Sphere(SPHERE_R, 1.0).discretise(100)
    charges = unit_potential_charges(rings, ground_plane=False)
    field = surface_field(rings, charges)
    exact = charges.sum() / (4.0 * math.pi * epsilon_0 * SPHERE_R**2)
    assert field[1:-1] == pytest.approx(exact, rel=0.01)
    # The two polar rings have a ~ 2 rw, where the thin-torus self term is weakest.
    assert field[[0, -1]] == pytest.approx(exact, rel=0.11)


@pytest.mark.parametrize("ratio,tol", [(0.02, 3e-3), (0.05, 6e-3)])
def test_thin_toroid_capacitance(ratio, tol):
    """Slender-torus limit C = 4 pi^2 eps0 a / ln(8a/b), the dual of the ring self term."""
    a = 0.25
    rings = Toroid(a, ratio * a, FAR).discretise(64)
    exact = 4.0 * math.pi**2 * epsilon_0 * a / math.log(8.0 / ratio)
    assert lumped_capacitance(rings, ground_plane=False) == pytest.approx(
        exact, rel=tol
    )


@pytest.mark.parametrize("d1,d2", [(6.0, 1.5), (4.0, 1.0), (10.0, 2.0), (30.0, 6.0)])
def test_toroid_capacitance_against_bert_pool(d1, d2):
    rings = toroid_of(d1, d2).discretise(64)
    capacitance = lumped_capacitance(rings, ground_plane=False)
    assert capacitance * 1e12 == pytest.approx(bert_pool(d1, d2), rel=0.03)


@pytest.mark.parametrize("hr", [4.0, 10.0, 20.0])
def test_ground_plane_image_correction(hr):
    rings = Sphere(SPHERE_R, hr * SPHERE_R).discretise(200)
    isolated = lumped_capacitance(rings, ground_plane=False)
    grounded = lumped_capacitance(rings, ground_plane=True)
    assert isolated == pytest.approx(
        lumped_capacitance(Sphere(SPHERE_R, FAR).discretise(200), False), rel=1e-12
    )
    assert grounded / isolated == pytest.approx(kelvin_ratio(hr), rel=1e-3)
    assert grounded / isolated == pytest.approx(1.0 + 0.5 / hr, rel=1.0 / hr**2)


def test_solenoid_static_capacitance_brackets_the_medhurst_value():
    """Static C is below the circumscribed sphere and above Medhurst's lumped C_L."""
    coil = Solenoid(radius=0.05, length=0.1, turns=200, wire_diameter=5e-4)
    capacitance = lumped_capacitance(coil.discretise(), ground_plane=False)
    outer = 4.0 * math.pi * epsilon_0 * math.hypot(0.05, 0.05)
    assert 0.46 * 10.0 * 1e-12 < capacitance < outer


def test_potential_matrix_is_symmetric_positive_definite():
    rings = full_rings(design(sections=120))
    matrix = potential_matrix(rings, ground_plane=True)
    assert np.array_equal(matrix, matrix.T)
    assert np.linalg.eigvalsh(matrix).min() > 0.0
    np.linalg.cholesky(matrix)


@pytest.mark.parametrize(
    "sections", [50, 200, pytest.param(400, marks=pytest.mark.slow)]
)
def test_capacitance_matrix_inverts_the_potential_matrix(sections):
    """cond(P) grows linearly in ring count: 2.2e2, 7.3e2, 1.4e3 at 50, 200, 400."""
    rings = full_rings(design(sections=sections))
    matrix = potential_matrix(rings, ground_plane=True)
    condition = np.linalg.cond(matrix)
    assert condition < 10.0 * len(rings)
    residual = capacitance_matrix(rings) @ matrix - np.eye(len(rings))
    assert np.abs(residual).max() < sys.float_info.epsilon * condition


def test_reduce_sections_block_means():
    matrix = np.arange(16.0).reshape(4, 4)
    reduced = reduce_sections(matrix, np.array([0, 0, 1, 1]))
    assert np.array_equal(reduced, [[2.5, 4.5], [10.5, 12.5]])
    assert np.array_equal(reduce_sections(matrix, np.arange(4)), matrix)


def test_reduce_sections_preserves_the_unit_potential_charge():
    """Averaging P is the reduction that leaves the unit-potential charge fixed."""
    rings = SECONDARY.discretise(60)
    matrix = potential_matrix(rings, ground_plane=True)
    reduced = reduce_sections(matrix, np.arange(60) // 10)
    assert np.linalg.solve(reduced, np.ones(6)).sum() == pytest.approx(
        np.linalg.solve(matrix, np.ones(60)).sum(), rel=0.02
    )


def test_reduce_sections_rejects_bad_groups():
    matrix = np.zeros((4, 4))
    with pytest.raises(ValueError, match="does not index"):
        reduce_sections(matrix, np.zeros(3, dtype=int))
    with pytest.raises(ValueError, match="contiguous"):
        reduce_sections(matrix, np.array([0, 1, 0, 1]))


def test_resonator_capacitance_rises_with_top_load_and_ground_plane():
    bare = replace(design(), top_load=None, ground_plane=False)
    loaded = replace(bare, top_load=TOP_LOAD)
    assert resonator_capacitance(bare) < resonator_capacitance(loaded)
    assert resonator_capacitance(loaded) < resonator_capacitance(design())
    assert resonator_capacitance(bare) == pytest.approx(
        lumped_capacitance(bare.secondary_rings(), ground_plane=False), rel=1e-15
    )


@pytest.mark.slow
def test_resonator_capacitance_is_converged_at_a_few_hundred_sections():
    coarse = resonator_capacitance(design(sections=400))
    fine = resonator_capacitance(replace(design(sections=1200), top_load_sections=100))
    assert coarse == pytest.approx(fine, rel=0.01)
