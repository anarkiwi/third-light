"""Bound-charge dielectric operators, the field kernel, and their validation."""

import math
import sys
from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import epsilon_0
from scipy.integrate import quad
from scipy.special import ellipk as scipy_ellipk

from thirdlight.em.capacitance import (
    lumped_capacitance,
    potential_matrix,
    potential_ring,
)
from thirdlight.em.dielectric import (
    _SPLIT,
    bound_operators,
    dellipk,
    field_ring,
    polarised_potential,
)
from thirdlight.geometry import (
    Design,
    Dielectric,
    Former,
    Primary,
    Rings,
    Solenoid,
    Sphere,
)
from thirdlight.secondary import resonance

INNER, MIDDLE, OUTER = 0.1, 0.15, 0.2
PERMITTIVITY = 4.0
PRIMARY = Primary(inner_radius=0.115, turns=5.5, pitch=0.012)
FIELD_POINTS = [
    (1.0, 0.0, 1.0, 0.5),
    (0.05, 0.1, 0.2, 0.0),
    (2.5, 0.0, 0.3, 1.0),
    (1e-3, 0.0, 1.0, 0.0),
    (1.0, 3.0, 1.0, 0.0),
    (0.5, 0.0, 0.5, 0.02),
    (100.0, 0.0, 1.0, 0.0),
    (1.0, 0.0, 0.01, 0.0),
]


def coated_sphere(sections, permittivity=PERMITTIVITY):
    """Concentric shell from MIDDLE to OUTER, normals pointing out of the dielectric."""
    outer = Sphere(OUTER, 0.0).discretise(sections)
    inner = Sphere(MIDDLE, 0.0).discretise(sections)
    return Dielectric(
        rings=Rings.concat(outer, inner),
        nr=np.concatenate([outer.a / OUTER, -inner.a / MIDDLE]),
        nz=np.concatenate([outer.z / OUTER, -inner.z / MIDDLE]),
        permittivity=permittivity,
    )


def coated_exact(permittivity):
    """C = 4 pi eps0 / (1/a - 1/b + (1/b - 1/c)/eps_r + 1/c) for the coated sphere."""
    gaps = 1.0 / INNER - 1.0 / MIDDLE + 1.0 / OUTER
    return (
        4.0 * math.pi * epsilon_0 / (gaps + (1.0 / MIDDLE - 1.0 / OUTER) / permittivity)
    )


def coated_capacitance(sections, permittivity=PERMITTIVITY, conductor=64):
    """Lumped capacitance of the conducting sphere inside its dielectric shell."""
    return lumped_capacitance(
        Sphere(INNER, 0.0).discretise(conductor),
        ground_plane=False,
        dielectric=coated_sphere(sections, permittivity),
    )


def solenoid_design(permittivity=None, sections=60, former_sections=60, turns=100):
    """Medhurst-like l/D = 1 coil, wound on a rod that clears the wire radius."""
    coil = Solenoid(radius=0.05, length=0.1, turns=turns, wire_diameter=5e-4, base=0.01)
    former = (
        None
        if permittivity is None
        else Former(
            outer_radius=coil.radius - 0.5 * coil.wire_diameter,
            length=coil.length,
            base=coil.base,
            permittivity=permittivity,
        )
    )
    return Design(
        secondary=coil,
        primary=PRIMARY,
        former=former,
        sections=sections,
        former_sections=former_sections,
    )


def difference(f, x, h):
    """Five-point central difference of ``f`` at ``x``, fourth order in ``h``."""
    return (f(x - 2 * h) - 8.0 * f(x - h) + 8.0 * f(x + h) - f(x + 2 * h)) / (12.0 * h)


def ring_field_quadrature(r, z_field, a, z_source):
    """Direct quadrature of the ring field, the dual of ``ring_quadrature``."""
    dz = z_field - z_source

    def cube(t):
        return (r * r + a * a - 2.0 * r * a * math.cos(t) + dz * dz) ** 1.5

    def component(numerator):
        value = quad(
            lambda t: numerator(t) / cube(t),
            0.0,
            2.0 * math.pi,
            epsabs=1e-13,
            epsrel=1e-12,
            limit=200,
        )[0]
        return value / (8.0 * math.pi**2 * epsilon_0)

    return component(lambda t: r - a * math.cos(t)), component(lambda t: dz)


def asymmetry(sections):
    """Relative asymmetry of P_eff before it is symmetrised."""
    rings = Sphere(INNER, 0.0).discretise(sections)
    boundary = coated_sphere(sections)
    p_cb, f_bc, f_bb = bound_operators(rings, boundary, ground_plane=False)
    ga = 2.0 * epsilon_0 * boundary.susceptibility * boundary.area[:, None]
    raw = potential_matrix(rings, False) + p_cb @ np.linalg.solve(
        np.eye(len(boundary)) - ga * f_bb, ga * f_bc
    )
    return np.abs(raw - raw.T).max() / np.abs(raw).max()


@pytest.mark.parametrize("m", [1e-3, 0.05, 0.19, 0.2, 0.21, 0.5, 0.9, 0.99])
def test_dellipk_matches_a_finite_difference_of_ellipk(m):
    step = 1e-3 * min(m, 1.0 - m)
    assert dellipk(m) == pytest.approx(difference(scipy_ellipk, m, step), rel=1e-8)


@pytest.mark.parametrize("m", [0.2, 0.21, 0.5, 0.9, 0.99, 1.0 - 1e-9])
def test_dellipk_matches_the_closed_form_where_it_does_not_cancel(m):
    from scipy.special import ellipe  # pylint: disable=import-outside-toplevel

    closed = (ellipe(m) - (1.0 - m) * scipy_ellipk(m)) / (2.0 * m * (1.0 - m))
    assert dellipk(m) == pytest.approx(closed, rel=1e-14)


def test_dellipk_is_continuous_across_the_series_crossover():
    below = dellipk(0.2 * (1.0 - 1e-13))
    assert below == pytest.approx(dellipk(0.2), rel=1e-13)


def test_dellipk_survives_the_cancellation_the_closed_form_does_not():
    """K'(0) = pi/8; at m = 1e-14 the closed form has lost every digit but two."""
    from scipy.special import ellipe  # pylint: disable=import-outside-toplevel

    m = 1e-14
    closed = (ellipe(m) - (1.0 - m) * scipy_ellipk(m)) / (2.0 * m * (1.0 - m))
    assert dellipk(0.0) == pytest.approx(0.125 * math.pi, rel=1e-15)
    assert dellipk(m) == pytest.approx(0.125 * math.pi, rel=1e-13)
    assert abs(closed / (0.125 * math.pi) - 1.0) > 1e-3


@pytest.mark.parametrize("r,z_field,a,z_source", FIELD_POINTS)
def test_field_ring_is_minus_the_gradient_of_the_ring_potential(
    r, z_field, a, z_source
):
    """To 1e-9 relative, or to the difference's own cancellation floor eps phi / h."""
    e_r, e_z = field_ring(r, z_field, a, z_source)
    step = 1e-3 * min(r, a, math.hypot(r - a, z_field - z_source))
    floor = (
        4.0 * sys.float_info.epsilon * potential_ring(r, z_field, a, z_source) / step
    )
    radial = -difference(lambda x: potential_ring(x, z_field, a, z_source), r, step)
    axial = -difference(lambda x: potential_ring(r, x, a, z_source), z_field, step)
    assert e_r == pytest.approx(radial, rel=1e-8, abs=floor)
    assert e_z == pytest.approx(axial, rel=1e-8, abs=floor)


@pytest.mark.parametrize("r,z_field,a,z_source", FIELD_POINTS)
def test_field_ring_matches_quadrature(r, z_field, a, z_source):
    e_r, e_z = field_ring(r, z_field, a, z_source)
    q_r, q_z = ring_field_quadrature(r, z_field, a, z_source)
    assert e_r == pytest.approx(q_r, rel=1e-10)
    assert e_z == pytest.approx(q_z, rel=1e-10, abs=1e-10 * abs(q_r))


@pytest.mark.parametrize("a,dz", [(0.5, 3.0), (1.0, 0.1), (2.0, -1.0)])
def test_field_ring_on_axis_is_the_point_charge_field(a, dz):
    e_r, e_z = field_ring(0.0, dz, a, 0.0)
    exact = dz / (4.0 * math.pi * epsilon_0 * math.hypot(a, dz) ** 3)
    assert e_r == 0.0
    assert e_z == pytest.approx(exact, rel=1e-15)


def test_unit_permittivity_leaves_the_potential_matrix_untouched():
    """lam = 0 zeroes the bound charge, so P_eff is P to the last bit."""
    rings = Sphere(INNER, 0.0).discretise(24)
    plain = potential_matrix(rings, ground_plane=False)
    assert np.array_equal(potential_matrix(rings, False, coated_sphere(24, 1.0)), plain)


def test_coated_sphere_capacitance():
    """The dielectric term is exact to 1e-4; the residual is the conductor's own 0.25/N."""
    exact = coated_exact(PERMITTIVITY)
    assert coated_capacitance(24) == pytest.approx(exact, rel=0.01)
    assert coated_capacitance(64) == pytest.approx(exact, rel=0.005)


@pytest.mark.slow
def test_coated_sphere_capacitance_converges_with_band_count():
    exact = coated_exact(PERMITTIVITY)
    error = np.array(
        [coated_capacitance(n, conductor=n) / exact - 1.0 for n in (16, 32, 64, 128)]
    )
    assert np.all(error > 0.0)
    assert np.all(np.diff(error) < 0.0)
    assert error[-1] < 0.0025


@pytest.mark.slow
def test_coated_sphere_error_is_insensitive_to_the_near_far_split():
    """Filament-only and always-subdivided sources agree to eight digits."""
    rings = Sphere(INNER, 0.0).discretise(64)
    plain = potential_matrix(rings, ground_plane=False)
    boundary = coated_sphere(64)
    values = np.array(
        [
            np.linalg.solve(
                polarised_potential(plain, rings, boundary, False, _SPLIT * f),
                np.ones(len(rings)),
            ).sum()
            for f in (0.0, 0.5, 1.0, 2.0, 1e9)
        ]
    )
    assert np.abs(values / values[2] - 1.0).max() < 1e-6


def test_bound_operators_are_reciprocal_before_symmetrisation():
    """P_cb and F_bc discretise the same band integral, so the raw P_eff is symmetric.

    What is left is truncation, falling as 1/N^2: 2.1e-3, 6.5e-4, 2.8e-4 at
    N = 16, 24, 32 bands per sphere.
    """
    coarse, fine = asymmetry(16), asymmetry(32)
    assert fine < 1e-3
    assert fine < 0.25 * coarse


def test_high_permittivity_approaches_a_conducting_shell():
    """As lam -> 1 the shell shorts out, leaving the a-to-b gap in series with 4 pi eps0 c."""
    conductor = 4.0 * math.pi * epsilon_0 / (1.0 / INNER - 1.0 / MIDDLE + 1.0 / OUTER)
    values = np.array([coated_capacitance(48, e) for e in (1.0, 2.56, 10.0, 100.0)])
    assert np.all(np.diff(values) > 0.0)
    assert np.all(values < 4.0 * math.pi * epsilon_0 * OUTER)
    assert values[-1] == pytest.approx(conductor, rel=0.01)


def test_former_dielectric_lowers_the_resonant_frequency():
    """Small coil, coarse former: the dielectric path costs nothing to exercise."""
    air = resonance(solenoid_design(sections=10, turns=20, former_sections=12), modes=1)
    loaded = resonance(
        solenoid_design(2.56, sections=10, turns=20, former_sections=12), modes=1
    )
    assert loaded.f[0] < air.f[0]


@pytest.mark.slow
def test_former_lowers_f_res_monotonically_in_permittivity():
    """A polystyrene rod moves f_res down 9 %, the sign and size of the doc's residual."""
    air = resonance(solenoid_design(), modes=1).f[0]
    f = np.array(
        [resonance(solenoid_design(e), modes=1).f[0] for e in (1.0, 1.5, 2.56, 4.0)]
    )
    assert f[0] == air
    assert np.all(np.diff(f) < 0.0)
    assert -0.12 < f[2] / air - 1.0 < -0.06


@pytest.mark.slow
def test_former_f_res_is_converged_in_former_sections():
    f = np.array(
        [
            resonance(solenoid_design(2.56, former_sections=n), modes=1).f[0]
            for n in (60, 200)
        ]
    )
    assert f[1] == pytest.approx(f[0], rel=0.002)


def test_former_discretisation_walks_the_closed_contour():
    former = Former(outer_radius=0.05, length=0.2, base=0.01, inner_radius=0.04)
    boundary = former.discretise(40)
    rings = boundary.rings
    assert len(boundary) == 40
    assert rings.w.sum() == pytest.approx(2.0 * former.length + 2.0 * 0.01)
    assert np.array_equal(np.hypot(boundary.nr, boundary.nz), np.ones(40))
    assert np.array_equal(rings.rw, 0.25 * rings.w)
    walls = boundary.nz == 0.0
    assert np.unique(rings.a[walls]) == pytest.approx([0.04, 0.05])
    assert np.array_equal(rings.a[walls] == 0.05, boundary.nr[walls] > 0.0)
    assert np.unique(rings.z[~walls]) == pytest.approx([0.01, 0.21])
    assert np.array_equal(rings.z[~walls] > 0.1, boundary.nz[~walls] > 0.0)


def test_solid_former_has_no_inner_wall_and_reaches_the_axis():
    boundary = Former(outer_radius=0.05, length=0.2).discretise(30)
    assert np.all(boundary.rings.a > 0.0)
    assert np.all(boundary.nr >= 0.0)
    assert boundary.rings.w.sum() == pytest.approx(0.3)
    assert boundary.area.sum() == pytest.approx(2.0 * math.pi * 0.05 * (0.2 + 0.05))


@pytest.mark.parametrize("sections", [4, 5, 37, 96])
def test_former_allocates_every_piece_at_least_one_band(sections):
    """A 1 mm wall against a 1 m tube: each face keeps a band, at most one over its share."""
    boundary = Former(
        outer_radius=0.05, length=1.0, base=0.0, inner_radius=0.049
    ).discretise(sections)
    faces = set(zip(boundary.nr, boundary.nz))
    assert faces == {(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)}
    assert sections <= len(boundary) <= sections + len(faces)


def test_dielectric_rejects_mismatched_normals():
    rings = Sphere(0.1, 0.0).discretise(8)
    with pytest.raises(ValueError, match="normal components"):
        Dielectric(rings=rings, nr=np.ones(7), nz=np.zeros(7), permittivity=2.0)


def test_susceptibility_spans_vacuum_to_conductor():
    assert Former(0.05, 0.1, permittivity=1.0).discretise(8).susceptibility == 0.0
    assert Former(0.05, 0.1).discretise(8).susceptibility == pytest.approx(1.56 / 3.56)


def test_design_carries_the_former_through_the_schema():
    spec = {
        "secondary": {
            "radius": 0.05,
            "length": 0.1,
            "turns": 100,
            "wire_diameter": 5e-4,
        },
        "primary": {"inner_radius": 0.08, "turns": 4.0, "pitch": 0.01},
        "former": {"outer_radius": 0.0495, "length": 0.1, "permittivity": 3.4},
        "former_sections": 24,
    }
    design = Design.from_dict(spec)
    assert design.former == Former(outer_radius=0.0495, length=0.1, permittivity=3.4)
    assert len(design.dielectric()) == 24
    assert replace(design, former=None).dielectric() is None


@pytest.mark.parametrize("ground", [False, True])
def test_the_bound_field_operator_carries_the_gauss_flux_exactly(ground):
    """Area-weighted column sums of F_bb are 1/(2 eps0), the constraint that sets its diagonal.

    A unit charge sitting on a closed surface sends half its flux through that
    surface; image bands lie outside it and send none, so the identity holds with
    or without the ground plane.
    """
    design = solenoid_design(permittivity=3.0)
    boundary = design.dielectric()
    rings = design.secondary_rings()
    flux = boundary.area @ bound_operators(rings, boundary, ground)[2]
    assert flux == pytest.approx(0.5 / epsilon_0, rel=1e-12)
