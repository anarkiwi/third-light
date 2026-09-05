"""Dielectric regions in the ring-charge method of moments.

A region of relative permittivity eps_r bounded by a closed surface of revolution
is replaced by an equivalent bound surface charge radiating in the same vacuum
Green's function as :mod:`thirdlight.em.capacitance`. With ``n`` the unit normal
out of the dielectric, continuity of normal D across the sheet gives

    sigma_b = 2 eps0 lam E_n,   lam = (eps_r - 1) / (eps_r + 1)

where E_n is the principal-value normal field of all charges, free and bound. The
boundary is discretised into bands, band ``i`` a ring at (a_i, z_i) of meridian
width w_i, area A_i and outward normal (nr_i, nz_i), carrying uniform charge. In
matrix form, with free ring charges ``q``, bound band charges ``b = A sigma_b``
and G = 2 eps0 lam,

    v = P_cc q + P_cb b,   b = G A (F_bc q + F_bb b)
    P_eff = P_cc + P_cb (I - G A F_bb)^-1 G A F_bc

so the dielectric enters as a dense correction to the potential coefficient
matrix and nothing downstream of P changes.
"""

import math

import numba
import numpy as np
from scipy.constants import epsilon_0

from thirdlight.backend import array_namespace, asnumpy, kernel
from thirdlight.em.capacitance import potential_ring
from thirdlight.em.elliptic import ellipk, ellipke

_TWO_PI2_EPS0 = 2.0 * math.pi * math.pi * epsilon_0
_SERIES_M = 0.2
_SERIES_TERMS = 40
_SERIES_TOL = 1e-18

_QUAD_ORDER = 6
_QUAD_TOL = 1e-10
_QUAD = tuple(zip(*np.polynomial.legendre.leggauss(_QUAD_ORDER)))

# Gauss-Legendre with n nodes integrates a kernel whose singularity sits at
# scaled distance x from the centre of the interval with relative error rho^-2n,
# rho = x + sqrt(x^2 - 1) the Bernstein parameter. Inverting rho^-2n = _QUAD_TOL
# gives the scaled distance at which the rule has converged, so a source band is
# a filament once its centre is that many observer half-widths away and is
# subdivided by the same rule when it is nearer.
_SPLIT = 0.5 * (
    _QUAD_TOL ** (-1.0 / (2 * _QUAD_ORDER)) + _QUAD_TOL ** (1.0 / (2 * _QUAD_ORDER))
)


@kernel
def dellipk(m):
    """dK/dm for 0 <= m < 1.

    (E - (1 - m) K) / (2 m (1 - m)) is a difference of O(1) terms whose value is
    O(m), so it loses m in relative precision as m -> 0, exactly as the
    inductance bracket does. Below ``_SERIES_M`` the term-by-term derivative of
    K = (pi/2) sum c_k m^k, c_k = c_{k-1} ((2k-1)/(2k))^2, gives the same
    quantity as a sum of positive terms, so nothing cancels at any m.
    """
    if m < _SERIES_M:
        c = 0.25
        p = 1.0
        s = 0.0
        for k in range(1, _SERIES_TERMS):
            t = k * c * p
            s += t
            if t < _SERIES_TOL * s:
                break
            r = (2 * k + 1) / (2 * k + 2)
            c *= r * r
            p *= m
        return 0.5 * math.pi * s
    ek, ee = ellipke(m)
    return (ee - (1.0 - m) * ek) / (2.0 * m * (1.0 - m))


@kernel
def field_ring(r, z_field, a, z_source):
    """Return (E_r, E_z) at (r, z_field) per unit charge on a coaxial ring (a, z_source).

    The gradient of the ring potential K(m) / (2 pi^2 eps0 u), u = sqrt(S),
    S = (r + a)^2 + dz^2, m = 4 r a / S, with dm/dr = 4a/S - 2 m (r + a)/S and
    dm/dz = -2 m dz / S. On the axis it reduces to the point-charge field of the
    whole ring charge at the slant distance.
    """
    dz = z_field - z_source
    s = (r + a) * (r + a) + dz * dz
    u = math.sqrt(s)
    m = 4.0 * r * a / s
    ek = ellipk(m)
    kp = dellipk(m) / s
    scale = -1.0 / (_TWO_PI2_EPS0 * u)
    e_r = scale * (kp * (4.0 * a - 2.0 * m * (r + a)) - ek * (r + a) / s)
    e_z = scale * (-2.0 * kp * m * dz - ek * dz / s)
    return e_r, e_z


@kernel
def grading(eta, k):
    """Sinh substitution coefficients for a band integrated against a nearby point.

    A point at tangential offset ``eta`` and perpendicular distance ``k`` from a
    band centre, both in band half-widths, puts the kernel's singularity at
    eta +- i k. Then xi = eta + k sinh(mu u - nu) maps u in [-1, 1] onto
    xi in [-1, 1] and clusters the nodes into the peak (Johnston and Elliott),
    degenerating to xi = u as k grows. At k = 0 the singularity is real and lies
    outside the band, where the plain rule already converges and the substitution
    is undefined; that case returns (0, 0), the plain rule.
    """
    if k <= 0.0:
        return 0.0, 0.0
    hi = math.asinh((1.0 + eta) / k)
    lo = math.asinh((1.0 - eta) / k)
    return 0.5 * (hi + lo), 0.5 * (hi - lo)


@kernel
def graded_node(u, eta, k, mu, nu):
    """Position and Jacobian of Gauss node ``u`` under the :func:`grading` substitution."""
    if mu == 0.0:
        return u, 1.0
    return eta + k * math.sinh(mu * u - nu), k * mu * math.cosh(mu * u - nu)


@kernel
def _band_frame(dr, dz, half, nr, nz):
    """Tangential offset and perpendicular distance of a point, in band half-widths.

    ``dr, dz`` is the point relative to the band centre and the band's meridian
    tangent is (-nz, nr).
    """
    return (dz * nr - dr * nz) / half, abs(dr * nr + dz * nz) / half


@kernel
def band_potential(r, z_field, src):
    """Potential at a point per unit charge spread uniformly over a source band.

    The band is integrated by the same graded rule that averages the field
    operator over it, which is what makes the two operators reciprocal.
    """
    a_src, z_src, w_src, nr_src, nz_src = src
    half = 0.5 * w_src
    eta, k = _band_frame(r - a_src, z_field - z_src, half, nr_src, nz_src)
    mu, nu = grading(eta, k)
    total = 0.0
    for u, weight in _QUAD:
        xi, jacobian = graded_node(u, eta, k, mu, nu)
        h = half * xi
        total += (
            weight
            * jacobian
            * potential_ring(r, z_field, a_src - h * nz_src, z_src + h * nr_src)
        )
    return 0.5 * total


@kernel
def band_field(band, a_src, z_src):
    """Normal field of a filament ring, averaged over an observer band.

    ``n . E`` is averaged along the band's meridian tangent (-nz, nr) by the same
    graded rule, because a conductor ring lies a wire radius off the former
    surface, where a centre-sampled kernel carries no information about the band.
    """
    a, z, w, nr, nz = band
    half = 0.5 * w
    eta, k = _band_frame(a_src - a, z_src - z, half, nr, nz)
    mu, nu = grading(eta, k)
    total = 0.0
    for u, weight in _QUAD:
        xi, jacobian = graded_node(u, eta, k, mu, nu)
        h = half * xi
        e_r, e_z = field_ring(a - h * nz, z + h * nr, a_src, z_src)
        total += weight * jacobian * (nr * e_r + nz * e_z)
    return 0.5 * total


@kernel
def band_pair_field(band, src, split):
    """Band-averaged normal field of a source band carrying uniform charge.

    The source is a filament once its centre clears the observer band centre by
    ``split`` observer half-widths, and is otherwise subdivided by the graded
    rule about that centre, each sub-ring carrying its weight fraction.
    """
    a_src, z_src, w_src, nr_src, nz_src = src
    dr, dz = a_src - band[0], z_src - band[1]
    if math.hypot(dr, dz) >= 0.5 * split * band[2]:
        return band_field(band, a_src, z_src)
    half = 0.5 * w_src
    eta, k = _band_frame(-dr, -dz, half, nr_src, nz_src)
    mu, nu = grading(eta, k)
    total = 0.0
    for u, weight in _QUAD:
        xi, jacobian = graded_node(u, eta, k, mu, nu)
        h = half * xi
        total += (
            weight * jacobian * band_field(band, a_src - h * nz_src, z_src + h * nr_src)
        )
    return 0.5 * total


@kernel
def _band(f, j, mirror):
    """Column ``j`` of stacked band rows as an (a, z, w, nr, nz) tuple.

    ``mirror`` reflects the band in the ground plane, which flips both its height
    and its meridian tangent, so its normal enters as (-nr, nz).
    """
    if mirror:
        return f[0, j], -f[1, j], f[2, j], -f[3, j], f[4, j]
    return f[0, j], f[1, j], f[2, j], f[3, j], f[4, j]


@numba.njit(cache=True, parallel=True)
def _conductor_bound(c, b, ground):
    """P_cb: potential at conductor rings due to unit bound band charges."""
    out = np.empty((c.shape[1], b.shape[1]))
    for i in numba.prange(c.shape[1]):  # pylint: disable=not-an-iterable
        for j in range(b.shape[1]):
            v = band_potential(c[0, i], c[1, i], _band(b, j, False))
            if ground:
                v -= band_potential(c[0, i], c[1, i], _band(b, j, True))
            out[i, j] = v
    return out


@numba.njit(cache=True, parallel=True)
def _bound_conductor(b, c, ground):
    """F_bc: band-averaged normal field at the surface due to conductor ring charges."""
    out = np.empty((b.shape[1], c.shape[1]))
    for i in numba.prange(b.shape[1]):  # pylint: disable=not-an-iterable
        band = _band(b, i, False)
        for j in range(c.shape[1]):
            v = band_field(band, c[0, j], c[1, j])
            if ground:
                v -= band_field(band, c[0, j], -c[1, j])
            out[i, j] = v
    return out


@numba.njit(cache=True, parallel=True)
def _bound_bound(b, area, ground, split):
    """F_bb: band-averaged principal-value normal field due to the bound charges.

    The flat-band self field vanishes at the band's own centre by symmetry, so
    what the diagonal carries is the band's curvature, and Gauss's law fixes that
    exactly: the principal-value flux of a unit charge sitting on a closed
    surface through that surface is 1 / (2 eps0), so the area-weighted column
    sums are known and the diagonal is whatever the off-diagonal terms leave
    over; the diagonal's own image term is already in place and is kept. Dropping
    the diagonal instead loses an order of convergence. Image bands lie outside
    the closed surface and so add no flux of their own.
    """
    count = b.shape[1]
    out = np.zeros((count, count))
    for i in numba.prange(count):  # pylint: disable=not-an-iterable
        band = _band(b, i, False)
        for j in range(count):
            v = 0.0
            if i != j:
                v = band_pair_field(band, _band(b, j, False), split)
            if ground:
                v -= band_pair_field(band, _band(b, j, True), split)
            out[i, j] = v
    column = np.dot(area, out)
    for j in range(count):
        out[j, j] += (0.5 / epsilon_0 - column[j]) / area[j]
    return out


def _boundary_fields(boundary):
    """Stack a dielectric boundary as contiguous float64 (a, z, w, nr, nz) rows."""
    rings = boundary.rings
    return np.ascontiguousarray(
        np.stack([rings.a, rings.z, rings.w, boundary.nr, boundary.nz]),
        dtype=np.float64,
    )


def _conductor_fields(rings):
    """Stack a ring set as contiguous float64 (a, z) rows."""
    return np.ascontiguousarray(np.stack([rings.a, rings.z]), dtype=np.float64)


def bound_operators(rings, boundary, ground_plane=True, split=_SPLIT):
    """Return (P_cb, F_bc, F_bb) for a conductor ring set and a dielectric boundary.

    Bound charges image in the ground plane exactly as free charges do: an image
    band of opposite sign at -z, subtracted, in all three operators.
    """
    b = _boundary_fields(boundary)
    c = _conductor_fields(rings)
    area = np.ascontiguousarray(boundary.area, dtype=np.float64)
    return (
        _conductor_bound(c, b, ground_plane),
        _bound_conductor(b, c, ground_plane),
        _bound_bound(b, area, ground_plane, split),
    )


def polarised_potential(p, rings, boundary, ground_plane=True, split=_SPLIT):
    """Potential coefficient matrix ``p`` corrected for a dielectric boundary.

    Reciprocity makes the result symmetric in exact arithmetic, but the
    discretisation is symmetric only to its own truncation error, so the returned
    matrix is symmetrised; the Cholesky downstream requires it.
    """
    xp = array_namespace()
    p_cb, f_bc, f_bb = (
        xp.asarray(m) for m in bound_operators(rings, boundary, ground_plane, split)
    )
    ga = xp.asarray(2.0 * epsilon_0 * boundary.susceptibility * boundary.area)[:, None]
    correction = p_cb @ xp.linalg.solve(xp.eye(len(boundary)) - ga * f_bb, ga * f_bc)
    effective = xp.asarray(p) + correction
    return asnumpy(0.5 * (effective + effective.T))
