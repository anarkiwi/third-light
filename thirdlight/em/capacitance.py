"""Ring-charge method of moments: potential coefficients and Maxwell capacitance.

Electrostatic dual of :mod:`thirdlight.em.inductance`. Ring ``i`` carries total
charge ``q_i`` spread around a coaxial ring of radius ``a_i`` at height ``z_i``;
``P q = v`` gives the potentials and ``C = P^-1`` the capacitance matrix.
"""

import math
from functools import lru_cache

import numba
import numpy as np
from scipy.constants import epsilon_0

from thirdlight.backend import array_namespace, asnumpy, kernel
from thirdlight.em.elliptic import ellipk
from thirdlight.geometry import Rings, Sphere

_FOUR_PI2_EPS0 = 4.0 * math.pi * math.pi * epsilon_0


@kernel
def potential_ring(r, z_field, a, z_source):
    """Potential at (r, z_field) per unit charge on a coaxial ring (a, z_source).

    phi = K(m) / (2 pi^2 eps0 sqrt((r + a)^2 + dz^2)), m = 4 r a / ((r + a)^2 + dz^2).
    K is positive and monotone, so unlike the inductance bracket nothing cancels
    as m -> 0; K(0) = pi/2 recovers the point charge 1 / (4 pi eps0 R).
    """
    dz = z_field - z_source
    s = (r + a) * (r + a) + dz * dz
    return 2.0 * ellipk(4.0 * r * a / s) / (_FOUR_PI2_EPS0 * math.sqrt(s))


@kernel
def self_potential(a, rw):
    """Singular diagonal: thin torus of capacitance 4 pi^2 eps0 a / ln(8 a / rw)."""
    return math.log(8.0 * a / rw) / (_FOUR_PI2_EPS0 * a)


@numba.njit(cache=True, parallel=True)
def _assemble(f, zi, ground):
    """Symmetric potential coefficient matrix from stacked (a, z, rw) rows.

    ``zi`` holds the image heights; with ``ground`` the image ring of opposite
    charge is subtracted, enforcing zero potential on the z = 0 plane.
    """
    a, z, rw = f[0], f[1], f[2]
    count = a.shape[0]
    out = np.empty((count, count))
    for i in numba.prange(count):  # pylint: disable=not-an-iterable
        v = self_potential(a[i], rw[i])
        if ground:
            v -= potential_ring(a[i], z[i], a[i], zi[i])
        out[i, i] = v
        for j in range(i + 1, count):
            v = potential_ring(a[i], z[i], a[j], z[j])
            if ground:
                v -= potential_ring(a[i], z[i], a[j], zi[j])
            out[i, j] = v
            out[j, i] = v
    return out


def _fields(rings):
    """Stack a ring set as contiguous float64 (a, z, rw) rows."""
    return np.ascontiguousarray(
        np.stack([rings.a, rings.z, rings.rw]), dtype=np.float64
    )


def _cho_solve(p, b):
    """Solve ``P x = b`` for symmetric positive definite ``P`` by Cholesky."""
    xp = array_namespace()
    L = xp.linalg.cholesky(xp.asarray(p))
    return xp.linalg.solve(L.T, xp.linalg.solve(L, xp.asarray(b)))


def potential_matrix(rings, ground_plane=True, dielectric=None):
    """Dense symmetric potential coefficient matrix of a ring set, in V/C.

    Off-diagonal terms are the ring-to-ring potential; the diagonal is the thin
    torus self term at the ring's equivalent conductor radius. A
    :class:`~thirdlight.geometry.Dielectric` boundary adds the equivalent
    bound-charge correction of :mod:`thirdlight.em.dielectric`.
    """
    image = rings.mirrored() if ground_plane else rings
    p = _assemble(
        _fields(rings), np.ascontiguousarray(image.z, dtype=np.float64), ground_plane
    )
    if dielectric is None:
        return p
    # Imported here because the dielectric operators are built on this module's kernels.
    from thirdlight.em.dielectric import (  # pylint: disable=import-outside-toplevel,cyclic-import
        polarised_potential,
    )

    return polarised_potential(p, rings, dielectric, ground_plane)


def capacitance_matrix(rings, ground_plane=True, dielectric=None):
    """Maxwell capacitance matrix C = P^-1, by Cholesky in float64."""
    xp = array_namespace()
    p = potential_matrix(rings, ground_plane, dielectric)
    return _cho_solve(p, xp.eye(p.shape[0]))


def reduce_sections(matrix, groups):
    """Block-mean ``matrix`` over contiguous ascending ``groups`` labels.

    A section's charge spreads over its rings rather than being carried by all of
    them, so the section coefficient is the block mean, not the block sum.
    """
    groups = np.asarray(groups)
    if groups.shape != matrix.shape[:1]:
        raise ValueError(f"groups length {groups.shape} does not index {matrix.shape}")
    starts = np.flatnonzero(np.concatenate(([True], np.diff(groups) != 0)))
    if not np.array_equal(groups[starts], np.arange(starts.size)):
        raise ValueError("groups must label contiguous ascending blocks from 0")
    counts = np.diff(np.append(starts, groups.size))
    block = np.add.reduceat(np.add.reduceat(matrix, starts, axis=0), starts, axis=1)
    return block / np.outer(counts, counts)


def surface_field(rings, charges):
    """Normal field at each ring conductor surface, in V/m.

    Gauss at a conductor: E = sigma / eps0 with sigma = q / (2 pi a w), the ring's
    own band area, which is exact for the sphere and toroid discretisations. No
    image term enters; the extracted charge already carries the whole field.
    """
    return np.asarray(charges) / (epsilon_0 * rings.area)


def unit_potential_charges(rings, ground_plane=True, dielectric=None):
    """Ring charges with every ring held at unit potential, in C/V."""
    xp = array_namespace()
    p = potential_matrix(rings, ground_plane, dielectric)
    return asnumpy(_cho_solve(p, xp.ones(p.shape[0])))


@lru_cache(maxsize=None)
def sphere_field_correction(sections):
    """Exact-to-model surface field ratio of each ring of a discretised sphere.

    An isolated sphere carries a uniform surface field, so its own solve
    calibrates the ring model's local error. The polar cap is a disc rather than
    a ring -- its radius is half its width at every ``sections`` -- and comes out
    10.6 % low, a discretisation error that does not converge because refining
    the sphere only makes a smaller cap of the same shape. The ratio is scale
    free, and it is local rather than a property of the isolated sphere: applied
    to a breakout point mounted over a coil it recovers the refined solution's
    pole field to 0.02 %.
    """
    rings = Sphere(1.0, 0.0).discretise(sections)
    charges = unit_potential_charges(rings, ground_plane=False)
    ratio = charges.sum() / (4.0 * math.pi * epsilon_0 * surface_field(rings, charges))
    ratio.setflags(write=False)
    return ratio


def field_correction(part, sections):
    """Surface field correction of a discretised component, ones where it needs none.

    Only the sphere has cap-like rings; a toroid's bands are all slender, and a
    ring model resolves them to the convergence of the potential solve itself.
    """
    if isinstance(part, Sphere):
        return sphere_field_correction(sections)
    return np.ones(sections)


def lumped_capacitance(rings, ground_plane=True, dielectric=None):
    """Total charge on a ring set held at unit potential: its lumped capacitance."""
    return float(unit_potential_charges(rings, ground_plane, dielectric).sum())


def resonator_capacitance(design):
    """Lumped capacitance at the base of a grounded secondary with its top load."""
    rings = Rings.concat(design.secondary_rings(), design.top_load_rings())
    return lumped_capacitance(rings, design.ground_plane, design.dielectric())
