"""Thin-wire method of moments over straight filament segments, mixed with the ring set.

Three-dimensional dual of :mod:`thirdlight.em.capacitance`. A discharge tree is
a set of straight segments each carrying its total charge spread uniformly along
it; ``P q = v`` over the rings and the segments together gives the charge the
tree adds to an electrode, and the segments' own subtree charges reduce the
tree's per-segment resistance to one series equivalent.
"""

import math
from dataclasses import dataclass

import numba
import numpy as np
from scipy.constants import epsilon_0

from thirdlight.backend import array_namespace, asnumpy, kernel
from thirdlight.em.capacitance import (
    _cho_solve,
    lumped_capacitance,
    potential_matrix,
    potential_ring,
)

_FOUR_PI_EPS0 = 4.0 * math.pi * epsilon_0


@kernel
def filament_potential(r1, r2, L):
    """Uniform line charge of length ``L`` seen at distances ``r1``, ``r2`` from its ends.

    phi = ln((r1 + r2 + L) / (r1 + r2 - L)) / (4 pi eps0 L) per unit total charge.
    The symmetric form is the exact closed form and stays conditioned where the
    projection form differences two nearly equal logarithms.
    """
    s = r1 + r2
    return math.log((s + L) / (s - L)) / (_FOUR_PI_EPS0 * L)


@kernel
def distance(px, py, pz, qx, qy, qz):
    """Euclidean distance between two points."""
    return math.sqrt((px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2)


@kernel
def potential_segment(px, py, pz, ax, ay, az, bx, by, bz):
    """Potential at (px, py, pz) per unit total charge on the segment a to b."""
    return filament_potential(
        distance(px, py, pz, ax, ay, az),
        distance(px, py, pz, bx, by, bz),
        distance(ax, ay, az, bx, by, bz),
    )


@kernel
def self_potential_segment(L, rw):
    """Singular diagonal: the wire surface at the midpoint, r1 = r2 = sqrt(L^2/4 + rw^2).

    Rationalising the difference in the denominator gives the equivalent
    2 ln((sqrt(L^2 + 4 rw^2) + L) / (2 rw)) / (4 pi eps0 L), which holds full
    precision at any aspect ratio and tends to 2 ln(L / rw) as the wire thins.
    """
    return (
        2.0
        * math.log(0.5 * (math.sqrt(L * L + 4.0 * rw * rw) + L) / rw)
        / (_FOUR_PI_EPS0 * L)
    )


@numba.njit(cache=True, parallel=True)
def _assemble(start, end, mid, length, rw, ground):
    """Symmetric segment-segment potential coefficient matrix, point matched at midpoints.

    Segment lengths are carried in rather than recovered per pair. With ``ground``
    the segment mirrored in z = 0 carries the opposite charge, as the image rings
    do. The Green's function is reciprocal, so one triangle is evaluated.
    """
    count = length.shape[0]
    out = np.empty((count, count))
    for i in numba.prange(count):  # pylint: disable=not-an-iterable
        x, y, z = mid[i, 0], mid[i, 1], mid[i, 2]
        for j in range(i, count):
            if j == i:
                v = self_potential_segment(length[i], rw)
            else:
                v = filament_potential(
                    distance(x, y, z, start[j, 0], start[j, 1], start[j, 2]),
                    distance(x, y, z, end[j, 0], end[j, 1], end[j, 2]),
                    length[j],
                )
            if ground:
                v -= filament_potential(
                    distance(x, y, z, start[j, 0], start[j, 1], -start[j, 2]),
                    distance(x, y, z, end[j, 0], end[j, 1], -end[j, 2]),
                    length[j],
                )
            out[i, j] = v
            out[j, i] = v
    return out


@numba.njit(cache=True, parallel=True)
def _assemble_mixed(a, z, mid, ground):
    """Ring-to-segment block: an axisymmetric ring potential at each segment midpoint.

    ``potential_ring`` depends on the field point only through (hypot(x, y), z).
    """
    count = a.shape[0]
    out = np.empty((count, mid.shape[0]))
    for i in numba.prange(count):  # pylint: disable=not-an-iterable
        for j in range(mid.shape[0]):
            r = math.hypot(mid[j, 0], mid[j, 1])
            v = potential_ring(r, mid[j, 2], a[i], z[i])
            if ground:
                v -= potential_ring(r, mid[j, 2], a[i], -z[i])
            out[i, j] = v
    return out


@numba.njit(cache=True)
def _path_sums(parent, element):
    """Forward prefix-add of ``element`` along parents: O(n) root-to-node sums."""
    out = element.copy()
    for k in range(1, out.shape[0]):
        out[k] += out[parent[k]]
    return out


@numba.njit(cache=True)
def _subtree_sums(parent, q):
    """Reverse scatter-add of ``q`` onto parents: O(n) subtree sums, root total in slot 0."""
    out = q.copy()
    for k in range(out.shape[0] - 1, 0, -1):
        out[parent[k]] += out[k]
    return out


@dataclass(frozen=True)
class Tree:
    """Branched discharge channel as node positions and one parent index per node.

    ``nodes`` is (n, 3) metres, ``parent`` is (n,) with ``parent[0] == -1`` for the
    root, and segment k, for k >= 1, joins node k to node ``parent[k]``. Growth
    appends, so every parent precedes its child and a reverse pass over segments
    accumulates subtree quantities. ``radius`` is the channel radius, used both
    for the singular diagonal and for the segment conduction cross section.
    """

    nodes: np.ndarray
    parent: np.ndarray
    radius: float

    def __post_init__(self):
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError(f"nodes must be (n, 3); got {self.nodes.shape}")
        if self.parent.shape != self.nodes.shape[:1]:
            raise ValueError(f"parent length {self.parent.shape} does not index nodes")
        child = np.arange(1, len(self))
        if len(self) and (
            self.parent[0] != -1
            or np.any((self.parent[1:] < 0) | (self.parent[1:] >= child))
        ):
            raise ValueError("parent[0] must be -1 and every other parent precede it")

    def __len__(self):
        return self.nodes.shape[0]

    @property
    def segments(self):
        """Number of segments, one fewer than the node count."""
        return max(len(self) - 1, 0)

    @property
    def endpoints(self):
        """Parent-side and child-side endpoint of each segment, as two (segments, 3) arrays."""
        return self.nodes[self.parent[1:]], np.ascontiguousarray(self.nodes[1:])

    @property
    def lengths(self):
        """Length of each segment."""
        start, end = self.endpoints
        return np.linalg.norm(end - start, axis=1)

    @property
    def midpoints(self):
        """Midpoint of each segment, the point-matching site."""
        start, end = self.endpoints
        return 0.5 * (start + end)


def segment_potential_matrix(start, end, radius, ground_plane=True):
    """Dense symmetric potential coefficient matrix of a segment set, in V/C."""
    start = np.ascontiguousarray(start, dtype=np.float64)
    end = np.ascontiguousarray(end, dtype=np.float64)
    return _assemble(
        start,
        end,
        0.5 * (start + end),
        np.linalg.norm(end - start, axis=1),
        float(radius),
        ground_plane,
    )


def mixed_potential_matrix(rings, tree, ground_plane=True):
    """Potential coefficients of a ring set and a discharge tree, ordered [rings, segments].

    The ring-ring block is :func:`thirdlight.em.capacitance.potential_matrix`, the
    segment-segment block the filament operator, and the off-diagonal blocks are
    transposes of one another by reciprocity of the Green's function.
    """
    start, end = tree.endpoints
    count = len(rings)
    out = np.empty((count + tree.segments,) * 2)
    out[:count, :count] = potential_matrix(rings, ground_plane)
    out[count:, count:] = segment_potential_matrix(
        start, end, tree.radius, ground_plane
    )
    cross = _assemble_mixed(
        np.ascontiguousarray(rings.a, dtype=np.float64),
        np.ascontiguousarray(rings.z, dtype=np.float64),
        np.ascontiguousarray(0.5 * (start + end), dtype=np.float64),
        ground_plane,
    )
    out[:count, count:] = cross
    out[count:, :count] = cross.T
    return out


def mixed_charges(rings, tree, ground_plane=True):
    """Charges with every ring and every segment held at unit potential, in C/V.

    The tree is an equipotential for the electrostatic problem; the resistive
    drop along it is the separate series reduction of :func:`series_resistance`.
    """
    xp = array_namespace()
    p = mixed_potential_matrix(rings, tree, ground_plane)
    return asnumpy(_cho_solve(p, xp.ones(p.shape[0])))


def subtree_charges(tree, charges):
    """Total charge at and below each segment, in segment order, in one reverse pass."""
    padded = np.concatenate(([0.0], np.asarray(charges, dtype=np.float64)))
    return _subtree_sums(np.ascontiguousarray(tree.parent, dtype=np.int64), padded)[1:]


def segment_resistance(tree, resistivity):
    """Ohmic resistance of each segment, rho L / A over the channel cross section."""
    return resistivity * tree.lengths / (math.pi * tree.radius**2)


def path_resistance(tree, resistivity):
    """Root-to-node resistance of every node, in ohm, in one forward pass.

    The mirror of :func:`subtree_charges`: parents precede children, so the
    ancestor sums are a prefix scan over segments and no path is ever walked.
    """
    element = np.concatenate(([0.0], segment_resistance(tree, resistivity)))
    return _path_sums(np.ascontiguousarray(tree.parent, dtype=np.int64), element)


def series_resistance(tree, charges, resistivity):
    """Series resistance dissipating the tree's own charging-current power, in ohm.

    Segment k has R_k = rho L_k / A over the channel cross section, and carries
    the charging current of everything at and below it, so the equivalent is
    sum_k R_k (q_k / Q)^2 with q_k the subtree charge and Q the tree's total.
    """
    share = subtree_charges(tree, charges) / np.sum(charges)
    return float(segment_resistance(tree, resistivity) @ share**2)


def channel_load(rings, tree, resistivity, ground_plane=True):
    """Added capacitance in F and equivalent series resistance in ohm of a tree on an electrode.

    The capacitance is what the tree adds, C_total(rings + tree) - C_total(rings),
    which is the branch a circuit hangs off the top node in place of a length
    times a capacitance per unit length.
    """
    charges = mixed_charges(rings, tree, ground_plane)
    added = float(charges.sum()) - lumped_capacitance(rings, ground_plane)
    return added, series_resistance(tree, charges[len(rings) :], resistivity)
