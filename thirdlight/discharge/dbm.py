"""Niemeyer-Pietronero-Wiesmann dielectric breakdown growth of the discharge tree [18], [19].

Candidates one step off each node compete on the mean field (V - phi)/h they
would cross, one is taken with probability proportional to field^eta, and a
candidate below the critical field terminates in place of design.md 3.4a's E l.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.constants import epsilon_0
from scipy.linalg import solve_triangular

from thirdlight.discharge.filament import (
    Tree,
    _assemble_mixed,
    self_potential_segment,
)
from thirdlight.em.capacitance import potential_matrix
from thirdlight.geometry import Sphere, Toroid

_FOUR_PI_EPS0 = 4.0 * math.pi * epsilon_0
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
_MIRROR = np.array([1.0, 1.0, -1.0])


def _filament(field, start, end, length):
    """Uniform line charge seen from a set of field points, vectorised over both."""
    r1 = np.linalg.norm(field[:, None, :] - start[None], axis=-1)
    r2 = np.linalg.norm(field[:, None, :] - end[None], axis=-1)
    total = r1 + r2
    return np.log((total + length) / (total - length)) / (_FOUR_PI_EPS0 * length)


def _line_potential(field, start, end, length, ground):
    """Potential at each field point per unit segment charge, as (points, segments)."""
    value = _filament(field, start, end, length)
    if ground:
        value -= _filament(field, start * _MIRROR, end * _MIRROR, length)
    return value


def cap_directions(count, cone, plane=False):
    """Quasi-uniform unit vectors in a forward cone of half-angle ``cone`` about +z.

    The cap carries a Fibonacci spiral, the compact deterministic quasi-uniform
    set; the planar restriction is the arc of the same cap. Both include the
    axis itself, so a channel under a dominant field goes straight on.
    """
    offset = np.arange(count) / (count - 1)
    if plane:
        angle = cone * (2.0 * offset - 1.0)
        return np.stack([np.sin(angle), np.zeros(count), np.cos(angle)], axis=-1)
    z = 1.0 - (1.0 - math.cos(cone)) * offset
    phi = _GOLDEN_ANGLE * np.arange(count)
    r = np.sqrt(1.0 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


def _rotation(direction, plane):
    """Rotation carrying +z onto ``direction``.

    In the plane it is the rotation about y, which leaves the x-z plane
    invariant where a general frame would not; in space it is Duff's branchless
    orthonormal basis, which has no pole at either end of the axis.
    """
    x, y, z = direction
    if plane:
        return np.array([[z, 0.0, x], [0.0, 1.0, 0.0], [-x, 0.0, z]])
    sign = math.copysign(1.0, z)
    a = -1.0 / (sign + z)
    b = x * y * a
    return np.array(
        [
            [1.0 + sign * x * x * a, b, x],
            [sign * b, sign + y * y * a, y],
            [-sign * x, -y, z],
        ]
    )


def interior(bodies, points):
    """Mask of the points inside any of ``bodies``, exact for the top-load shapes."""
    inside = np.zeros(points.shape[0], dtype=bool)
    radial = np.hypot(points[:, 0], points[:, 1])
    for body in bodies:
        if isinstance(body, Sphere):
            gap = np.hypot(radial, points[:, 2] - body.height) - body.radius
        elif isinstance(body, Toroid):
            gap = (
                np.hypot(radial - body.major_radius, points[:, 2] - body.height)
                - body.minor_radius
            )
        else:
            raise TypeError(f"no interior test for {type(body).__name__}")
        inside |= gap < 0.0
    return inside


class BorderedCholesky:
    """Lower Cholesky factor of a symmetric positive definite matrix under append.

    Bordering ``[[P, b], [b.T, d]]`` leaves the old factor untouched, so the new
    row is ``L c = b`` by one triangular solve and ``sqrt(d - c.c)``: O(n^2) and
    BLAS-2, against O(n^3) to refactorise. The buffer is sized for the growth.
    """

    def __init__(self, matrix, capacity):
        size = matrix.shape[0]
        self._buffer = np.zeros((capacity, capacity))
        self._buffer[:size, :size] = np.linalg.cholesky(matrix)
        self._size = size

    def __len__(self):
        return self._size

    @property
    def factor(self):
        """The maintained lower-triangular factor, as an (n, n) view."""
        return self._buffer[: self._size, : self._size]

    def append(self, border, diagonal):
        """Extend by the row and column ``[border, diagonal]``; False if that is indefinite.

        A rejected border leaves the factor untouched, the trial row lying in
        the unused part of the buffer.
        """
        row = self._buffer[self._size, : self._size]
        row[:] = solve_triangular(self.factor, border, lower=True)
        pivot = diagonal - row @ row
        if pivot <= 0.0:
            return False
        self._buffer[self._size, self._size] = math.sqrt(pivot)
        self._size += 1
        return True

    def truncate(self, size):
        """Drop the trailing rows and columns beyond ``size``; the prefix is its own factor."""
        self._size = min(self._size, size)

    def solve(self, b):
        """Solve ``P x = b`` through the maintained factor."""
        factor = self.factor
        return solve_triangular(
            factor.T, solve_triangular(factor, b, lower=True), lower=False
        )


@dataclass(frozen=True)
class Growth:
    """Parameters of the breakdown growth rule.

    ``critical`` is a field per unit channel potential, the electrostatic
    problem being solved at unit potential, so the propagation field in V/m is
    it scaled by the channel voltage.
    """

    step: float
    radius: float
    eta: float = 1.0
    critical: float = 0.0
    directions: int = 16
    cone: float = 0.5 * math.pi
    plane: bool = False


class Discharge:  # pylint: disable=too-many-instance-attributes
    """One growing tree, its bordered factor and its live candidate pool.

    A step adds one column of candidate-to-source coefficients for the new
    segment and one block of rows for the new node's own candidates, so no
    coefficient is ever recomputed and the pool is never rebuilt.
    """

    def __init__(
        self, rings, seed, direction, growth, steps, rng, bodies=(), ground=True
    ):
        self.growth = growth
        self.rings = rings
        self.bodies = tuple(bodies)
        self.ground = ground
        self.rng = rng
        self._capacity = steps
        self._nodes = np.empty((steps + 1, 3))
        self._nodes[0] = seed
        self._parent = np.full(steps + 1, -1)
        self._count = 1
        self._start = np.empty((steps, 3))
        self._end = np.empty((steps, 3))
        self._mid = np.empty((steps, 3))
        self._segments = 0
        self._cone = cap_directions(growth.directions, growth.cone, growth.plane)
        self._factor = BorderedCholesky(
            potential_matrix(rings, ground), len(rings) + steps
        )
        self._sites = np.empty((0, 3))
        self._roots = np.empty(0, dtype=int)
        self._coefficients = np.empty((0, len(rings)))
        self._offer(0, np.asarray(direction, dtype=float))

    def __len__(self):
        return self._count

    @property
    def tree(self):
        """The tree grown so far."""
        return Tree(
            self._nodes[: self._count].copy(),
            self._parent[: self._count].copy(),
            self.growth.radius,
        )

    @property
    def factor(self):
        """The incrementally maintained Cholesky factor of the mixed system."""
        return self._factor.factor

    @property
    def nodes(self):
        """Node positions grown so far, as an (n, 3) view."""
        return self._nodes[: self._count]

    @property
    def sites(self):
        """Live candidate positions."""
        return self._sites

    @property
    def roots(self):
        """Node each live candidate hangs off."""
        return self._roots

    @property
    def _length(self):
        """Lengths of the segments built so far, every one a growth step."""
        return np.full(self._segments, self.growth.step)

    def _rings(self, points):
        """Ring-to-point potential coefficients, in the dense assembly's own order."""
        return _assemble_mixed(
            np.ascontiguousarray(self.rings.a, dtype=np.float64),
            np.ascontiguousarray(self.rings.z, dtype=np.float64),
            np.ascontiguousarray(points),
            self.ground,
        )

    def _offer(self, node, direction):
        """Add the admissible candidates one step off ``node`` along its forward cone."""
        rotation = _rotation(direction, self.growth.plane)
        sites = self._nodes[node] + self.growth.step * (self._cone @ rotation.T)
        keep = sites[:, 2] > (2.0 * self.growth.radius if self.ground else 0.0)
        if self.bodies:
            keep &= ~interior(self.bodies, sites)
        gap = np.linalg.norm(
            sites[:, None, :] - self._nodes[None, : self._count], axis=-1
        )
        gap[:, node] = np.inf
        sites = sites[keep & (np.min(gap, axis=1) >= self.growth.step)]
        rows = np.empty((sites.shape[0], len(self.rings) + self._segments))
        rows[:, : len(self.rings)] = self._rings(sites).T
        if self._segments:
            rows[:, len(self.rings) :] = _line_potential(
                sites,
                self._start[: self._segments],
                self._end[: self._segments],
                self._length,
                self.ground,
            )
        self._sites = np.concatenate([self._sites, sites])
        self._roots = np.concatenate([self._roots, np.full(sites.shape[0], node)])
        self._coefficients = np.concatenate([self._coefficients, rows])

    def _keep(self, mask):
        """Restrict the candidate pool to ``mask``."""
        self._sites = self._sites[mask]
        self._roots = self._roots[mask]
        self._coefficients = self._coefficients[mask]

    def charges(self, source=None):
        """Source charges at the potentials ``source``, unit potential everywhere by default."""
        if source is None:
            source = np.ones(len(self._factor))
        return self._factor.solve(source)

    def potentials(self, charges):
        """Potential at every live candidate from a source charge vector."""
        return self._coefficients @ charges

    def fields(self):
        """Mean field each live candidate would be crossed at, per unit channel potential."""
        return (1.0 - self.potentials(self.charges())) / self.growth.step

    def prune(self, count):
        """Drop the node suffix beyond ``count``, keeping every parent before its child.

        Growth appends, so the survivors are a prefix of the factor as well as of
        the tree: the factor truncates, the pool loses the candidates its own
        node no longer exists at and the columns of the dropped segments, and
        nothing is refactorised or recomputed.
        """
        count = min(max(count, 1), self._count)
        if count == self._count:
            return
        self._count, self._segments = count, count - 1
        self._factor.truncate(len(self.rings) + self._segments)
        self._keep(self._roots < count)
        self._coefficients = self._coefficients[:, : len(self.rings) + self._segments]

    def step(self, field=None, critical=None):
        """Grow one segment; return False when nothing is left above the critical field.

        A candidate whose channel would interpenetrate another leaves the mixed
        system indefinite, which is the exact statement that the model cannot
        carry it, so it is dropped and another drawn. ``field`` and ``critical``
        override the unit-potential field and :attr:`Growth.critical` where the
        channel is not an equipotential, in whatever units the caller works in.
        """
        if self._segments >= self._capacity:
            return False
        field = self.fields() if field is None else field
        critical = self.growth.critical if critical is None else critical
        live = np.flatnonzero(field >= critical)
        while live.size:
            weight = (field[live] / field[live].max()) ** self.growth.eta
            chosen = live[self.rng.choice(live.size, p=weight / weight.sum())]
            if self._extend(self._roots[chosen], self._sites[chosen]):
                return True
            self._keep(np.arange(field.size) != chosen)
            field = np.delete(field, chosen)
            live = np.flatnonzero(field >= critical)
        return False

    def _extend(self, parent, site):
        """Append the node, border the factor, and refresh the pool around it."""
        index = self._segments
        self._start[index] = self._nodes[parent]
        self._end[index] = site
        self._mid[index] = 0.5 * (self._nodes[parent] + site)
        if not self._border(index):
            return False
        node = self._count
        self._nodes[node] = site
        self._parent[node] = parent
        self._count += 1
        self._segments += 1
        self._keep(np.linalg.norm(self._sites - site, axis=1) >= self.growth.step)
        column = _line_potential(
            self._sites,
            self._start[index : index + 1],
            self._end[index : index + 1],
            np.full(1, self.growth.step),
            self.ground,
        )
        self._coefficients = np.concatenate([self._coefficients, column], axis=1)
        self._offer(node, (site - self._nodes[parent]) / self.growth.step)
        return True

    def _border(self, index):
        """Row and column of the mixed matrix for segment ``index``, in assembly order.

        Point matching is not reciprocal, so every entry is taken with the field
        at the lower-indexed midpoint, exactly as the dense assembly takes it.
        """
        mid = self._mid[index : index + 1]
        start, end = self._start[index : index + 1], self._end[index : index + 1]
        one = np.full(1, self.growth.step)
        border = np.empty(len(self.rings) + index)
        border[: len(self.rings)] = self._rings(mid).ravel()
        if index:
            border[len(self.rings) :] = _line_potential(
                self._mid[:index], start, end, one, self.ground
            ).ravel()
        diagonal = self_potential_segment(self.growth.step, self.growth.radius)
        if self.ground:
            diagonal -= _filament(mid, start * _MIRROR, end * _MIRROR, one)[0, 0]
        return self._factor.append(border, diagonal)


def grow(rings, seed, direction, growth, steps, rng, bodies=(), ground=True):
    """Grow a discharge tree of at most ``steps`` segments off an electrode.

    ``seed`` is the root node, on the electrode surface, and ``direction`` the
    direction it first offers candidates along. Growth stops early when no
    candidate reaches the critical propagation field.
    """
    discharge = Discharge(rings, seed, direction, growth, steps, rng, bodies, ground)
    for _ in range(steps):
        if not discharge.step():
            break
    return discharge.tree


def gyration(nodes):
    """Radius of gyration of every leading prefix of ``nodes``, in O(n)."""
    count = np.arange(1, nodes.shape[0] + 1)[:, None]
    mean = np.cumsum(nodes, axis=0) / count
    square = np.cumsum(np.sum(nodes * nodes, axis=1)) / count.ravel()
    return np.sqrt(np.maximum(square - np.sum(mean * mean, axis=1), 0.0))


def fractal_dimension(nodes, first=None):
    """Dimension from the R_g ~ N^(1/D) scaling over one cluster's own growth.

    The fit runs from ``first`` nodes, a tenth of the cluster by default, to the
    whole of it, which drops the needle-like transient before branching sets in.
    """
    radius = gyration(nodes)
    first = max(first or nodes.shape[0] // 10, 2)
    count = np.arange(1, nodes.shape[0] + 1)[first:]
    slope = np.polyfit(np.log(count), np.log(radius[first:]), 1)[0]
    return 1.0 / slope
