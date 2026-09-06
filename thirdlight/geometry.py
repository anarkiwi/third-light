"""Coil, primary, top-load and ground descriptions, and their ring discretisation.

Every electromagnetic element reduces to a set of coaxial rings sharing the z
axis. A ring carries ``n`` turns at radius ``a`` and axial centre ``z``, occupies
an axial extent ``w``, and has an equivalent conductor radius ``rw`` used for the
self terms of the inductance and potential-coefficient matrices.

Coordinates are SI, with z measured from the ground plane.
"""

import math
from dataclasses import dataclass

import numpy as np
import yaml


def _apportion(total, weights):
    """Largest-remainder split of ``total`` bands proportional to ``weights``, at least one each."""
    share = total * np.asarray(weights, dtype=float) / np.sum(weights)
    counts = np.maximum(np.floor(share), 1.0).astype(int)
    order = np.argsort(counts - share)[: max(total - counts.sum(), 0)]
    counts[order] += 1
    return counts


@dataclass(frozen=True)
class Rings:
    """Coaxial ring discretisation of one or more components."""

    a: np.ndarray
    z: np.ndarray
    n: np.ndarray
    w: np.ndarray
    rw: np.ndarray

    def __post_init__(self):
        sizes = {len(getattr(self, f)) for f in ("a", "z", "n", "w", "rw")}
        if len(sizes) != 1:
            raise ValueError(f"ring field lengths disagree: {sizes}")

    def __len__(self):
        return len(self.a)

    def __getitem__(self, item):
        """Sub-set of rings, indexed by slice or integer array."""
        return Rings(*(getattr(self, f)[item] for f in ("a", "z", "n", "w", "rw")))

    @property
    def area(self):
        """Band area 2 pi a w of each ring."""
        return 2.0 * np.pi * self.a * self.w

    @classmethod
    def concat(cls, *parts):
        """Concatenate ring sets in order."""
        return cls(
            *(
                np.concatenate([getattr(p, f) for p in parts])
                for f in ("a", "z", "n", "w", "rw")
            )
        )

    def mirrored(self):
        """Image rings in a ground plane at z = 0, with the sign carried by charge."""
        return Rings(self.a, -self.z, self.n, self.w, self.rw)


@dataclass(frozen=True)
class Solenoid:
    """Single-layer helical winding of round wire."""

    radius: float
    length: float
    turns: float
    wire_diameter: float
    base: float = 0.0

    @property
    def pitch(self):
        """Axial distance between adjacent turns."""
        return self.length / self.turns

    @property
    def wire_length(self):
        """Developed conductor length."""
        return self.turns * np.hypot(2.0 * np.pi * self.radius, self.pitch)

    def discretise(self, sections=None):
        """Equal-length, equal-turn ring sections; one ring per turn if ``sections`` is None."""
        count = int(np.ceil(self.turns)) if sections is None else sections
        w = self.length / count
        z = self.base + w * (np.arange(count) + 0.5)
        return Rings(
            a=np.full(count, self.radius),
            z=z,
            n=np.full(count, self.turns / count),
            w=np.full(count, w),
            rw=np.full(count, 0.5 * self.wire_diameter),
        )


@dataclass(frozen=True)
class Toroid:
    """Toroidal top load, discretised as rings around the minor circle."""

    major_radius: float
    minor_radius: float
    height: float

    @property
    def outer_radius(self):
        """Radius of the outer equator."""
        return self.major_radius + self.minor_radius

    @property
    def curvature_radius(self):
        """Radius of curvature of the surface, the tube radius."""
        return self.minor_radius

    def clearance(self, point):
        """Gap between the tube surface and an on-axis sphere, negative if they meet."""
        return (
            math.hypot(self.major_radius, point.height - self.height)
            - self.minor_radius
            - point.radius
        )

    def discretise(self, sections):
        """Rings spaced uniformly in poloidal angle over the tube surface."""
        theta = 2.0 * np.pi * (np.arange(sections) + 0.5) / sections
        arc = 2.0 * np.pi * self.minor_radius / sections
        return Rings(
            a=self.major_radius + self.minor_radius * np.cos(theta),
            z=self.height + self.minor_radius * np.sin(theta),
            n=np.ones(sections),
            w=np.full(sections, arc),
            rw=np.full(sections, 0.25 * arc),
        )


@dataclass(frozen=True)
class Sphere:
    """Spherical top load, discretised as rings of constant polar angle."""

    radius: float
    height: float

    @property
    def outer_radius(self):
        """Equatorial radius."""
        return self.radius

    @property
    def curvature_radius(self):
        """Radius of curvature of the surface."""
        return self.radius

    def clearance(self, point):
        """Gap between this surface and an on-axis sphere, negative if they meet."""
        return abs(point.height - self.height) - self.radius - point.radius

    def discretise(self, sections):
        """Rings spaced uniformly in polar angle."""
        theta = np.pi * (np.arange(sections) + 0.5) / sections
        arc = np.pi * self.radius / sections
        return Rings(
            a=self.radius * np.sin(theta),
            z=self.height + self.radius * np.cos(theta),
            n=np.ones(sections),
            w=np.full(sections, arc),
            rw=np.full(sections, 0.25 * arc),
        )


@dataclass(frozen=True)
class Primary:
    """Flat spiral, helical or conical primary; one ring per turn.

    ``rise`` is the axial gain per turn and ``pitch`` the radial gain per turn, so
    a flat spiral has rise = 0 and a helix has pitch = 0.
    """

    inner_radius: float
    turns: float
    pitch: float = 0.0
    rise: float = 0.0
    base: float = 0.0
    wire_diameter: float = 0.006

    def discretise(self, sections=None):
        """One ring per turn, or ``sections`` equal turn-count rings."""
        count = int(np.ceil(self.turns)) if sections is None else sections
        t = self.turns * (np.arange(count) + 0.5) / count
        return Rings(
            a=self.inner_radius + self.pitch * t,
            z=self.base + self.rise * t,
            n=np.full(count, self.turns / count),
            w=np.full(count, self.wire_diameter),
            rw=np.full(count, 0.5 * self.wire_diameter),
        )


@dataclass(frozen=True)
class Dielectric:
    """Closed boundary of a dielectric region: rings with outward unit normals."""

    rings: Rings
    nr: np.ndarray
    nz: np.ndarray
    permittivity: float

    def __post_init__(self):
        if len(self.nr) != len(self.rings) or len(self.nz) != len(self.rings):
            raise ValueError("normal components must match the ring count")

    def __len__(self):
        return len(self.rings)

    @property
    def area(self):
        """Band area of each boundary ring."""
        return self.rings.area

    @property
    def susceptibility(self):
        """Bound-charge coefficient lam = (eps_r - 1) / (eps_r + 1)."""
        return (self.permittivity - 1.0) / (self.permittivity + 1.0)


@dataclass(frozen=True)
class Former:
    """Dielectric winding former: a coaxial tube, or a solid rod when inner_radius = 0."""

    outer_radius: float
    length: float
    base: float = 0.0
    inner_radius: float = 0.0
    permittivity: float = 2.56
    loss_tangent: float = 0.0

    def discretise(self, sections):
        """Bands around the closed meridian contour, allocated by meridian length.

        The contour runs outer wall, top annulus, inner wall where the former is
        hollow, then bottom annulus, every normal pointing out of the dielectric.
        Band ``rw`` is w / 4, the flat-strip equivalent radius used for the
        toroid and sphere discretisations.
        """
        top = self.base + self.length
        edges = [
            ((self.outer_radius, self.base), (self.outer_radius, top), (1.0, 0.0)),
            ((self.outer_radius, top), (self.inner_radius, top), (0.0, 1.0)),
            ((self.inner_radius, top), (self.inner_radius, self.base), (-1.0, 0.0)),
            (
                (self.inner_radius, self.base),
                (self.outer_radius, self.base),
                (0.0, -1.0),
            ),
        ]
        if self.inner_radius == 0.0:
            del edges[2]
        start, end, normal = (np.array(f) for f in zip(*edges))
        span = np.linalg.norm(end - start, axis=1)
        counts = _apportion(sections, span)
        w = np.repeat(span / counts, counts)
        t = np.concatenate([(np.arange(k) + 0.5) / k for k in counts])[:, None]
        point = np.repeat(start, counts, axis=0) + t * np.repeat(
            end - start, counts, axis=0
        )
        normal = np.repeat(normal, counts, axis=0)
        return Dielectric(
            rings=Rings(
                a=point[:, 0],
                z=point[:, 1],
                n=np.ones(w.size),
                w=w,
                rw=0.25 * w,
            ),
            nr=normal[:, 0],
            nz=normal[:, 1],
            permittivity=self.permittivity,
        )


@dataclass(frozen=True)
class Design:  # pylint: disable=too-many-instance-attributes
    """A complete coil: secondary, top load, former, primary, and ground plane."""

    secondary: Solenoid
    primary: Primary
    top_load: Toroid | Sphere | None = None
    breakout: Sphere | None = None
    former: Former | None = None
    ground_plane: bool = True
    sections: int = 120
    top_load_sections: int = 32
    breakout_sections: int = 8
    former_sections: int = 96

    def __post_init__(self):
        if self.breakout is None or self.top_load is None:
            return
        gap = self.top_load.clearance(self.breakout)
        if gap <= 0.0:
            raise ValueError(f"the breakout point intersects the top load by {-gap} m")

    @property
    def electrodes(self):
        """Top-node conductors in ring order: the top load, then the breakout point."""
        return tuple(
            part for part in (self.top_load, self.breakout) if part is not None
        )

    def secondary_rings(self):
        """Ring discretisation of the secondary winding."""
        return self.secondary.discretise(self.sections)

    def top_load_rings(self):
        """Rings of the top load and breakout point, empty if there is neither.

        The breakout point is a sphere at the end of a stalk; the stalk itself is
        left out, its surface field being far below the tip's.
        """
        parts = [
            part.discretise(count)
            for part, count in (
                (self.top_load, self.top_load_sections),
                (self.breakout, self.breakout_sections),
            )
            if part is not None
        ]
        if not parts:
            return Rings(*(np.zeros(0) for _ in range(5)))
        return Rings.concat(*parts)

    def top_load_curvature(self):
        """Surface curvature radius of each ring of :meth:`top_load_rings`."""
        counts = (self.top_load_sections, self.breakout_sections)
        return np.concatenate(
            [
                np.full(count, part.curvature_radius)
                for part, count in zip(self.electrodes, counts)
            ]
            or [np.zeros(0)]
        )

    def dielectric(self):
        """Discretised boundary of the winding former, or None if there is none."""
        if self.former is None:
            return None
        return self.former.discretise(self.former_sections)

    def primary_rings(self):
        """Ring discretisation of the primary winding."""
        return self.primary.discretise()

    @classmethod
    def from_yaml(cls, path):
        """Load a design from the YAML schema in ``docs/schema.md``."""
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))

    @classmethod
    def from_dict(cls, spec):
        """Build a design from a plain mapping."""
        spec = dict(spec)
        former = spec.pop("former", None)
        top = spec.pop("top_load", None)
        point = spec.pop("breakout", None)
        kinds = {"toroid": Toroid, "sphere": Sphere}
        if top is not None:
            top = dict(top)
            kind = top.pop("kind")
            if kind not in kinds:
                raise ValueError(f"unknown top load kind {kind!r}")
            top = kinds[kind](**top)
        return cls(
            secondary=Solenoid(**spec.pop("secondary")),
            primary=Primary(**spec.pop("primary")),
            top_load=top,
            breakout=None if point is None else Sphere(**point),
            former=None if former is None else Former(**former),
            **spec,
        )
