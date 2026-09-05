"""Coil, primary, top-load and ground descriptions, and their ring discretisation.

Every electromagnetic element reduces to a set of coaxial rings sharing the z
axis. A ring carries ``n`` turns at radius ``a`` and axial centre ``z``, occupies
an axial extent ``w``, and has an equivalent conductor radius ``rw`` used for the
self terms of the inductance and potential-coefficient matrices.

Coordinates are SI, with z measured from the ground plane.
"""

from dataclasses import dataclass

import numpy as np
import yaml


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
class Design:
    """A complete coil: secondary, top load, primary, and ground plane."""

    secondary: Solenoid
    primary: Primary
    top_load: Toroid | Sphere | None = None
    ground_plane: bool = True
    sections: int = 120
    top_load_sections: int = 32

    def secondary_rings(self):
        """Ring discretisation of the secondary winding."""
        return self.secondary.discretise(self.sections)

    def top_load_rings(self):
        """Ring discretisation of the top load, empty if there is none."""
        if self.top_load is None:
            return Rings(*(np.zeros(0) for _ in range(5)))
        return self.top_load.discretise(self.top_load_sections)

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
        top = spec.pop("top_load", None)
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
            **spec,
        )
