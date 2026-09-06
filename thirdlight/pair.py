"""Two coils standing side by side: electrode coupling and the pair's modes.

Each coil is solved on its own axisymmetric problem, neither tower perturbing
the other's, and the two meet only at their top nodes, where essentially all of
a grounded quarter-wave resonator's charge sits. See docs/design.md 3.8.
"""

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.constants import epsilon_0

from thirdlight.em import capacitance
from thirdlight.geometry import Design
from thirdlight.secondary import Modes, resonance

FOUR_PI_EPS0 = 4.0 * math.pi * epsilon_0


def electrode_height(design):
    """Charge-weighted centroid height of the top-node electrode, m.

    The electrode is held at unit potential over its own ground plane, the
    winding below it left out: it carries little of the top node's charge and
    what it carries sits near the axis. With no electrode, the winding's top.
    """
    rings = design.top_load_rings()
    if len(rings) == 0:
        return design.secondary.base + design.secondary.length
    charges = capacitance.unit_potential_charges(rings, design.ground_plane)
    return float(charges @ rings.z / charges.sum())


def electrode_radius(design):
    """Radius of the top-node electrode about the coil axis, m."""
    parts = [part.outer_radius for part in design.electrodes]
    return max(parts) if parts else design.secondary.radius


def mutual_coefficient(separation, height_a, height_b, ground_plane=True):
    """Potential coefficient between two point-charge electrodes, V/C.

    (1/sqrt(s^2 + (h_a - h_b)^2) - 1/sqrt(s^2 + (h_a + h_b)^2)) / 4 pi eps0, the
    second term the ground plane's image of the other electrode. Leading order
    in electrode size over separation; what it omits is in docs/design.md 3.8.
    """
    direct = 1.0 / math.hypot(separation, height_a - height_b)
    image = 0.0
    if ground_plane:
        image = 1.0 / math.hypot(separation, height_a + height_b)
    return (direct - image) / FOUR_PI_EPS0


def maxwell_capacitance(c_a, c_b, p12):
    """Maxwell capacitance matrix C = P^-1 of the two top nodes, F.

    P = [[1/c_a, p12], [p12, 1/c_b]] over the top-referred modal capacitances.
    The 2x2 inverse is taken as [[c_a, -k], [-k, c_b]] / (1 - k p12) with
    k = c_a c_b p12, which is exact rather than merely accurate at p12 = 0.
    """
    k = c_a * c_b * p12
    det = 1.0 - k * p12
    if det <= 0.0:
        raise ValueError(
            f"p12 = {p12} exceeds the two-node reduction's range: P is indefinite"
        )
    return np.array([[c_a, -k], [-k, c_b]]) / det


def mutual_capacitance(matrix):
    """Mutual capacitance between the two towers, -C[0, 1], in F."""
    return float(-matrix[0, 1])


def coupling(matrix):
    """Mutual capacitance over the geometric mean of the two self terms.

    For identical coils this is the fractional mode splitting to leading order,
    and it is the off-diagonal of the two-level problem :func:`locks` weighs.
    """
    return float(-matrix[0, 1] / math.sqrt(matrix[0, 0] * matrix[1, 1]))


def locks(coupling_, detune):
    """Whether the pair's modes are shared between the towers, not localised.

    The two-level splitting goes as sqrt(detune^2 + coupling^2), so the mixing
    angle turns on coupling / detune alone: the modes delocalise once the
    coupling exceeds the fractional detune and localise once it does not.
    """
    return bool(coupling_ > detune)


@dataclass(frozen=True)
class Split:
    """The pair's two coupled modes, ascending in frequency.

    ``v`` carries one unit-norm row per mode, its entries the mode's amplitude
    on each tower in the basis that diagonalises the uncoupled pair.
    """

    f: np.ndarray
    v: np.ndarray

    def __len__(self):
        return len(self.f)

    @property
    def participation(self):
        """Participation ratio 1 / sum(v^4) per mode.

        2 at equal amplitudes, 1 with the whole mode on one tower; the two-level
        mixing angle puts the :func:`locks` threshold at 4/3.
        """
        return 1.0 / np.sum(self.v**4, axis=1)


def coupled_modes(l_a, l_b, matrix):
    """The pair's two resonances and their tower amplitudes.

    diag(1/l_a, 1/l_b) v = omega^2 C v, scaled by D = diag(sqrt(l_a), sqrt(l_b))
    into the standard symmetric D C D u = omega^-2 u, whose eigenvectors are
    orthonormal in the tower basis and so measure how the mode is shared.
    """
    scale = np.sqrt([l_a, l_b])
    lam, vec = np.linalg.eigh(matrix * np.outer(scale, scale))
    vec = vec[:, ::-1]
    vec = vec * np.where(vec[np.abs(vec).argmax(axis=0), [0, 1]] < 0.0, -1.0, 1.0)
    return Split(f=1.0 / (2.0 * math.pi * np.sqrt(lam[::-1])), v=vec.T)


def bridged_gap(reach_a, reach_b):
    """Gap two coils driven in antiphase can bridge, m.

    The gap sees the sum of the two terminal voltages, so each channel need only
    cover its own coil's reach to ground and the pair spans the sum of the two.
    """
    return reach_a + reach_b


def bridges(gap, reach_a, reach_b):
    """Whether a gap is within the antiphase pair's :func:`bridged_gap`."""
    return bool(gap <= bridged_gap(reach_a, reach_b))


@dataclass(frozen=True)
class Terminal:
    """A coil's top node as the pair sees it: a point charge with an l and a c.

    ``l_m`` and ``c_m`` are the mode-1 modal inductance and capacitance referred
    to the top node, so the reduction adds nothing to the single-coil solve but
    the height the charge sits at.
    """

    height: float
    l_m: float
    c_m: float
    ground_plane: bool
    radius: float

    @property
    def f(self):
        """Mode-1 frequency of the isolated coil, Hz."""
        return 1.0 / (2.0 * math.pi * math.sqrt(self.l_m * self.c_m))


def terminal(design, modes=None):
    """Reduce a design's top node to a :class:`Terminal`.

    ``modes`` is the design's own :func:`~thirdlight.secondary.resonance`, the
    expensive part of a pair, and is worth passing where the caller has it.
    """
    if modes is None:
        modes = resonance(design, modes=1)
    return Terminal(
        height=electrode_height(design),
        l_m=float(modes.l_m[0]),
        c_m=float(modes.c_m[0]),
        ground_plane=design.ground_plane,
        radius=electrode_radius(design),
    )


@dataclass(frozen=True)
class Pair:
    """Two coils standing ``separation`` apart, centre to centre, coupled at the top.

    ``modes_a`` and ``modes_b`` are the two designs' own resonances where the
    caller has them; they are solved once on first use otherwise.
    """

    a: Design
    b: Design
    separation: float
    modes_a: Modes | None = None
    modes_b: Modes | None = None

    def __post_init__(self):
        if self.separation <= 0.0:
            raise ValueError(f"separation {self.separation} is not positive")
        if self.a.ground_plane != self.b.ground_plane:
            raise ValueError("the two designs disagree about the ground plane")

    @cached_property
    def terminals(self):
        """The two coils' top nodes, reduced to point charges."""
        return (terminal(self.a, self.modes_a), terminal(self.b, self.modes_b))

    @property
    def frequencies(self):
        """The two coils' isolated mode-1 frequencies, Hz."""
        return np.array([node.f for node in self.terminals])

    @property
    def detune(self):
        """Frequency difference of the two coils over their mean."""
        f = self.frequencies
        return float(abs(f[0] - f[1]) / f.mean())

    @property
    def coefficient(self):
        """Mutual potential coefficient of the two electrodes, V/C."""
        first, second = self.terminals
        return mutual_coefficient(
            self.separation, first.height, second.height, first.ground_plane
        )

    @cached_property
    def capacitance(self):
        """Maxwell capacitance matrix of the two top nodes, F."""
        first, second = self.terminals
        return maxwell_capacitance(first.c_m, second.c_m, self.coefficient)

    @property
    def mutual(self):
        """Mutual capacitance between the two towers, F."""
        return mutual_capacitance(self.capacitance)

    @property
    def coupling(self):
        """Mutual capacitance as a fraction of the towers' own, the coupling."""
        return coupling(self.capacitance)

    @cached_property
    def split(self):
        """The pair's two coupled resonances, ascending, as a :class:`Split`.

        The in-phase mode drives no current through the mutual capacitance but
        sees each tower screened by the other, which lifts it above the isolated
        f0; the antiphase mode drives the mutual capacitance and falls below it.
        """
        first, second = self.terminals
        return coupled_modes(first.l_m, second.l_m, self.capacitance)

    @property
    def locks(self):
        """Whether the two modes are shared between the towers; see :func:`locks`."""
        return locks(self.coupling, self.detune)

    @property
    def gap(self):
        """Air gap between the two electrodes at their own radii, m."""
        first, second = self.terminals
        return self.separation - first.radius - second.radius

    def bridges(self, reach_a, reach_b):
        """Whether antiphase drive bridges :attr:`gap`; see :func:`bridged_gap`."""
        return bridges(self.gap, reach_a, reach_b)
