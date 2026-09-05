"""Piecewise-linear state space of the primary tank driving M secondary modes.

State x = [i_p, i_1 .. i_M, v_Cp, v_1 .. v_M], and v_bus last when the bus is a
reservoir rather than stiff; input u = [drop, i_load, v_supply]. The equations and
the sign conventions are §3.2 of docs/design.md.
"""

import math
from dataclasses import dataclass

import numpy as np

from thirdlight import secondary
from thirdlight.circuit.devices import DIODE, IGBT, OPEN, STATES, Bridge, index
from thirdlight.em import inductance, losses


@dataclass(frozen=True)
class Tank:
    """Primary tank: series capacitance, loop resistance, and inductance override."""

    capacitance: float
    resistance: float = 0.0
    inductance: float | None = None


@dataclass(frozen=True)
class Bus:
    """DC bus feeding the bridge: stiff by default, or a reservoir that sags.

    A ``capacitance`` makes v_bus a state charged from the supply through
    ``resistance``, the rectifier and mains impedance, which must then be
    positive; the supply voltage itself, ripple and all, stays an input.
    """

    capacitance: float | None = None
    resistance: float = 0.0

    def __post_init__(self):
        if self.capacitance is not None and self.resistance <= 0.0:
            raise ValueError("a bus reservoir needs a positive supply resistance")

    @property
    def reservoir(self):
        """Whether the bus carries a state of its own."""
        return self.capacitance is not None


@dataclass(frozen=True)
class Network:  # pylint: disable=too-many-instance-attributes
    """Piecewise-linear state space of the primary tank, M secondary modes and the bus."""

    a: np.ndarray
    b: np.ndarray
    modes: int
    inductances: np.ndarray
    capacitances: np.ndarray
    frequencies: np.ndarray
    bridge: Bridge
    bus: Bus

    @property
    def size(self):
        """State dimension, 2 (M + 1) and one more for a bus reservoir."""
        return self.a.shape[-1]

    @property
    def loops(self):
        """Number of current loops, the primary and each mode."""
        return self.modes + 1

    @property
    def resistances(self):
        """Loop resistance of each bridge state, shape (5, M + 1).

        Recovered as diag(-L A_ii) from the current block, so it always reports
        the resistance the state matrices actually carry.
        """
        return -np.diagonal(
            self.inductances @ self.a[:, : self.loops, : self.loops], axis1=-2, axis2=-1
        )

    def currents(self, x):
        """Current part of the state, [i_p, i_1 .. i_M]."""
        return np.asarray(x)[..., : self.loops]

    def voltages(self, x):
        """Capacitor-voltage part of the state, [v_Cp, v_1 .. v_M]."""
        return np.asarray(x)[..., self.loops : 2 * self.loops]

    def primary_current(self, x):
        """Primary tank current i_p."""
        return np.asarray(x)[..., 0]

    def top_voltage(self, x):
        """Top-node potential, the sum of the modal voltages."""
        return np.asarray(x)[..., self.loops + 1 : 2 * self.loops].sum(axis=-1)

    def bus_voltage(self, x, supply):
        """Bus voltage: the reservoir state if there is one, else the supply itself."""
        return np.asarray(x)[..., -1] if self.bus.reservoir else supply

    def energy(self, x):
        """Stored energy (1/2) i L i + (1/2) sum_j C_j v_j^2, the bus excluded."""
        i, v = self.currents(x), self.voltages(x)
        magnetic = np.einsum("...i,ij,...j->...", i, self.inductances, i)
        return 0.5 * (magnetic + (self.capacitances * v * v).sum(axis=-1))

    def state(self, gate, current):
        """Conduction state ``(kind, polarity sigma, current sign)``."""
        return self.bridge.state(gate, current)

    def index(self, gate, current):
        """Row of :attr:`a` and :attr:`b` for the present bridge state."""
        kind, sigma, _ = self.bridge.state(gate, current)
        return index(kind, sigma)

    def offset(self, gate, current):
        """Input u[0]: the constant part of the conduction drop, in volts."""
        return self.bridge.offset(gate, current)

    def drive(self, gate, current, v_bus):
        """Bridge output across the tank, less its differential resistance."""
        return self.bridge.drive(gate, current, v_bus)


def _state_matrix(inverse, cinv, resistances):
    """Assemble [[-L^-1 R, -L^-1], [C^-1, 0]] for one loop-resistance vector."""
    return np.block(
        [
            [-inverse * resistances, -inverse],
            [np.diag(cinv), np.zeros_like(inverse)],
        ]
    )


def _border(core, drive, coupling, sigma, bus):
    """Extend one (A, B) pair with the bus, as a state or as a third input."""
    size = core.shape[0]
    b = np.zeros((size + bus.reservoir, 3))
    b[:size] = drive
    if not bus.reservoir:
        b[: size // 2, 2] = sigma * coupling
        return core, b
    rate = 1.0 / (bus.resistance * bus.capacitance)
    a = np.zeros((size + 1, size + 1))
    a[:size, :size] = core
    a[: size // 2, size] = sigma * coupling
    a[size, 0] = -sigma / bus.capacitance
    a[size, size] = -rate
    b[size, 2] = rate
    return a, b


def from_modes(modes, coupling, quality, primary_inductance, tank, bridge, bus=Bus()):
    """State space from top-referred modal equivalents, a primary tank and a bus.

    ``modes`` supplies f_m, l_m and c_m, ``coupling`` the k_m to the primary and
    ``quality`` the unloaded modal Q, from which r_m = 2 pi f_m l_m / Q_m.
    ``tank.inductance`` overrides ``primary_inductance`` when it is not None.
    """
    count = len(modes)
    l_p = primary_inductance if tank.inductance is None else tank.inductance
    inductances = np.diag(np.concatenate(([l_p], modes.l_m)))
    mutual = np.asarray(coupling) * np.sqrt(l_p * modes.l_m)
    inductances[0, 1:] = inductances[1:, 0] = mutual
    capacitances = np.concatenate(([tank.capacitance], modes.c_m))
    modal_r = 2.0 * math.pi * modes.f * modes.l_m / np.asarray(quality)
    inverse = np.linalg.inv(inductances)
    cinv = 1.0 / capacitances
    blocked = np.zeros_like(inverse)
    blocked[1:, 1:] = np.diag(1.0 / modes.l_m)
    drive = np.zeros((2 * (count + 1), 3))
    drive[: count + 1, 0] = inverse[:, 0]
    drive[count + 2 :, 1] = -cinv[1:]
    core = {
        kind: _state_matrix(
            inverse,
            cinv,
            np.concatenate(
                (
                    [tank.resistance + bridge.devices * bridge.conducting(kind).r],
                    modal_r,
                )
            ),
        )
        for kind in (IGBT, DIODE)
    }
    core[OPEN] = _state_matrix(
        blocked,
        np.concatenate(([0.0], cinv[1:])),
        np.concatenate(([0.0], modal_r)),
    )
    rows = [(OPEN, 0.0)] * STATES
    for kind in (IGBT, DIODE):
        for sigma in (1.0, -1.0):
            rows[index(kind, sigma)] = (kind, sigma)
    built = [
        _border(core[kind], drive, inverse[:, 0], sigma * bridge.gain, bus)
        for kind, sigma in rows
    ]
    return Network(
        a=np.stack([pair[0] for pair in built]),
        b=np.stack([pair[1] for pair in built]),
        modes=count,
        inductances=inductances,
        capacitances=capacitances,
        frequencies=np.asarray(modes.f),
        bridge=bridge,
        bus=bus,
    )


def from_design(design, tank, bridge, bus=Bus(), modes=2, **loss_kwargs):
    """State space of a design's primary tank and its lowest ``modes`` secondary modes."""
    eigen = secondary.resonance(design, modes=modes)
    return from_modes(
        eigen,
        secondary.coupling(design, eigen),
        losses.quality_factor(design, eigen, **loss_kwargs),
        float(inductance.inductance_matrix(design.primary_rings()).sum()),
        tank,
        bridge,
        bus,
    )


def tune(primary_inductance, frequency, ratio=1.0):
    """Series tank capacitance placing the primary resonance at ``ratio`` times ``frequency``."""
    return 1.0 / (primary_inductance * (2.0 * math.pi * ratio * frequency) ** 2)
