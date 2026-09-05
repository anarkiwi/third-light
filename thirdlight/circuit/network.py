"""Piecewise-linear state space of the primary tank driving M secondary modes.

State x = [i_p, i_1 .. i_M, v_Cp, v_1 .. v_M], n = 2 (M + 1), obeying

    L di/dt = e_p v_drive - R i - v,  dv_Cp/dt = i_p / C_p,
    dv_m/dt = (i_m - i_load) / c_m,

so A = [[-L^-1 R, -L^-1], [C^-1, 0]] and B carries u = [v_drive, i_load]. Modes
are C-orthogonal, so L is an arrowhead matrix: primary self, modal self referred
to the top node, and the mutuals k_m sqrt(L_p l_m). A load current drawn at the
top node forces every mode identically, hence a -1/c_m entry per mode row and
none on v_Cp.

Only the conducting device's differential resistance distinguishes a conducting
bridge state from another, and the blocked bridge constrains i_p to zero by
dropping the primary row and column of L and freezing the tank charge, so there
are three A matrices, indexed
by :data:`~thirdlight.circuit.devices.IGBT`, ``DIODE`` and ``OPEN``, and one B.
"""

import math
from dataclasses import dataclass

import numpy as np

from thirdlight import secondary
from thirdlight.circuit.devices import DIODE, IGBT, Bridge
from thirdlight.em import inductance, losses


@dataclass(frozen=True)
class Tank:
    """Primary tank: series capacitance, loop resistance, and inductance override."""

    capacitance: float
    resistance: float = 0.0
    inductance: float | None = None


@dataclass(frozen=True)
class Network:
    """Piecewise-linear state space of the primary tank and M secondary modes."""

    a: np.ndarray
    b: np.ndarray
    modes: int
    inductances: np.ndarray
    capacitances: np.ndarray
    frequencies: np.ndarray
    bridge: Bridge

    @property
    def size(self):
        """State dimension n = 2 (M + 1)."""
        return self.a.shape[-1]

    @property
    def resistances(self):
        """Loop resistance of each conduction state, shape (3, M + 1).

        Recovered as diag(-L A_ii) from the current block, so it always reports
        the resistance the state matrices actually carry.
        """
        loops = self.modes + 1
        return -np.diagonal(
            self.inductances @ self.a[:, :loops, :loops], axis1=-2, axis2=-1
        )

    def currents(self, x):
        """Current part of the state, [i_p, i_1 .. i_M]."""
        return np.asarray(x)[..., : self.modes + 1]

    def voltages(self, x):
        """Capacitor-voltage part of the state, [v_Cp, v_1 .. v_M]."""
        return np.asarray(x)[..., self.modes + 1 :]

    def primary_current(self, x):
        """Primary tank current i_p."""
        return np.asarray(x)[..., 0]

    def top_voltage(self, x):
        """Top-node potential, the sum of the modal voltages."""
        return np.asarray(x)[..., self.modes + 2 :].sum(axis=-1)

    def energy(self, x):
        """Stored energy (1/2) i L i + (1/2) sum_j C_j v_j^2."""
        i, v = self.currents(x), self.voltages(x)
        magnetic = np.einsum("...i,ij,...j->...", i, self.inductances, i)
        return 0.5 * (magnetic + (self.capacitances * v * v).sum(axis=-1))

    def state(self, gate, current):
        """Conduction state ``(index into a, polarity sigma, current sign)``."""
        return self.bridge.state(gate, current)

    def drive(self, gate, current, v_bus):
        """Input u[0]: bridge output less the constant part of the conduction drop."""
        return self.bridge.drive(gate, current, v_bus)


def _state_matrix(inverse, cinv, resistances):
    """Assemble [[-L^-1 R, -L^-1], [C^-1, 0]] for one loop-resistance vector."""
    return np.block(
        [
            [-inverse * resistances, -inverse],
            [np.diag(cinv), np.zeros_like(inverse)],
        ]
    )


def from_modes(modes, coupling, quality, primary_inductance, tank, bridge):
    """State space from top-referred modal equivalents and a primary tank.

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
    a = np.stack(
        [
            _state_matrix(
                inverse,
                cinv,
                np.concatenate(
                    (
                        [tank.resistance + bridge.devices * bridge.conducting(index).r],
                        modal_r,
                    )
                ),
            )
            for index in (IGBT, DIODE)
        ]
        + [
            _state_matrix(
                blocked,
                np.concatenate(([0.0], cinv[1:])),
                np.concatenate(([0.0], modal_r)),
            )
        ]
    )
    b = np.zeros((2 * (count + 1), 2))
    b[: count + 1, 0] = inverse[:, 0]
    b[count + 2 :, 1] = -cinv[1:]
    return Network(
        a=a,
        b=b,
        modes=count,
        inductances=inductances,
        capacitances=capacitances,
        frequencies=np.asarray(modes.f),
        bridge=bridge,
    )


def from_design(design, tank, bridge, modes=2, **loss_kwargs):
    """State space of a design's primary tank and its lowest ``modes`` secondary modes."""
    eigen = secondary.resonance(design, modes=modes)
    return from_modes(
        eigen,
        secondary.coupling(design, eigen),
        losses.quality_factor(design, eigen, **loss_kwargs),
        float(inductance.inductance_matrix(design.primary_rings()).sum()),
        tank,
        bridge,
    )


def tune(primary_inductance, frequency, ratio=1.0):
    """Series tank capacitance placing the primary resonance at ``ratio`` times ``frequency``."""
    return 1.0 / (primary_inductance * (2.0 * math.pi * ratio * frequency) ** 2)
