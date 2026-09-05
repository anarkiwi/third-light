"""Piecewise-linear semiconductor and bridge models for the primary drive.

A conducting device is a constant drop plus a differential resistance,
v = v0 + r i. Which device conducts follows from the gate command and the sign
of the primary current. Which of those conducts, and with which polarity the
bridge stands across the tank, together index five linear circuits: an IGBT or a
diode conducting at either polarity, and the blocked bridge that carries no
primary current at all and stands at no polarity.
"""

from dataclasses import dataclass

import numpy as np

IGBT = 0
DIODE = 1
OPEN = 2
STATES = 5


def index(conduction, sigma):
    """Row of the state-matrix stack for a conduction kind and bridge polarity."""
    return 2 * conduction + (sigma < 0.0)


def polarity(row):
    """Bridge polarity of a state-matrix row, 0 where the bridge is blocked."""
    row = np.asarray(row)
    return np.where(row < 2 * OPEN, 1.0 - 2.0 * (row % 2), 0.0)


@dataclass(frozen=True)
class Switch:
    """Piecewise-linear conducting device, v = v0 + r i."""

    v0: float
    r: float


@dataclass(frozen=True)
class Bridge:
    """Full or half bridge of IGBT/diode pairs feeding the primary tank."""

    igbt: Switch
    diode: Switch
    full: bool = True

    @property
    def devices(self):
        """Devices in series with the tank: two for a full bridge, one for a half."""
        return 2 if self.full else 1

    @property
    def gain(self):
        """Output swing per unit bus voltage: 1 for a full bridge, 1/2 for a half."""
        return 1.0 if self.full else 0.5

    def conducting(self, kind):
        """The :class:`Switch` conducting in state ``IGBT`` or ``DIODE``, None when open."""
        return (self.igbt, self.diode, None)[kind]

    def state(self, gate, current):
        """Conduction state ``(kind, polarity sigma, current sign)`` of the bridge.

        The commanded IGBTs conduct only while the current already flows with the
        command; against it the anti-parallel diodes of the same legs carry it at
        the same polarity. A zero command freewheels through the diodes back into
        the bus, so the polarity opposes the current and drives it to zero. A zero
        current is the blocked bridge: nothing conducts until some polarity is
        consistent with the loop voltage, which the integrator tests for.
        """
        if current == 0.0:
            return OPEN, 0.0, 0.0
        sign = 1.0 if current > 0.0 else -1.0
        if gate == 0:
            return DIODE, -sign, sign
        return (IGBT if sign == gate else DIODE), float(gate), sign

    def offset(self, gate, current):
        """Constant part of the conduction drop seen by the tank, in volts.

        The differential part n_dev r i is linear in the state and belongs in the
        primary loop resistance, and sigma gain v_bus is carried by the state
        matrices, so only the n_dev v0 term is left; it opposes the current.
        """
        kind, _, sign = self.state(gate, current)
        switch = self.conducting(kind)
        return 0.0 if switch is None else -self.devices * switch.v0 * sign

    def drive(self, gate, current, v_bus):
        """Bridge output across the tank, less its differential resistance."""
        sigma = self.state(gate, current)[1]
        return sigma * self.gain * v_bus + self.offset(gate, current)
