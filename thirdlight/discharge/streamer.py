"""Fritz streamer load: a series R-C branch on the top node, with its own length.

The channel is a resistance in series with a capacitance proportional to its
length, and that length is a state of its own: it grows while the top voltage
exceeds the gradient the channel needs to sustain itself, and decays with a
cooling time constant.

The sustaining gradient is not fitted. It is the 1.5-3 kV/cm coilers measure
across long Tesla coil sparks, whose channels are thermalised leaders rather
than the cold positive streamers that need 5 kV/cm, and it alone caps the
length at V / E. In a burst long enough for the channel to reach that cap --
which a DRSSTC bang is -- the growth gain and the cooling time only set how
quickly it gets there: raising the gain by 7.5x moves the settled length by 5 %,
and quadrupling the cooling time by 4 %. Both are therefore set at their
physical scales, the gain at a channel velocity within the published leader
range and the cooling time at the channel's own, rather than pinned by data that
does not resolve them. See docs/design.md 3.4a.
"""

import math
from dataclasses import dataclass

from thirdlight.discharge.breakout import Breakout

FRITZ_RESISTANCE = 220.0e3
FRITZ_CAPACITANCE = 1.0e-12 / 0.3048
SUSTAINING_GRADIENT = 2.0e5
GROWTH_GAIN = 0.4
COOLING_TIME = 5.0e-4


@dataclass(frozen=True)
class Streamer:  # pylint: disable=too-many-instance-attributes
    """Fritz channel load and the length dynamics that set its capacitance."""

    breakout: Breakout
    frequency: float
    growth: float = GROWTH_GAIN
    cooling: float = COOLING_TIME
    resistance: float = FRITZ_RESISTANCE
    capacitance: float = FRITZ_CAPACITANCE
    gradient: float = SUSTAINING_GRADIENT
    tolerance: float = 0.02
    floor: float = 1.0e-3

    @property
    def minimum(self):
        """Length below which the branch is immaterial, and where its capacitance is held.

        omega R C(l) is the branch's admittance as a fraction of its own 1 / R;
        below ``floor`` of that the channel neither loads nor detunes the top
        node, so holding the capacitance there keeps one state space for the
        whole run instead of attaching and detaching one.
        """
        return self.floor / (
            2.0 * math.pi * self.frequency * self.resistance * self.capacitance
        )

    @property
    def rate(self):
        """Relaxation rate of the length ODE while the channel is being driven."""
        return self.growth * self.gradient + 1.0 / self.cooling

    def equilibrium(self, voltage):
        """Length a held top voltage sustains, growth against gradient and cooling."""
        return self.growth * abs(voltage) / self.rate

    def level(self, length):
        """Quantised capacitance level of a length, geometric in ``tolerance``."""
        if length <= self.minimum:
            return 0
        return round(math.log(length / self.minimum) / math.log1p(self.tolerance))

    def capacitance_at(self, level):
        """Channel capacitance of a quantised level, F."""
        return self.capacitance * self.minimum * (1.0 + self.tolerance) ** level

    def advance(self, length, voltage, margin, dt):
        """Length after ``dt`` with the top voltage held, by the exact linear update.

        Initiation needs the electrode surface to reach Peek's threshold; once a
        channel exists its own tip carries the field, so growth continues on the
        top voltage alone. Both regimes of ``(|v| - E l)_+`` are linear, so each
        step is one exponential rather than a sub-stepped integration.
        """
        if length <= 0.0 and margin < 1.0:
            return 0.0
        voltage = abs(voltage)
        if voltage > self.gradient * length:
            target = self.equilibrium(voltage)
            return target + (length - target) * math.exp(-self.rate * dt)
        return length * math.exp(-dt / self.cooling)
