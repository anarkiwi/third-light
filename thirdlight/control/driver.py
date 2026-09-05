"""Self-oscillating bridge driver: comparator, gate delay, dead time, bus envelope.

The sequencer is a pure state machine over two event streams the integrator
supplies, comparator crossings and interrupter edges. It never steps time
itself: it reports the next scheduled transition and applies transitions on
demand, so the integrator can propagate exactly to each one.
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from thirdlight.control.feedback import PhaseLead
from thirdlight.control.interrupter import Interrupter, Melody


@dataclass(frozen=True)
class Ramp:
    """Bus-voltage envelope within a burst: a QCW ramp, or a flat DC bus.

    Linear from ``initial`` to ``final`` over ``rise`` seconds measured from the
    burst start, then held; ``rise = 0`` is a flat bus at ``final``. A ramp from
    zero does not start a self-oscillating driver: without a step to ring the
    tank the seed pulse only charges it, so ``initial`` carries the pedestal a
    QCW modulator starts from.
    """

    final: float
    initial: float = 0.0
    rise: float = 0.0

    def voltage(self, elapsed):
        """Bus voltage ``elapsed`` seconds into a burst."""
        elapsed = np.asarray(elapsed, dtype=float)
        span = (
            np.clip(elapsed / self.rise, 0.0, 1.0)
            if self.rise > 0.0
            else np.ones_like(elapsed)
        )
        return self.initial + (self.final - self.initial) * span


@dataclass(frozen=True)
class Driver:
    """Self-oscillating bridge driver.

    The phase-led feedback signal drives a comparator; its zero crossings command
    the bridge polarity after a fixed propagation delay, with dead time inserted
    at every reversal. An interrupter gates bursts and a ramp sets the bus.
    """

    lead: PhaseLead
    delay: float = 0.0
    dead_time: float = 0.0
    interrupter: Interrupter | Melody | None = None
    ramp: Ramp | None = None
    bus: float = 0.0

    def sequencer(self, start=0.0):
        """Gate state machine for this driver, from time ``start``."""
        return GateSequencer(self, start)

    def edges(self, duration):
        """Interrupter transition times in (0, duration]; empty when ungated."""
        if self.interrupter is None:
            return np.zeros(0)
        return self.interrupter.edges(duration)


class GateSequencer:
    """Gate-command state machine driven by comparator crossings and burst edges.

    The integrator calls :meth:`crossing` when the phase-led feedback signal
    changes sign and :meth:`burst` at interrupter edges, then propagates only as
    far as :meth:`next_time` before calling :meth:`fire`.
    """

    def __init__(self, driver, start=0.0):
        self.driver = driver
        self._pending = deque()
        self._burst_start = start
        ungated = driver.interrupter is None
        self._enabled = ungated
        self._gate = 1 if ungated else 0

    @property
    def gate(self):
        """Present bridge command, -1, 0 or +1."""
        return self._gate

    @property
    def enabled(self):
        """Whether a burst is in progress."""
        return self._enabled

    @property
    def burst_start(self):
        """Start time of the burst the bus ramp is measured from."""
        return self._burst_start

    def next_time(self):
        """Time of the earliest scheduled transition, math.inf if none is pending."""
        return self._pending[0][0] if self._pending else math.inf

    def fire(self, t):
        """Apply every transition due at or before ``t`` and return the command."""
        while self._pending and self._pending[0][0] <= t:
            self._gate = self._pending.popleft()[1]
        return self._gate

    def crossing(self, t, sign):
        """Feedback crossed to ``sign`` at ``t``; command it after the gate delay.

        A crossing supersedes any pending command of the other polarity, as the
        gate signal follows the comparator rather than queueing stale commands.
        """
        if not self._enabled:
            return
        sign = 1 if sign > 0 else -1
        if sign == (self._pending[-1][1] if self._pending else self._gate):
            return
        self._pending.clear()
        if sign == self._gate:
            return
        target = t + self.driver.delay
        if self._gate != 0 and self.driver.dead_time > 0.0:
            self._pending.append((target, 0))
            target += self.driver.dead_time
        self._pending.append((target, sign))

    def burst(self, t, on):
        """Interrupter edge: seed a positive pulse on enable, force 0 on disable."""
        self._pending.clear()
        self._enabled = bool(on)
        if on:
            self._burst_start = t
        self._gate = 1 if on else 0

    def bus_voltage(self, t):
        """Bus voltage at ``t``, from the ramp measured since the burst start."""
        if self.driver.ramp is None:
            return self.driver.bus
        return float(self.driver.ramp.voltage(t - self._burst_start))
