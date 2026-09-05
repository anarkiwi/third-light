"""Event-driven stepping of the bridge, primary tank and secondary modes.

The network is linear between switching instants, so each interval advances by an
exact propagator. Commutation and the comparator are affine functionals of the
state, so their crossings are located inside the step rather than quantised to it.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from thirdlight.circuit import Network
from thirdlight.circuit.devices import polarity
from thirdlight.solver.propagator import Propagator

_XTOL = 1e-14


@dataclass(frozen=True)
class Result:
    """Time history of a run, sampled at every step boundary and switching instant."""

    t: np.ndarray
    x: np.ndarray
    gate: np.ndarray
    state: np.ndarray
    u: np.ndarray
    network: Network

    def __len__(self):
        return len(self.t)

    @property
    def primary_current(self):
        """Primary tank current, A."""
        return self.network.primary_current(self.x)

    @property
    def bus_voltage(self):
        """DC bus voltage, the reservoir state or the supply that stood in for it, V."""
        return self.network.bus_voltage(self.x, self.u[:, 2])

    @property
    def drive(self):
        """Bridge output across the tank, less its differential resistance, V."""
        swing = polarity(self.state) * self.network.bridge.gain
        return swing * self.bus_voltage + self.u[:, 0]

    @property
    def tank_voltage(self):
        """Primary tank capacitor voltage, V."""
        return self.network.voltages(self.x)[..., 0]

    @property
    def top_voltage(self):
        """Top-load potential, the sum of the modal voltages, V."""
        return self.network.top_voltage(self.x)

    @property
    def energy(self):
        """Stored electric and magnetic energy, J."""
        return self.network.energy(self.x)


def _crossing(prop, ends, u, functional, span, sign):
    """First time in [0, span) at which c.x + d leaves the half plane of ``sign``.

    A functional pinned to zero at a switching instant, and one already on the
    far side because a switch made it discontinuous, are both due at once. Both
    ends of the span are supplied, since the caller needs the far one anyway.
    """
    x, x_end = ends
    c, d = functional
    start = c @ x + d
    if start * sign < 0.0:
        return 0.0
    if (c @ x_end + d) * sign >= 0.0:
        return math.inf
    if start == 0.0:
        return 0.0
    return brentq(lambda s: c @ prop.advance(x, u, s) + d, 0.0, span, xtol=span * _XTOL)


def _breakout(network, gate, x, u):
    """Polarity a blocked bridge starts to conduct with, or 0 while it stays blocked.

    Each candidate sign fixes which devices conduct and hence the loop equation;
    the sign is admissible when the resulting di_p/dt agrees with it. Neither
    agreeing is the diode dead zone, where the loop voltage forward biases nothing.
    """
    for sign in (1.0, -1.0):
        row = network.index(gate, sign)
        u[0] = network.offset(gate, sign)
        if (network.a[row] @ x + network.b[row] @ u)[0] * sign > 0.0:
            return sign
    return 0.0


def _synchronise(seq, t, edge, schedule):
    """Apply the gate transitions and burst edges due at ``t``, and return the edge index."""
    seq.fire(t)
    edge_t, edge_on = schedule
    while edge < edge_t.size and edge_t[edge] <= t:
        seq.burst(t, bool(edge_on[edge]))
        edge += 1
    return edge


def _burst_schedule(driver, duration):
    """Interrupter transition times, with the burst state each one leaves behind."""
    times = np.asarray(driver.edges(duration), dtype=float)
    if not times.size:
        return times, np.zeros(0, dtype=bool)
    return times, driver.interrupter.active(np.nextafter(times, math.inf))


def _horizon(seq, t, step, duration, schedule, edge):
    """Span the interval may run for: to the step, the run, a gate or a burst edge."""
    limit = min(t + step, duration, seq.next_time())
    edge_t = schedule[0]
    if edge < edge_t.size:
        limit = min(limit, edge_t[edge])
    return min(limit - t, step)


def simulate(network, driver, duration, step, load=None, x0=None):
    """Run ``network`` under ``driver`` for ``duration`` seconds at nominal ``step``.

    ``load`` is an optional ``(t, v_top) -> current`` injection at the top node,
    held over the interval; its RC time constant is far above the step, so the
    explicit coupling is stable.
    """
    props = [Propagator.build(a, b, step) for a, b in zip(network.a, network.b)]
    unit = np.eye(network.size)[0]
    seq = driver.sequencer()
    schedule = _burst_schedule(driver, duration)
    if driver.interrupter is not None:
        seq.burst(0.0, bool(driver.interrupter.active(0.0)))
    sign_i, sign_fb, edge = 0.0, 0.0, 0
    t = 0.0
    x = np.zeros(network.size) if x0 is None else np.array(x0, dtype=float)
    history = []
    while True:
        edge = _synchronise(seq, t, edge, schedule)
        gate = seq.gate
        current = 0.0 if load is None else load(t, float(network.top_voltage(x)))
        u = np.array([0.0, current, seq.bus_voltage(t)])
        if sign_i == 0.0:
            sign_i = _breakout(network, gate, x, u)
        row = network.index(gate, sign_i)
        prop = props[row]
        u[0] = network.offset(gate, sign_i)
        history.append((t, x, gate, row, u.copy()))
        if t >= duration:
            break
        span = _horizon(seq, t, step, duration, schedule, edge)
        if span <= 0.0:
            t = np.nextafter(t, math.inf)
            continue
        lead = driver.lead.functional(network.a[row], network.b[row], u)
        ends = x, prop.advance(x, u, span)
        hit_i = math.inf
        if sign_i != 0.0:
            hit_i = _crossing(prop, ends, u, (unit, 0.0), span, sign_i)
        if sign_fb != 0.0:
            hit_fb = _crossing(prop, ends, u, lead, span, sign_fb)
            if hit_fb < min(span, hit_i):
                sign_fb = -sign_fb
                seq.crossing(t + hit_fb, sign_fb)
                shortened = min(span, seq.next_time() - t)
                if shortened < span:
                    span, ends = shortened, (x, prop.advance(x, u, shortened))
        first = min(span, hit_i)
        x = ends[1] if first == span else prop.advance(x, u, first)
        t += first
        if first == hit_i:
            sign_i = 0.0
            x = x.copy()
            x[0] = 0.0
        elif sign_fb == 0.0:
            sign_fb = np.sign(lead[0] @ x + lead[1])
    times, states, gates, rows, inputs = zip(*history)
    return Result(
        t=np.array(times),
        x=np.array(states),
        gate=np.array(gates, dtype=np.int8),
        state=np.array(rows, dtype=np.int8),
        u=np.array(inputs),
        network=network,
    )
