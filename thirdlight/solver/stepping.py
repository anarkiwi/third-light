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
from thirdlight.solver.propagator import Propagator

_XTOL = 1e-14


@dataclass(frozen=True)
class Result:
    """Time history of a run, sampled at every step boundary and switching instant."""

    t: np.ndarray
    x: np.ndarray
    gate: np.ndarray
    device: np.ndarray
    drive: np.ndarray
    network: Network

    def __len__(self):
        return len(self.t)

    @property
    def primary_current(self):
        """Primary tank current, A."""
        return self.network.primary_current(self.x)

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


def _crossing(prop, x, u, c, d, span, sign):
    """First time in [0, span) at which c.x + d leaves the half plane of ``sign``.

    Both ends are evaluated through the propagator, as the root finder is, so a
    functional pinned to zero at a switching instant brackets consistently; that
    and a functional already on the far side are both due at once.
    """

    def value(s):
        return c @ prop.advance(x, u, s) + d

    start = value(0.0)
    if start * sign < 0.0:
        return 0.0
    if value(span) * sign >= 0.0:
        return math.inf
    return 0.0 if start == 0.0 else brentq(value, 0.0, span, xtol=span * _XTOL)


def _breakout(network, gate, x, u_load, v_bus):
    """Polarity a blocked bridge starts to conduct with, or 0 while it stays blocked.

    Each candidate sign fixes which devices conduct and hence the loop equation;
    the sign is admissible when the resulting di_p/dt agrees with it. Neither
    agreeing is the diode dead zone, where the loop voltage cannot forward bias
    anything.
    """
    for sign in (1.0, -1.0):
        index = network.state(gate, sign)[0]
        u = np.array([network.drive(gate, sign, v_bus), u_load])
        if (network.a[index] @ x + network.b @ u)[0] * sign > 0.0:
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


def simulate(network, driver, duration, step, load=None, x0=None):
    """Run ``network`` under ``driver`` for ``duration`` seconds at nominal ``step``.

    ``load`` is an optional ``(t, v_top) -> current`` injection at the top node,
    held over the interval; its RC time constant is far above the step, so the
    explicit coupling is stable.
    """
    props = [Propagator.build(a, network.b, step) for a in network.a]
    unit = np.zeros(network.size)
    unit[0] = 1.0
    seq = driver.sequencer()
    schedule = edge_t, _ = _burst_schedule(driver, duration)
    if driver.interrupter is not None:
        seq.burst(0.0, bool(driver.interrupter.active(0.0)))
    sign_i, sign_fb, edge = 0.0, 0.0, 0
    t = 0.0
    x = np.zeros(network.size) if x0 is None else np.array(x0, dtype=float)
    history = []
    while True:
        edge = _synchronise(seq, t, edge, schedule)
        gate = seq.gate
        v_bus = seq.bus_voltage(t)
        current = 0.0 if load is None else load(t, float(network.top_voltage(x)))
        if sign_i == 0.0:
            sign_i = _breakout(network, gate, x, current, v_bus)
        device = network.state(gate, sign_i)[0]
        prop = props[device]
        u = np.array([network.drive(gate, sign_i, v_bus), current])
        history.append((t, x, gate, device, u[0]))
        if t >= duration:
            break
        horizon = min(t + step, duration, seq.next_time())
        if edge < edge_t.size:
            horizon = min(horizon, edge_t[edge])
        span = min(horizon - t, step)
        if span <= 0.0:
            t = np.nextafter(t, math.inf)
            continue
        lead = driver.lead.functional(network.a[device], network.b, u)
        hit_i = (
            math.inf
            if sign_i == 0.0
            else _crossing(prop, x, u, unit, 0.0, span, sign_i)
        )
        if sign_fb != 0.0:
            hit_fb = _crossing(prop, x, u, lead[0], lead[1], span, sign_fb)
            if hit_fb < min(span, hit_i):
                sign_fb = -sign_fb
                seq.crossing(t + hit_fb, sign_fb)
                span = min(span, seq.next_time() - t)
        first = min(span, hit_i)
        x = prop.advance(x, u, first)
        t += first
        if first == hit_i:
            sign_i = 0.0
            x = x.copy()
            x[0] = 0.0
        elif sign_fb == 0.0:
            sign_fb = np.sign(lead[0] @ x + lead[1])
    times, states, gates, devices, drives = zip(*history)
    return Result(
        t=np.array(times),
        x=np.array(states),
        gate=np.array(gates, dtype=np.int8),
        device=np.array(devices, dtype=np.int8),
        drive=np.array(drives),
        network=network,
    )
