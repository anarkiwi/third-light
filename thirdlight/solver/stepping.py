"""Event-driven stepping of the bridge, primary tank and secondary modes.

The network is linear between switching instants, so each interval advances by an
exact propagator. Commutation and the comparator are affine functionals of the
state, so their crossings are located inside the step rather than quantised to it.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from thirdlight.circuit import Network, with_streamer
from thirdlight.circuit.devices import polarity
from thirdlight.solver.propagator import Propagator
from thirdlight.thermal.ledger import branch_energies, integrate, ledger

_XTOL = 1e-14


@dataclass(frozen=True)
class Result:  # pylint: disable=too-many-instance-attributes
    """Time history of a run, sampled at every step boundary and switching instant."""

    t: np.ndarray
    x: np.ndarray
    gate: np.ndarray
    state: np.ndarray
    u: np.ndarray
    network: Network
    length: np.ndarray
    channel: np.ndarray
    loss: np.ndarray
    resistance: np.ndarray | None = None

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
        """Stored electric and magnetic energy, J.

        The streamer's channel capacitance follows its length, so its stored
        energy is corrected off the one the network was last built with.
        """
        stored = self.network.energy(self.x)
        if self.network.streamer is None:
            return stored
        voltage = self.network.streamer_voltage(self.x)
        return stored + 0.5 * (self.channel - self.network.streamer[1]) * voltage**2

    @property
    def streamer_current(self):
        """Current drawn from the top node by the streamer, A.

        A grown channel's resistance moves with it, so the current is taken at
        the resistance each sample was stepped at.
        """
        if self.network.streamer is None or self.resistance is None:
            return self.network.streamer_current(self.x)
        top = self.network.top_voltage(self.x) - self.network.streamer_voltage(self.x)
        return top / self.resistance

    @property
    def streamer_power(self):
        """Power dissipated in the streamer channel resistance, W."""
        if self.network.streamer is None:
            return np.zeros_like(self.t)
        ohmic = self.network.streamer[0] if self.resistance is None else self.resistance
        return ohmic * self.streamer_current**2

    @property
    def channel_energies(self):
        """Energy the channel resistance dissipates over each interval, J.

        Exact for the branch's own first-order dynamics rather than a trapezoid
        of its power, which cannot resolve a branch faster than the step; see
        §3.5. A run whose branch was never built falls back to the trapezoid.
        """
        power = np.asarray(self.streamer_power, dtype=float)
        if self.network.streamer is None or self.resistance is None:
            return 0.5 * (power[:-1] + power[1:]) * np.diff(self.t)
        top = self.network.top_voltage(self.x)
        drop = top - self.network.streamer_voltage(self.x)
        return branch_energies(self.t, drop, top, self.resistance, self.channel)

    @property
    def _swing(self):
        """Bridge output voltage before the device drop, held over each interval, V."""
        return polarity(self.state) * self.network.bridge.gain * self.bus_voltage

    @property
    def input_energy(self):
        """Energy drawn from the bus over the run, J."""
        return integrate(self.t, self._swing, self.primary_current)

    def losses(self, tj=None):
        """Component-resolved energy ledger, and the switching energy beside it.

        :attr:`dissipation` is its conduction total; the switching energy at
        junction temperature ``tj`` is additive to that, since the state space
        carries no transition dynamics. See :mod:`thirdlight.thermal`.
        """
        return ledger(self, tj)

    @property
    def dissipation(self):
        """Energy lost to the loop resistances, the conducting devices and the streamer.

        Every weight -- which devices conduct, which bridge polarity stands
        across the tank -- is held over the interval that follows the sample it
        was recorded at, so the integrand is trapezoidal within an interval and
        discontinuous only between them. The channel is the exception, and is
        integrated exactly; see :attr:`channel_energies`.
        """
        i = self.network.currents(self.x)
        ohmic = self.network.resistances[self.state]
        loops = integrate(self.t, ohmic[:-1], i * i, sums=True)
        devices = integrate(self.t, -self.u[:, 0], self.primary_current)
        channel = float(self.channel_energies.sum())
        return loops + devices + channel + float(self.loss[-1] - self.loss[0])


def _bracket(value, span, sign, halvings=60):
    """Largest dyadic fraction of ``span`` at which ``value`` still holds ``sign``.

    Used where a functional starts on its own zero: the crossing wanted is the one
    it returns to, not the one it is leaving, so the search has to begin inside
    the half plane rather than on its boundary. Zero if it never enters.
    """
    s = span
    for _ in range(halvings):
        if value(s) * sign > 0.0:
            return s
        s *= 0.5
    return 0.0


def _crossing(prop, ends, u, functional, span, sign):
    """First time in [0, span) at which c.x + d leaves the half plane of ``sign``.

    A functional already on the far side because a switch made it discontinuous
    is due at once. One sitting exactly on its zero is not: the primary current
    is pinned there by the commutation that just happened, and the polarity the
    bridge took leaves it in the admissible direction, so the crossing due is the
    one it comes back to. Returning that instant as zero would leave the step
    unable to advance at all. Both ends of the span are supplied, since the
    caller needs the far one anyway.
    """
    x, x_end = ends
    c, d = functional
    start = c @ x + d
    if start * sign < 0.0:
        return 0.0
    if (c @ x_end + d) * sign >= 0.0:
        return math.inf
    value = prop.evaluator(x, u, functional)
    low = 0.0 if start != 0.0 else _bracket(value, span, sign)
    if start == 0.0 and low == 0.0:
        return 0.0
    return brentq(value, low, span, xtol=span * _XTOL)


def _conduction(network, gate, x, u):
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


def _propagators(network, step):
    """Propagator of each bridge state of a network."""
    return [Propagator.build(a, b, step) for a, b in zip(network.a, network.b)]


class _Channel:  # pylint: disable=too-many-instance-attributes
    """A channel model's state, and the network stage its level calls for.

    The model owns its state and quantises it into levels whose capacitance and
    resistance the branch is built at, so a run visits a few hundred stages and
    revisits them as the channel decays; each is diagonalised once and kept.
    Without a model this is the one stage the network was handed in.
    """

    def __init__(self, network, step, model, seed=0.0, rng=None):
        self.model = model
        self.step = step
        self.base = network
        self.stages = {}
        self.state = None if model is None else model.initial(seed, rng)
        self.loss = 0.0
        self.level = 0 if model is None else model.level(self.state)
        self.network, self.props = self.stage(self.level)

    def stage(self, level):
        """The (network, propagators) pair of a quantised channel level."""
        if self.model is None:
            return self.base, _propagators(self.base, self.step)
        if level not in self.stages:
            grown = with_streamer(
                self.base,
                self.model.resistance_at(level),
                self.model.capacitance_at(level),
            )
            self.stages[level] = (grown, _propagators(grown, self.step))
        return self.stages[level]

    @property
    def length(self):
        """Scalar spark length of the present state, what a photograph measures, m."""
        return 0.0 if self.model is None else self.model.extent(self.state)

    @property
    def capacitance(self):
        """Channel capacitance the network is presently built at, F."""
        return 0.0 if self.model is None else self.network.streamer[1]

    @property
    def resistance(self):
        """Channel resistance the network is presently built at, ohm."""
        return 0.0 if self.model is None else self.network.streamer[0]

    def retune(self, x):
        """Rebuild at the present length's level, and return the state it leaves.

        A channel that has grown carries its charge into the longer channel, so
        q is continuous across the change and the voltage falls; one that has
        cooled takes the charge of the part that recombined with it, so the
        voltage is what carries over. Either way the stored energy falls, by the
        work of extending the channel or with the charge the cooled part removed,
        and neither direction creates any.
        """
        if self.model is None or self.model.level(self.state) == self.level:
            return x
        before = self.capacitance
        self.level = self.model.level(self.state)
        self.network, self.props = self.stage(self.level)
        after = self.capacitance
        x = x.copy()
        stored = 0.5 * before * x[-1] ** 2
        if after > before:
            x[-1] *= before / after
        self.loss += stored - 0.5 * after * x[-1] ** 2
        return x

    def grow(self, x, voltage, dt):
        """Advance the state over ``dt`` with the top voltage held at ``voltage``."""
        if self.model is None:
            return
        margin = self.model.breakout.margin(self.network.voltages(x)[1:])
        current = float(self.network.streamer_current(x))
        self.state = self.model.advance(self.state, voltage, margin, dt, current)


def simulate(  # pylint: disable=too-many-statements
    network,
    driver,
    duration,
    step,
    load=None,
    x0=None,
    streamer=None,
    length0=0.0,
    rng=None,
):
    """Run ``network`` under ``driver`` for ``duration`` seconds at nominal ``step``.

    ``load`` is an optional ``(t, v_top) -> current`` injection at the top node,
    held over the interval. A ``streamer`` instead adds a state: the channel
    branch enters the state space and its length follows the top voltage, so the
    loading it applies is exact at any length rather than held across a step.
    ``x0`` may be shorter than the state, which zero-fills the rest, so a run can
    be seeded from one made without a streamer, and ``length0`` seeds the channel
    with what the last burst left behind. A grown channel takes ``rng``, which is
    the whole of its randomness, so a run at a fixed seed repeats exactly.
    """
    channel = _Channel(network, step, streamer, length0, rng)
    network, props = channel.network, channel.props
    unit = np.eye(network.size)[0]
    seq = driver.sequencer()
    schedule = _burst_schedule(driver, duration)
    if driver.interrupter is not None:
        seq.burst(0.0, bool(driver.interrupter.active(0.0)))
    sign_i, sign_fb, edge = 0.0, 0.0, 0
    t = 0.0
    x = np.zeros(network.size)
    if x0 is not None:
        x[: len(x0)] = x0
    history = []
    while True:
        edge = _synchronise(seq, t, edge, schedule)
        gate = seq.gate
        top = float(network.top_voltage(x))
        current = 0.0 if load is None else load(t, top)
        u = np.array([0.0, current, seq.bus_voltage(t)])
        x = channel.retune(x)
        network, props = channel.network, channel.props
        if sign_i == 0.0:
            sign_i = _conduction(network, gate, x, u)
        row = network.index(gate, sign_i)
        prop = props[row]
        u[0] = network.offset(gate, sign_i)
        history.append(
            (
                t,
                x,
                gate,
                row,
                u.copy(),
                channel.length,
                channel.capacitance,
                channel.loss,
                channel.resistance,
            )
        )
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
        channel.grow(x, top, first)
        t += first
        if first == hit_i:
            sign_i = 0.0
            x = x.copy()
            x[0] = 0.0
        elif sign_fb == 0.0:
            sign_fb = np.sign(lead[0] @ x + lead[1])
    times, states, gates, rows, inputs, lengths, channels, losses, ohmic = zip(*history)
    return Result(
        t=np.array(times),
        x=np.array(states),
        gate=np.array(gates, dtype=np.int8),
        state=np.array(rows, dtype=np.int8),
        u=np.array(inputs),
        network=network,
        length=np.array(lengths),
        channel=np.array(channels),
        loss=np.array(losses),
        resistance=np.array(ohmic),
    )
