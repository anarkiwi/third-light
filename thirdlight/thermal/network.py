"""Thermal networks, junction temperature and the between-burst update.

Cauer ladders and Foster chains are one state space, C dT/dt = -G T + S p,
differing only in how the rungs assemble; both propagate exactly by the matrix
exponential, and the map over an interrupter cycle is affine. See §3.6 of design.
"""

import math
from dataclasses import dataclass, fields, replace

import numpy as np

from thirdlight.circuit import from_modes
from thirdlight.discharge import calibration
from thirdlight.em import losses
from thirdlight.solver.propagator import Propagator
from thirdlight.thermal.ledger import ledger

AMBIENT = 25.0
PORTS = ("igbt", "diode", "coil", "capacitor")
FEEDBACK = ("igbt", "coil")
WINDOWS = 16
PASSES = 6
RTOL = 1e-3
REST = 1


@dataclass(frozen=True)
class Rungs:
    """One thermal branch, from what dissipates towards ambient.

    Cauer rungs are physical: capacitance i is node i's heat capacity and
    resistance i carries heat to the next node. Foster rungs are the fitted Zth a
    datasheet publishes, all carrying the same flow, so only their sum is a rise.
    """

    resistances: tuple = ()
    capacitances: tuple = ()
    foster: bool = False

    def __post_init__(self):
        if len(self.resistances) != len(self.capacitances):
            raise ValueError("a thermal branch needs one capacitance per resistance")
        if any(value <= 0.0 for value in self.resistances + self.capacitances):
            raise ValueError("thermal resistances and capacitances must be positive")

    def __len__(self):
        return len(self.resistances)

    @property
    def total(self):
        """Steady thermal resistance of the branch, sum R, K/W."""
        return float(np.sum(self.resistances))

    @classmethod
    def from_dict(cls, spec):
        """Branch from a mapping; ``time_constants`` stand in for capacitances."""
        spec = dict(spec)
        resistances = tuple(float(value) for value in spec.pop("resistances", ()))
        constants = spec.pop("time_constants", None)
        if constants is not None:
            spec["capacitances"] = np.asarray(constants, dtype=float) / resistances
        return cls(
            resistances=resistances,
            capacitances=tuple(float(value) for value in spec.pop("capacitances", ())),
            **spec,
        )


def _append(rungs, capacitance, conductance, below, at):
    """Add ``rungs`` standing on the terminal ``below``, and return their own terminal.

    A terminal reads the temperature at the top of a branch and, transposed,
    spreads a flow entering there. Every rung is an edge adding g (e-f)(e-f)^T to
    G: Cauer between successive nodes and last onto ``below``, Foster onto ground.
    """
    if not rungs:
        return below
    unit = np.eye(len(capacitance))[at : at + len(rungs)]
    capacitance[at : at + len(rungs)] = rungs.capacitances
    edges = (
        unit if rungs.foster else np.vstack([unit[:-1] - unit[1:], unit[-1] - below])
    )
    conductance += edges.T @ (edges / np.asarray(rungs.resistances)[:, None])
    return below + unit.sum(axis=0) if rungs.foster else unit[0]


@dataclass(frozen=True)
class Model:
    """Thermal state space C dT/dt = -G T + S p of ports dissipating towards ambient.

    ``injection`` maps a port's dissipation onto the states it enters and,
    transposed, reads that port's temperature back out of them, so an energy E
    is the state jump C^-1 S E and the temperature is ambient + S^T T.
    """

    capacitance: np.ndarray
    conductance: np.ndarray
    injection: np.ndarray
    ports: tuple
    ambient: float = AMBIENT

    @property
    def size(self):
        """Number of thermal states."""
        return len(self.capacitance)

    @property
    def a(self):
        """State matrix -C^-1 G of dT/dt = A T + B p."""
        return -self.conductance / self.capacitance[:, None]

    @property
    def b(self):
        """Input matrix C^-1 S, the rise per unit power held."""
        return self.injection / self.capacitance[:, None]

    def temperature(self, state):
        """Port temperatures of a state or a history of states, C."""
        return self.ambient + np.asarray(state) @ self.injection

    def equilibrium(self, power):
        """State under constant port powers, G^-1 S p, the DC limit of the network."""
        return np.linalg.solve(self.conductance, self.injection @ np.asarray(power))

    def propagator(self, step):
        """Exact propagator of the network over intervals up to ``step``."""
        return Propagator.build(self.a, self.b, step)


def assemble(*groups, ports=PORTS, ambient=AMBIENT):
    """State space of ``(branches, path)`` groups, each sharing one path to ambient.

    Every branch is a port, in the order given; the path is what its branches
    stand on in common, such as the case and sink of a module carrying two dies.
    An empty branch is a short, reporting the temperature it stands on.
    """
    branches = [rungs for group, _ in groups for rungs in group]
    if len(branches) != len(ports):
        raise ValueError(f"{len(branches)} branches against {len(ports)} ports")
    size = sum(map(len, branches)) + sum(len(path) for _, path in groups)
    if size == 0:
        raise ValueError("the thermal networks are empty")
    capacitance = np.zeros(size)
    conductance = np.zeros((size, size))
    injection = np.zeros((size, len(branches)))
    at = port = 0
    for group, path in groups:
        below = _append(path, capacitance, conductance, np.zeros(size), at)
        at += len(path)
        for rungs in group:
            injection[:, port] = _append(rungs, capacitance, conductance, below, at)
            at, port = at + len(rungs), port + 1
    return Model(capacitance, conductance, injection, tuple(ports), ambient)


@dataclass(frozen=True)
class Stack:
    """A machine's thermal networks: two dies over a shared sink, the coil, the tank.

    ``igbt`` and ``diode`` are junction-to-case impedances of the devices in one
    bridge leg, standing on the ``sink`` path their module shares; ``coil`` and
    ``capacitor`` stand on ambient themselves.
    """

    igbt: Rungs = Rungs()
    diode: Rungs = Rungs()
    sink: Rungs = Rungs()
    coil: Rungs = Rungs()
    capacitor: Rungs = Rungs()
    ambient: float = AMBIENT

    @property
    def model(self):
        """Assembled state space over :data:`PORTS`."""
        return assemble(
            ((self.igbt, self.diode), self.sink),
            ((self.coil,), Rungs()),
            ((self.capacitor,), Rungs()),
            ambient=self.ambient,
        )

    @classmethod
    def from_dict(cls, spec, **branches):
        """Stack from the ``thermal`` mapping, the device impedances given apart."""
        given = {
            name: Rungs.from_dict(value) if isinstance(value, dict) else value
            for name, value in dict(spec).items()
        }
        given.update(
            (name, Rungs.from_dict(value))
            for name, value in branches.items()
            if value is not None
        )
        return cls(**given)


def windows(result, count, until=None):
    """Slices of a run at ``count`` equal steps over [t0, ``until``], out to its end.

    Consecutive windows share their boundary sample, so every interval closes in
    exactly one of them and the windows sum to the run. ``until`` is where the
    step is measured against: the burst's own end, so a boundary lands on it.
    """
    start, stop = float(result.t[0]), float(result.t[-1])
    step = ((stop if until is None else float(until)) - start) / count
    reach = math.ceil((stop - start) / step) + 1
    edges = np.searchsorted(
        result.t, np.unique(np.minimum(start + step * np.arange(reach), stop))
    )
    edges[0], edges[-1] = 0, len(result) - 1
    arrays = [
        field.name
        for field in fields(result)
        if isinstance(getattr(result, field.name), np.ndarray)
    ]
    return [
        replace(
            result,
            **{name: getattr(result, name)[first : last + 1] for name in arrays},
        )
        for first, last in zip(edges[:-1], edges[1:])
    ]


def sources(entry):
    """Energy each port of :data:`PORTS` takes from a loss ledger, J.

    The IGBTs carry their conduction and both hard-switching terms, the diodes
    theirs and the recovery; winding and former heat the one coil. The primary
    loop is busbar and the channel is in the air, so neither has a network.
    """
    switched = entry.switching
    return np.array(
        [
            entry.igbt + float(switched.on.sum() + switched.off.sum()),
            entry.diode + float(switched.rr.sum()),
            float(np.sum(entry.winding)) + entry.former,
            entry.esr,
        ]
    )


def energies(result, count=WINDOWS, tj=None, until=None):
    """Duration and port energies of each window of a run, as ``(spans, joules)``.

    A window holding one sample closes no interval and carries no energy, so it
    is dropped rather than propagated over nothing.
    """
    parts = [part for part in windows(result, count, until) if part.t[-1] > part.t[0]]
    return (
        np.array([part.t[-1] - part.t[0] for part in parts]),
        np.array([sources(ledger(part, tj)) for part in parts]),
    )


@dataclass(frozen=True)
class Steady:
    """The settled interrupter cycle of a thermal model, sampled at its segments.

    ``converged`` is false when the loss and temperature loop that produced it
    had not stopped moving, which a coil in thermal runaway never does.
    """

    t: np.ndarray
    state: np.ndarray
    power: np.ndarray
    model: Model
    bursts: int
    converged: bool = True

    @property
    def temperature(self):
        """Port temperature at each sample of the cycle, C."""
        return self.model.temperature(self.state)

    @property
    def mean(self):
        """Cycle-mean port temperature, C.

        Exact rather than quadrature: over a settled cycle the stored heat
        returns to what it was, so G times the mean state is S times the mean
        power, and the mean is the DC state at that power.
        """
        return self._named(self.model.temperature(self.model.equilibrium(self.power)))

    @property
    def peak(self):
        """Highest port temperature of the settled cycle, C, which is what a die survives."""
        return self._named(self.temperature.max(axis=0))

    @property
    def ripple(self):
        """Within-cycle swing of each port, K."""
        return self._named(np.ptp(self.temperature, axis=0))

    @property
    def burst(self):
        """Mean port temperature over the burst, C, where the loss fits are evaluated."""
        during = self.temperature[: self.bursts + 1]
        weight = np.diff(self.t[: self.bursts + 1])
        return self._named(0.5 * (during[:-1] + during[1:]).T @ weight / weight.sum())

    def _named(self, values):
        """Port values as a mapping keyed by port name."""
        return dict(zip(self.model.ports, np.asarray(values)))


def steady(model, spans, joules, gap, rest=REST):
    """Settled cycle of a network driven by one burst per interrupter period.

    ``joules`` is what each port takes over each sub-interval of a burst, of
    ``spans`` seconds each, and ``gap`` the quiescent time after it, reported in
    ``rest`` samples. The affine cycle map is solved, not iterated.
    """
    joules = np.atleast_2d(np.asarray(joules, dtype=float))
    spans = np.broadcast_to(np.asarray(spans, dtype=float), (len(joules),))
    if spans.min() <= 0.0 or gap <= 0.0:
        raise ValueError("the cycle needs positive sub-intervals and a positive gap")
    quiet = gap / rest
    propagator = model.propagator(max(spans.max(), quiet))
    pairs = {span: propagator.at(span) for span in set(spans.tolist()) | {quiet}}
    segments = [(pairs[span], row / span) for span, row in zip(spans, joules)]
    segments += [(pairs[quiet], np.zeros(joules.shape[1]))] * rest
    phi, q = np.eye(model.size), np.zeros(model.size)
    for (advance, drive), power in segments:
        phi, q = advance @ phi, advance @ q + drive @ power
    history = [np.linalg.solve(np.eye(model.size) - phi, q)]
    for (advance, drive), power in segments:
        history.append(advance @ history[-1] + drive @ power)
    burst = spans.sum()
    return Steady(
        t=np.concatenate(
            [[0.0], np.cumsum(spans), burst + quiet * np.arange(1, rest + 1)]
        ),
        state=np.array(history),
        power=joules.sum(axis=0) / (burst + gap),
        model=model,
        bursts=len(joules),
    )


def _retuned(machine, temperature):
    """The machine with its winding resistance re-evaluated at ``temperature``.

    Only the modal Q moves; the eigen-solve, the coupling and the primary
    inductance are the geometry's and are read back off the network the machine
    carries, so a rebuild costs one resistance sweep and the assembly.
    """
    net = machine.network
    primary = float(net.inductances[0, 0])
    return replace(
        machine,
        network=from_modes(
            machine.eigen,
            net.inductances[0, 1:] / np.sqrt(primary * machine.eigen.l_m),
            losses.quality_factor(
                machine.design, machine.eigen, temperature=temperature
            ),
            primary,
            machine.tank,
            machine.bridge,
            machine.bus,
            net.loss_tangent,
        ),
    )


def _converged(before, after, ambient, rtol):
    """Whether a pass returned the temperatures it was run at, relative to ``rtol``."""
    return all(
        abs(after[port] - before[port]) <= rtol * max(abs(after[port] - ambient), 1.0)
        for port in after
    )


def _secant(pairs, ports=FEEDBACK):
    """Temperature at which a pass would return what it was run at, per port.

    The outer map is affine to the order of the fits' own coefficients, so the
    secant through the last two passes' residuals crosses zero at its fixed
    point. A loop gain of one or more has none, and the plain iterate stands.
    """
    ends = np.array([[step[port] for port in ports] for step, _ in pairs[-2:]])
    residual = (
        np.array([[step[port] for port in ports] for _, step in pairs[-2:]]) - ends
    )
    move, slope = ends[1] - ends[0], residual[1] - residual[0]
    gain = np.divide(slope, move, out=np.ones_like(slope), where=move != 0.0)
    step = np.divide(
        -residual[1] * move, slope, out=residual[1].copy(), where=gain < 0.0
    )
    return dict(zip(ports, ends[1] + step))


def equilibrium(
    machine,
    streamer=None,
    count=WINDOWS,
    passes=PASSES,
    rtol=RTOL,
    guess=None,
    rest=REST,
    tail=5.0,
):
    """Settled temperatures of a machine's repeated interrupter cycle.

    Losses depend on temperature and temperature on losses, so each pass runs one
    burst at the last pass's :data:`FEEDBACK` temperatures and solves the cycle in
    closed form. ``guess`` starts it off ambient, which its fixed point does not
    depend on; a coil in thermal runaway has none, and comes back not converged.
    """
    model = machine.thermal.model
    interrupter = machine.driver.interrupter
    gap = interrupter.period - calibration.span(machine, tail)
    hot = None if guess is None else dict.fromkeys(FEEDBACK, float(guess))
    length, settled, pairs = 0.0, None, []
    for _ in range(passes):
        tuned = machine if hot is None else _retuned(machine, hot["coil"])
        result = calibration.burst(tuned, streamer, length, tail)
        junction = None if hot is None else hot["igbt"]
        spans, joules = energies(result, count, junction, interrupter.on_time)
        settled = steady(model, spans, joules, gap, rest)
        if streamer is not None:
            length = float(result.length[-1] * math.exp(-gap / streamer.cooling))
        after = {port: settled.burst[port] for port in FEEDBACK}
        if hot is not None:
            if _converged(hot, after, model.ambient, rtol):
                return settled
            pairs.append((hot, after))
        hot = after if len(pairs) < 2 else _secant(pairs)
    return replace(settled, converged=False)
