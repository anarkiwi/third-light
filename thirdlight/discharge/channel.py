"""The grown tree as a channel model on the time-domain solver, in place of a length.

The tree of 3.4c presented as the stepper's channel model, its load from 3.4b's
mixed solve rather than a capacitance per unit length and a lumped 220 kOhm.
Growth is clocked at a leader velocity, cooling prunes; see docs/design.md 3.4d.
"""

import math
from dataclasses import dataclass

import numpy as np

from thirdlight.discharge.dbm import Discharge
from thirdlight.discharge.filament import path_resistance, series_resistance
from thirdlight.discharge.streamer import (
    COOLING_TIME,
    FRITZ_RESISTANCE,
    GROWTH_GAIN,
    SUSTAINING_GRADIENT,
)
from thirdlight.em.capacitance import lumped_capacitance
from thirdlight.geometry import Toroid

CHANNEL_STEPS = 512


@dataclass
class State:
    """Growth state of one channel: its tree, its growth clock and its thermal reach.

    ``reach`` is the extent the channel stays hot enough to hold and ``budget``
    the fraction of a segment the growth clock carries between circuit steps.
    """

    discharge: Discharge
    reach: float = 0.0
    budget: float = 0.0
    measure: tuple | None = None

    @property
    def tree(self):
        """The tree grown so far."""
        return self.discharge.tree

    @property
    def distances(self):
        """Running maximum of the distance from the root, nondecreasing in node order."""
        nodes = self.discharge.nodes
        return np.maximum.accumulate(np.linalg.norm(nodes - nodes[0], axis=1))


class TreeChannel:  # pylint: disable=too-many-instance-attributes
    """A DBM-grown tree presented as the stepper's channel model.

    ``resistivity`` defaults to the one putting Fritz's 220 kOhm on a metre of
    channel, his lumped value re-expressed as a gradient; ``gradient`` is the
    critical propagation field in V/m, so :attr:`Growth.critical` is unused.
    """

    def __init__(
        self,
        rings,
        breakout,
        frequency,
        seed,
        direction,
        growth,
        rng=None,
        bodies=(),
        ground=True,
        steps=CHANNEL_STEPS,
        resistivity=None,
        gradient=SUSTAINING_GRADIENT,
        cooling=COOLING_TIME,
        velocity=GROWTH_GAIN,
        resistance=FRITZ_RESISTANCE,
        tolerance=0.02,
        floor=1.0e-3,
    ):
        self.rings = rings
        self.breakout = breakout
        self.frequency = frequency
        self.seed = seed
        self.direction = direction
        self.growth = growth
        self.rng = rng
        self.bodies = tuple(bodies)
        self.ground = ground
        self.steps = steps
        self.gradient = gradient
        self.cooling = cooling
        self.velocity = velocity
        self.resistance = resistance
        self.tolerance = tolerance
        self.floor = floor
        self.resistivity = (
            math.pi * growth.radius**2 * resistance
            if resistivity is None
            else resistivity
        )
        self._bare = lumped_capacitance(rings, ground)
        self._resistance = {}

    @property
    def minimum(self):
        """Capacitance the branch is held at while it is immaterial, F.

        The scalar model's own floor, at the nominal resistance the held branch
        is built with, which is the origin of the capacitance ladder above it.
        """
        return self.floor / (2.0 * math.pi * self.frequency * self.resistance)

    @property
    def resolution(self):
        """Seed length a burst cannot tell from none, which is one growth step."""
        return self.growth.step

    def admittance(self, resistance, capacitance):
        """Branch admittance as a fraction of its own 1 / R, omega R C.

        The floor is a statement about the branch, so it is taken against the
        resistance the branch is actually built at rather than a nominal one:
        below ``floor`` of that the channel neither loads nor detunes the top
        node, and is held.
        """
        return 2.0 * math.pi * self.frequency * resistance * capacitance

    def initial(self, seed=0.0, rng=None):
        """Channel state a run starts from: a state carries over, a length seeds the reach.

        The run's generator wins over the model's own, and fresh entropy stands
        in for neither, which is a run that does not repeat.
        """
        if isinstance(seed, State):
            return seed
        discharge = Discharge(
            self.rings,
            self.seed,
            self.direction,
            self.growth,
            self.steps,
            np.random.default_rng(self.rng if rng is None else rng),
            self.bodies,
            self.ground,
        )
        return State(discharge=discharge, reach=float(seed))

    def extent(self, state):
        """Greatest distance from the root, the scalar spark length of a tree, m."""
        return float(state.distances[-1])

    def level(self, state):
        """Quantised capacitance level of a state, geometric in ``tolerance``."""
        return self._measure(state)[0]

    def branch(self, state):
        """Added capacitance in F and series resistance in ohm of a state's own tree."""
        return self._measure(state)[1:]

    def capacitance_at(self, level):
        """Channel capacitance of a quantised level, F."""
        return self.minimum * (1.0 + self.tolerance) ** level

    def resistance_at(self, level):
        """Series resistance of a quantised level, ohm.

        3.4b's reduction of the tree's own segment resistances, recorded when a
        level is first reached, so a channel decaying back through its levels
        finds the branch it was built with.
        """
        return self._resistance.get(level, self.resistance)

    def advance(self, state, voltage, margin, dt, current=0.0):
        """Grow at the leader velocity ``voltage`` sets, then cool by pruning.

        Initiation needs the electrode surface to reach Peek's threshold. Growth
        that stalls is field limited rather than clock limited, so the carried
        remainder is dropped rather than spent on the step after. A tip refreshes
        the reach; the tree then follows it down to within half a growth step.
        """
        voltage = abs(voltage)
        grew = False
        if len(state.discharge) > 1 or margin >= 1.0:
            state.budget += self.velocity * voltage * dt / self.growth.step
            while state.budget >= 1.0:
                state.budget -= 1.0
                if not self._extend(state.discharge, voltage, abs(current)):
                    state.budget = 0.0
                    break
                state.measure, grew = None, True
        distances = state.distances
        if grew:
            state.reach = max(state.reach, float(distances[-1]))
        state.reach *= math.exp(-dt / self.cooling)
        keep = int(
            np.searchsorted(
                distances, state.reach + 0.5 * self.growth.step, side="right"
            )
        )
        if keep < distances.size:
            state.discharge.prune(keep)
            state.measure = None
        return state

    def potential(self, tree, voltage, current):
        """Node potentials driving the growth rule: the drive less each path's own drop.

        A drop past the drive is a branch the channel cannot hold at all, and is
        clamped rather than left to draw charge of the opposite sign.
        """
        drop = abs(current) * path_resistance(tree, self.resistivity)
        return np.maximum(abs(voltage) - drop, 0.0)

    def _extend(self, discharge, voltage, current):
        """One growth step under the tip potentials the path resistance leaves."""
        return discharge.step(self._field(discharge, voltage, current), self.gradient)

    def _field(self, discharge, voltage, current):
        """Mean field each live candidate would be crossed at, in V/m.

        Each segment sits at the mean of its own endpoints, the point its charge
        is matched at.
        """
        tree = discharge.tree
        count = len(self.rings)
        node = self.potential(tree, voltage, current)
        source = np.empty(count + tree.segments)
        source[:count] = voltage
        source[count:] = 0.5 * (node[1:] + node[tree.parent[1:]])
        potential = discharge.potentials(discharge.charges(source))
        return (node[discharge.roots] - potential) / self.growth.step

    def _measure(self, state):
        """Level, added capacitance and series resistance of a state, cached per change.

        The channel is held at the floor until its own omega R C reaches
        ``floor``; the ladder above that is geometric in the capacitance, which
        is what a level has to determine.
        """
        if state.measure is None:
            tree, count = state.tree, len(self.rings)
            charges = state.discharge.charges()
            added = max(float(charges.sum()) - self._bare, 0.0)
            ohmic = (
                self.resistance
                if tree.segments == 0
                else series_resistance(tree, charges[count:], self.resistivity)
            )
            level = (
                0
                if self.admittance(ohmic, added) <= self.floor
                else max(
                    round(math.log(added / self.minimum) / math.log1p(self.tolerance)),
                    0,
                )
            )
            self._resistance.setdefault(level, ohmic if level else self.resistance)
            state.measure = (level, added, ohmic)
        return state.measure

    @classmethod
    def from_design(cls, design, breakout, frequency, growth, **kwargs):
        """Channel model rooted on the top-node electrode of a design.

        The root is the breakout point's pole where there is one, that being
        where the surface field is, and the top load's outer equator otherwise.
        """
        point = design.breakout
        if point is not None:
            seed, direction = (0.0, 0.0, point.height + point.radius), (0.0, 0.0, 1.0)
        elif design.top_load is not None:
            load = design.top_load
            reach = (
                load.major_radius + load.minor_radius
                if isinstance(load, Toroid)
                else load.radius
            )
            seed, direction = (reach, 0.0, load.height), (1.0, 0.0, 0.0)
        else:
            raise ValueError("a channel needs a top-node electrode to grow off")
        return cls(
            design.top_load_rings(),
            breakout,
            frequency,
            seed,
            direction,
            growth,
            bodies=design.electrodes,
            **kwargs,
        )
