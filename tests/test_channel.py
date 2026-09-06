"""The grown tree as a channel model, and the seam it shares with the scalar one."""

import math
from dataclasses import replace

import numpy as np
import pytest
import yaml

from thirdlight.discharge import Growth, Tree
from thirdlight.discharge.channel import State, TreeChannel
from thirdlight.discharge.filament import (
    channel_load,
    path_resistance,
    segment_resistance,
    series_resistance,
)
from thirdlight.discharge.streamer import GROWTH_GAIN
from thirdlight.machine import Machine
from thirdlight.solver.stepping import _Channel

with open("examples/drsstc.yaml", encoding="utf-8") as _handle:
    SPEC = yaml.safe_load(_handle)

STEP = 0.02
RADIUS = 1.0e-3
SERIES = ("t", "x", "gate", "state", "u", "length", "channel", "loss", "resistance")
OBSERVABLES = (
    "primary_current",
    "top_voltage",
    "energy",
    "streamer_current",
    "streamer_power",
    "input_energy",
    "dissipation",
)


def small(bus=3.0e4, **changes):
    """The example machine shrunk, with a needle breakout point on the top load."""
    spec = {
        **SPEC,
        "secondary": {**SPEC["secondary"], "turns": 60},
        "sections": 20,
        "top_load_sections": 8,
        "breakout": {"radius": 2e-4, "height": 0.665},
        "breakout_sections": 6,
        "driver": {
            **SPEC["driver"],
            "bus": bus,
            "interrupter": {"on_time": 4e-6, "frequency": 20000.0},
        },
    }
    spec.update(changes)
    return Machine.from_dict(spec)


def growth(step=STEP, **changes):
    """Growth rule of the coupled tests, one step of the DBM per segment."""
    return Growth(step=step, radius=RADIUS, eta=1.0, directions=16, **changes)


def channel(machine, seed=0, **changes):
    """Tree channel of a machine, seeded so that a model-level test repeats."""
    changes.setdefault("rng", np.random.default_rng(seed))
    return machine.channel(growth(**changes.pop("growth", {})), **changes)


def chain(count, spread=0.0, seed=0):
    """A tree of ``count`` segments over the plane, a needle at ``spread`` 0."""
    rng = np.random.default_rng(seed)
    nodes = np.zeros((count + 1, 3))
    nodes[0, 2] = 1.0
    parent = np.full(count + 1, -1)
    for k in range(1, count + 1):
        parent[k] = 0 if k == 1 else rng.integers(max(k - 3, 1), k) if spread else k - 1
        nodes[k] = nodes[parent[k]] + np.array([spread * rng.normal(), 0.0, STEP])
    return Tree(nodes, parent, RADIUS)


def grown(model, count, voltage=3.0e5, current=0.0):
    """Grow ``count`` segments off a fresh state at a held voltage, by the growth clock."""
    state = model.initial(0.0)
    span = count * model.growth.step / (model.velocity * voltage)
    return model.advance(state, voltage, 1.0, span, current)


class Legacy:  # pylint: disable=unused-argument
    """The seam before it was generalised: a length for state, a constant resistance."""

    def __init__(self, streamer):
        self.streamer = streamer
        self.breakout = streamer.breakout

    def initial(self, seed=0.0, rng=None):
        return float(seed)

    def extent(self, state):
        return state

    def level(self, state):
        return self.streamer.level(state)

    def capacitance_at(self, level):
        return self.streamer.capacitance_at(level)

    def resistance_at(self, level):
        return self.streamer.resistance

    def advance(self, state, voltage, margin, dt, current=0.0):
        return self.streamer.advance(state, voltage, margin, dt)


def test_the_scalar_model_runs_bit_identically_through_the_generalised_seam():
    """The generalised seam asks the scalar model only what the old one did."""
    machine = small()
    streamer = machine.streamer(growth=2.0, cooling=2e-5)
    new = machine.run(6e-6, streamer=streamer, length0=0.05)
    old = machine.run(6e-6, streamer=Legacy(streamer), length0=0.05)
    for name in SERIES:
        assert np.array_equal(getattr(new, name), getattr(old, name)), name
    for name in OBSERVABLES:
        assert np.array_equal(getattr(new, name), getattr(old, name)), name
    assert np.array_equal(new.x[-1], old.x[-1])
    assert new.losses().total == old.losses().total


def test_the_scalar_seam_reports_the_branch_the_scalar_model_asked_for():
    """Every sample's capacitance and resistance are the model's own of that length."""
    machine = small()
    streamer = machine.streamer(growth=2.0, cooling=2e-5)
    result = machine.run(6e-6, streamer=streamer, length0=0.05)
    assert result.channel == pytest.approx(
        [streamer.capacitance_at(streamer.level(l)) for l in result.length], rel=0.0
    )
    assert np.all(result.resistance == streamer.resistance)
    voltages = machine.network.voltages(result.x)[:, 1:]
    margin = streamer.breakout.margin(voltages)
    length = result.length[0]
    for k in range(1, len(result)):
        length = streamer.advance(
            length, result.top_voltage[k - 1], margin[k], result.t[k] - result.t[k - 1]
        )
        assert result.length[k] == pytest.approx(length, rel=1e-9, abs=1e-15)


def test_the_path_resistance_is_the_ancestor_walk_it_replaces():
    """One forward prefix pass against walking every node's own ancestry."""
    tree = chain(60, spread=0.4, seed=3)
    element = segment_resistance(tree, 7.0)
    walked = np.zeros(len(tree))
    for k in range(1, len(tree)):
        ancestry, node = [], k
        while node > 0:
            ancestry.append(node)
            node = tree.parent[node]
        for step in reversed(ancestry):
            walked[k] += element[step - 1]
    assert path_resistance(tree, 7.0) == pytest.approx(walked, rel=0.0, abs=0.0)


def test_the_tip_potential_falls_by_the_path_resistance_at_the_channel_current():
    """The drop each node's own path carries, and nothing else, moves it off the drive."""
    machine = small()
    model = channel(machine)
    tree = chain(12, spread=0.3, seed=1)
    voltage, current = 4.0e5, 1.0e-3
    node = model.potential(tree, voltage, current)
    assert node == pytest.approx(
        voltage - current * path_resistance(tree, model.resistivity), rel=0.0
    )
    assert node[0] == voltage
    assert np.all(np.diff(node[1:]) <= 0.0) or np.all(node[1:] < voltage)
    assert model.potential(tree, voltage, 0.0) == pytest.approx(
        np.full(len(tree), voltage), rel=0.0
    )


def test_the_default_resistivity_is_fritz_s_own_ohms_per_metre():
    """220 kOhm re-expressed as a gradient along a channel of the growth radius."""
    machine = small()
    model = channel(machine)
    metre = Tree(
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]), np.array([-1, 0]), RADIUS
    )
    assert series_resistance(
        metre, np.array([1.0]), model.resistivity
    ) == pytest.approx(model.resistance, rel=1e-12)


def test_cooling_prunes_the_youngest_suffix_and_leaves_a_valid_tree():
    """The decay sets the extent; the tree follows it down to within half a step."""
    machine = small()
    model = channel(machine, cooling=1.0e-5)
    state = grown(model, 6)
    before = state.tree
    assert before.segments >= 3
    reach = state.reach
    for _ in range(4):
        span = 3.0e-6
        reach *= math.exp(-span / model.cooling)
        model.advance(state, 0.0, 0.0, span)
        tree = state.tree
        Tree(tree.nodes, tree.parent, tree.radius)
        assert len(tree) <= len(before)
        assert np.array_equal(tree.nodes, before.nodes[: len(tree)])
        assert np.array_equal(tree.parent, before.parent[: len(tree)])
        assert model.extent(state) <= reach + 0.5 * model.growth.step
    model.advance(state, 0.0, 0.0, 20.0 * model.cooling)
    assert len(state.tree) == 1
    assert model.extent(state) == 0.0
    assert state.tree.segments == 0


def test_growth_is_clocked_at_the_leader_velocity_the_voltage_sets():
    """A segment takes h / v, and the remainder carries rather than quantising down."""
    machine = small()
    model = channel(machine, cooling=1.0e3)
    voltage = 3.0e5
    third = model.growth.step / (3.0 * model.velocity * voltage)
    state = model.initial(0.0)
    for _ in range(30):
        model.advance(state, voltage, 1.0, third)
    assert 9 <= state.tree.segments <= 10
    assert (
        model.advance(model.initial(0.0), voltage, 0.0, 30.0 * third).tree.segments == 0
    )


def test_a_lengthening_channel_raises_both_its_capacitance_and_its_resistance():
    """Level for level, the branch the tree calls for is larger and more resistive.

    Along a chain: branching puts the segments in parallel and lowers the series
    reduction, which is what a parallel path does.
    """
    model = channel(small())
    loads = [
        channel_load(model.rings, chain(count), model.resistivity)
        for count in (1, 2, 4, 8)
    ]
    assert [c for c, _ in loads] == sorted(c for c, _ in loads)
    assert [r for _, r in loads] == sorted(r for _, r in loads)
    assert all(c > 0.0 and r > 0.0 for c, r in loads)


def test_levels_rise_with_the_tree_and_the_stage_cache_is_revisited():
    """The level climbs as the tree grows and returns to its own stages as it decays."""
    machine = small()
    model = channel(machine, cooling=1.0e-5)
    state = model.initial(0.0)
    seam = _Channel(machine.network, machine.step, model, state)
    levels = []
    for _ in range(5):
        model.advance(state, 3.0e5, 1.0, model.growth.step / (model.velocity * 3.0e5))
        seam.retune(np.zeros(seam.network.size))
        levels.append(model.level(state))
        assert seam.capacitance == model.capacitance_at(levels[-1])
        assert seam.resistance == model.resistance_at(levels[-1])
    assert levels == sorted(levels)
    assert levels[-1] > levels[0]
    assert model.capacitance_at(levels[-1]) > model.capacitance_at(levels[0])
    built, seen = len(seam.stages), {0, *levels}
    for _ in range(6):
        model.advance(state, 0.0, 0.0, 3.0e-6)
        seam.retune(np.zeros(seam.network.size))
        assert seam.level in seen
    assert model.level(state) < levels[-1]
    assert len(seam.stages) == built


def test_a_run_at_a_seed_repeats_and_another_seed_differs():
    """The generator is the whole of the model's randomness."""
    machine = small()
    runs = [
        machine.run(6e-6, streamer=channel(machine), rng=np.random.default_rng(seed))
        for seed in (0, 0, 1)
    ]
    for name in SERIES:
        assert np.array_equal(getattr(runs[0], name), getattr(runs[1], name)), name
    assert not np.array_equal(runs[0].length, runs[2].length)


def test_a_coupled_bang_grows_a_tree_that_stays_valid_throughout():
    """The channel grows while driven, its extent climbs, and every tree is a tree."""
    machine = small(bus=1.0e5)
    model = channel(machine)
    seen = []
    state = model.initial(0.0)

    def watched(_state, voltage, margin, dt, current=0.0):
        out = TreeChannel.advance(model, _state, voltage, margin, dt, current)
        tree = out.tree
        Tree(tree.nodes, tree.parent, tree.radius)
        seen.append((len(tree), model.extent(out)))
        return out

    model.advance = watched
    result = machine.run(6e-6, streamer=model, length0=state)
    counts = np.array([n for n, _ in seen])
    extent = np.array([e for _, e in seen])
    assert counts[-1] > 1
    assert np.all(np.diff(counts) >= 0)
    assert np.all(np.diff(extent) >= 0.0)
    assert result.length[-1] == pytest.approx(extent[-1])
    assert np.all(np.abs(state.tree.lengths - model.growth.step) < 1e-12)


@pytest.mark.parametrize("divisor,tol", [(1, 4e-2), (4, 1.2e-2)])
def test_the_energy_ledger_closes_around_the_grown_channel(divisor, tol):
    """Bus energy in equals dissipation plus storage, the channel included.

    The branch relaxes faster than a step where the channel is short, so the
    trapezoid resolves the level changes only to first order; see 3.4d.
    """
    machine = small()
    result = machine.run(
        6e-6,
        step=machine.step / divisor,
        streamer=channel(machine),
        rng=np.random.default_rng(0),
    )
    stored = result.energy[-1] - result.energy[0]
    residual = result.input_energy - result.dissipation - stored
    assert abs(residual) < tol * abs(result.input_energy)
    assert np.all(np.diff(result.loss) >= 0.0)
    assert result.losses().total == pytest.approx(result.dissipation, rel=1e-12)


def test_growth_stops_at_the_critical_field_and_at_the_buffer_it_was_sized_for():
    """A stall is field limited or capacity limited, and neither corrupts the state."""
    machine = small()
    voltage = 3.0e5
    span = 8.0 * STEP / (GROWTH_GAIN * voltage)
    dead = channel(machine, gradient=1.0e12)
    state = dead.advance(dead.initial(0.0), voltage, 1.0, span)
    assert state.tree.segments == 0
    assert state.budget == 0.0
    capped = channel(machine, steps=2)
    state = capped.advance(capped.initial(0.0), voltage, 1.0, span)
    assert state.tree.segments == 2
    state.discharge.prune(len(state.tree) + 4)
    assert state.tree.segments == 2


def test_a_channel_roots_on_the_top_load_where_there_is_no_breakout_point():
    """The outer equator carries the field once the needle is taken away."""
    spec = {key: value for key, value in SPEC.items() if key != "breakout"}
    machine = Machine.from_dict({**spec, "sections": 20, "top_load_sections": 8})
    model = machine.channel(growth())
    load = machine.design.top_load
    assert model.seed == (load.major_radius + load.minor_radius, 0.0, load.height)
    assert model.direction == (1.0, 0.0, 0.0)
    bare = replace(machine.design, top_load=None, breakout=None)
    with pytest.raises(ValueError):
        TreeChannel.from_design(bare, machine.breakout(), machine.frequency, growth())


def test_a_channel_model_carries_its_own_state_across_bursts():
    """A state seeds the next run where a length seeds a fresh tree."""
    machine = small()
    model = channel(machine)
    state = grown(model, 3)
    assert model.initial(state) is state
    fresh = model.initial(0.05)
    assert isinstance(fresh, State)
    assert fresh.reach == 0.05
    assert fresh.tree.segments == 0
