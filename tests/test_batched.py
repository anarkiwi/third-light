"""Batched stepping of a design space, against the reference stepper design by design."""

import copy
import math
from dataclasses import replace

import numpy as np
import pytest
import yaml

from thirdlight.circuit import with_streamer
from thirdlight.control import Driver, Interrupter, PhaseLead
from thirdlight.machine import Machine
from thirdlight.solver import batched
from thirdlight.solver.batched import (  # pylint: disable=protected-access
    _edge_time,
    _fire,
    _queue,
)
from thirdlight.solver.propagator import Propagator

with open("examples/drsstc.yaml", encoding="utf-8") as _handle:
    SPEC = yaml.safe_load(_handle)
SPEC["secondary"]["turns"] = 60
SPEC["sections"] = 20
SPEC["top_load_sections"] = 8
STEP = Machine.from_dict(SPEC).step
RESERVOIR = {"capacitance": 1.0e-6, "resistance": 0.2}
TINY = 160
LONG = 1200


def machine(driver=None, **changes):
    """The example machine shrunk to a size the interpreted pass can run.

    The gate delay, the dead time and the gating are given in steps: the shrunk
    coil resonates two decades above the one the example file is written for.
    """
    spec = copy.deepcopy(SPEC)
    spec.update(changes)
    spec["driver"] = {
        **spec["driver"],
        "delay": 4 * STEP,
        "dead_time": 3 * STEP,
        "interrupter": None,
        **(driver or {}),
    }
    return Machine.from_dict(spec)


def burst(on, period):
    """Interrupter gating for ``on`` steps out of every ``period``."""
    return {"on_time": on * STEP, "frequency": 1.0 / (period * STEP)}


def ramp(rise):
    """QCW bus envelope rising over ``rise`` steps from the burst start."""
    return {"initial": 150.0, "final": 400.0, "rise": rise * STEP}


def spread(**changes):
    """Three designs over one half cycle, together reaching every driver path.

    A flat bus with the gate delay a few steps out; a gated one on a rising ramp;
    and one whose sub-step delay and absent dead time put the commanded
    transition inside the interval that queued it, on a ramp already held.
    """
    return [
        machine(**changes),
        machine(driver={"interrupter": burst(40, 70), "ramp": ramp(30)}, **changes),
        machine(
            driver={"delay": 0.1 * STEP, "dead_time": 0.0, "ramp": ramp(0)}, **changes
        ),
    ]


def fleet():
    """Designs apart in lead, bus, tune, dead time, gating and bus envelope."""
    return [
        machine(),
        machine(driver={"lead_angle": 20.0}),
        machine(driver={"bus": 200.0}),
        machine(tank={"tune": 1.04, "resistance": 0.05, "dissipation_factor": 0.001}),
        machine(driver={"dead_time": 0.0}),
        machine(driver={"interrupter": burst(60, 100)}),
        machine(driver={"interrupter": burst(60, 100), "ramp": ramp(40)}),
    ]


def compare(machines, steps, rel=1e-9):
    """Assert each design's batch observables against its own reference run.

    ``bisect`` roots where ``brentq`` roots, both inside a span times 1e-14, so
    the two trajectories agree to rather better than the tolerance asked for.
    """
    duration = steps * STEP
    packed = batched.pack(machines)
    out = batched.run(packed, duration)
    assert packed.designs == len(machines)
    for d, item in enumerate(machines):
        result = item.run(duration)
        scale = float(np.abs(result.x[-1]).max())
        assert out.steps[d] == len(result) - 1
        assert out.peak_current[d] == pytest.approx(
            np.abs(result.primary_current).max(), rel=rel
        )
        assert out.peak_voltage[d] == pytest.approx(
            np.abs(result.top_voltage).max(), rel=rel
        )
        assert out.input_energy[d] == pytest.approx(result.input_energy, rel=rel)
        assert out.dissipation[d] == pytest.approx(result.dissipation, rel=rel)
        assert out.state[:, d] == pytest.approx(result.x[-1], rel=rel, abs=rel * scale)


@pytest.mark.parametrize("bus", [{}, {"bus": RESERVOIR}], ids=["stiff", "reservoir"])
def test_a_small_batch_matches_the_reference_stepper(bus):
    """Two designs over a few dozen intervals, small enough to run interpreted."""
    compare(spread(**bus), TINY)


@pytest.mark.slow
def test_the_batch_matches_the_reference_stepper():
    compare(fleet(), LONG)


@pytest.mark.slow
def test_a_reservoir_batch_matches_the_reference_stepper():
    """A sagging bus puts the swing on a state rather than on the supply input."""
    compare(spread(bus=RESERVOIR), LONG)


def test_the_packed_tables_carry_every_switch_state():
    machines = spread()
    packed = batched.pack(machines)
    net = machines[0].network
    states, size, inputs = len(net.a), net.size, net.b.shape[-1]
    assert packed.lam_re.shape == (states, size, 3)
    assert packed.basis_re.shape == packed.inverse_re.shape == (states, size, size, 3)
    assert packed.inject_re.shape == (states, size, inputs, 3)
    assert packed.a0[:, :, 0] == pytest.approx(net.a[:, 0, :])
    assert packed.b0[:, :, 0] == pytest.approx(net.b[:, 0, :])
    assert packed.resist[:, :, 0] == pytest.approx(net.resistances)
    assert packed.rows[2, 2, 0] == net.index(1, 1.0)
    assert packed.offsets[2, 0, 0] == pytest.approx(net.offset(1, -1.0))
    assert packed.drive[batched.STEP, 0] == pytest.approx(machines[0].step)
    assert not packed.drive[batched.GATED, 0]
    assert packed.drive[batched.RAMPED, 1]
    assert len(batched.POLARITY) == states


def test_float32_packing_agrees_with_float64():
    """Halved precision costs about five digits of the observables.

    The interval count is not held to: an eigenbasis of this conditioning
    resolves the pinned-zero commutation test to about float32's own epsilon, so
    a run can take a few extra intervals through the diode dead zone.
    """
    machines = spread()
    duration = TINY * STEP
    wide = batched.run(batched.pack(machines), duration)
    narrow = batched.run(batched.pack(machines, dtype=np.float32), duration)
    for name in ("peak_current", "peak_voltage", "input_energy", "dissipation"):
        assert getattr(narrow, name) == pytest.approx(getattr(wide, name), rel=1e-4)


def test_pack_rejects_an_empty_batch():
    with pytest.raises(ValueError, match="at least one machine"):
        batched.pack([])


def test_pack_rejects_a_streamer():
    """A channel capacitance re-levels mid-run, which no packed eigenbasis survives."""
    grown = machine()
    grown = replace(grown, network=with_streamer(grown.network, 1e6, 5e-12))
    with pytest.raises(ValueError, match="design 1 carries a streamer"):
        batched.pack([machine(), grown])


def test_pack_rejects_a_load_callback():
    with pytest.raises(ValueError, match="no top-node load callback"):
        batched.pack([machine()], load=lambda t, top: 0.0)


def test_pack_rejects_a_melody():
    """A note schedule is per-design array data, not a handful of scalars."""
    notes = {"notes": [[0.0, 1e-4, 60]], "on_time": 15 * STEP}
    with pytest.raises(ValueError, match="design 0 is gated by a Melody"):
        batched.pack([machine(driver={"interrupter": notes})])


def test_pack_rejects_designs_of_disagreeing_shape():
    with pytest.raises(ValueError, match="must agree on state size"):
        batched.pack([machine(), machine(bus=RESERVOIR)])


def test_pack_rejects_a_pade_fallback(monkeypatch):
    """Without an eigenbasis there is nothing to pack, whatever the propagator does."""
    exact = Propagator.build
    monkeypatch.setattr(
        Propagator, "build", lambda a, b, step: exact(a, b, step, cond_max=1.0)
    )
    with pytest.raises(ValueError, match=r"designs \[0, 1, 2\] fell back to Pade"):
        batched.pack(spread())


def test_the_analytic_edges_match_the_interrupter():
    """Edge ``e`` is at (e // 2) period plus on_time or period, and turns on when odd."""
    gating = Interrupter(on_time=1.5e-4, frequency=200.0)
    duration = 0.02
    times = []
    while True:
        when = _edge_time(len(times), gating.period, gating.on_time, duration)
        if math.isinf(when):
            break
        times.append(when)
    expected = gating.edges(duration)
    assert times == pytest.approx(expected)
    on = gating.active(np.nextafter(expected, math.inf))
    assert on.tolist() == [e % 2 == 1 for e in range(len(times))]


def test_the_gate_queue_reproduces_the_sequencer():
    """The two-slot device queue against GateSequencer over a chattering event list.

    The events supersede in every way :meth:`GateSequencer.crossing` allows: to
    the sign already queued, to the sign already commanded, and against a queue
    holding one entry and two.
    """
    driver = Driver(lead=PhaseLead(0.0), delay=2.0, dead_time=1.0, bus=1.0)
    seq = driver.sequencer()
    gate, count, t0, g0, t1, g1 = 1, 0, math.inf, 0, math.inf, 0
    for t, sign in [
        (0.0, -1.0),
        (0.5, 1.0),
        (1.0, -1.0),
        (1.5, -1.0),
        (3.5, -1.0),
        (4.5, 1.0),
        (8.0, 1.0),
        (9.0, -1.0),
    ]:
        seq.fire(t)
        seq.crossing(t, sign)
        gate, count, t0, g0, t1, g1 = _fire(t, gate, count, t0, g0, t1, g1)
        count, t0, g0, t1, g1 = _queue(
            t, sign, gate, count, t0, g0, t1, g1, driver.delay, driver.dead_time
        )
        assert gate == seq.gate
        assert (t0 if count > 0 else math.inf) == seq.next_time()
