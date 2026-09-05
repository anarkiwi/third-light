"""Phase-lead feedback, burst gating and the gate sequencer."""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from thirdlight.control import (
    Driver,
    GateSequencer,
    Interrupter,
    Melody,
    PhaseLead,
    Ramp,
    note_frequency,
)

DELAY = 1e-6
DEAD = 2e-7
GATED = Driver(
    lead=PhaseLead.from_angle(20.0, 3e5),
    delay=DELAY,
    dead_time=DEAD,
    interrupter=Interrupter(on_time=1e-3, frequency=100.0),
)
# (time, sequencer method, argument): a burst, a reversal, a superseded pending
# reversal, a burst end while disabled, then a burst end with a command pending.
SCRIPT = (
    (0.0, "burst", True),
    (2e-6, "crossing", 1),
    (5e-6, "crossing", -1),
    (1e-5, "crossing", 1),
    (1.5e-5, "crossing", -1),
    (1.55e-5, "crossing", 1),
    (2e-5, "burst", False),
    (2.2e-5, "crossing", -1),
    (3e-5, "burst", True),
    (3.45e-5, "crossing", -1),
    (3.5e-5, "burst", False),
)
HISTORY = (
    (0.0, 1),
    (5e-6 + DELAY, 0),
    (5e-6 + DELAY + DEAD, -1),
    (1e-5 + DELAY, 0),
    (1e-5 + DELAY + DEAD, 1),
    (2e-5, 0),
    (3e-5, 1),
    (3.5e-5, 0),
)


def replay(driver, script, start=0.0):
    """Run a scripted event list as an integrator would, returning gate changes."""
    sequencer = GateSequencer(driver, start)
    history = []
    previous = sequencer.gate

    def record(time):
        nonlocal previous
        if sequencer.gate != previous:
            previous = sequencer.gate
            history.append((time, sequencer.gate))

    for time, method, argument in script:
        while sequencer.next_time() <= time:
            due = sequencer.next_time()
            sequencer.fire(due)
            record(due)
        getattr(sequencer, method)(time, argument)
        record(time)
    while math.isfinite(sequencer.next_time()):
        due = sequencer.next_time()
        sequencer.fire(due)
        record(due)
    return history


def alternates(gate, edges):
    """Whether ``gate`` changes state across every edge and nowhere between them."""
    return np.all(gate.active(edges - 1e-9) != gate.active(edges + 1e-9))


@pytest.mark.parametrize("degrees", [0.0, 5.0, 20.0, 45.0, -10.0])
def test_phase_lead_round_trip(degrees):
    """from_angle and angle invert each other at the design frequency."""
    frequency = 3.4e5
    lead = PhaseLead.from_angle(degrees, frequency)
    assert lead.angle(frequency) == pytest.approx(degrees, abs=1e-12)
    assert PhaseLead(0.0).angle(frequency) == 0.0


def test_functional_matches_state_equation():
    """c x + d equals x[index] + tau (A x + B u)[index] for random data."""
    rng = np.random.default_rng(20240912)
    a = rng.normal(size=(6, 6))
    b = rng.normal(size=(6, 3))
    u = rng.normal(size=3)
    x = rng.normal(size=6)
    lead = PhaseLead(tau=1.7e-7)
    for index in range(6):
        c, d = lead.functional(a, b, u, index)
        expected = x[index] + lead.tau * (a @ x + b @ u)[index]
        assert c @ x + d == pytest.approx(expected, rel=1e-12)


def test_functional_leads_the_current_zero_crossing():
    """On an ideal LC the functional crosses arctan(omega tau)/omega early."""
    inductance, capacitance = 60e-6, 0.15e-6
    omega = 1.0 / math.sqrt(inductance * capacitance)
    a = np.array([[0.0, -1.0 / inductance], [1.0 / capacitance, 0.0]])
    b = np.zeros((2, 1))
    lead = PhaseLead.from_angle(15.0, omega / (2.0 * math.pi))
    c, d = lead.functional(a, b, np.zeros(1))
    assert d == 0.0

    def signal(t):
        state = np.array(
            [math.sin(omega * t), -inductance * omega * math.cos(omega * t)]
        )
        return c @ state + d

    period = 2.0 * math.pi / omega
    root = brentq(signal, -0.25 * period, 0.0, xtol=1e-18, rtol=8.9e-16)
    assert -root == pytest.approx(math.atan(omega * lead.tau) / omega, rel=1e-12)
    assert signal(0.0) == pytest.approx(omega * lead.tau, rel=1e-12)


def test_note_frequency_and_prf():
    """Equal temperament from A440, and an interrupter tuned to a note."""
    assert note_frequency(69) == 440.0
    assert note_frequency(81) == 880.0
    assert note_frequency(57) == 220.0
    np.testing.assert_allclose(
        note_frequency([60, 72]), [261.6255653006, 523.2511306012]
    )
    interrupter = Interrupter.from_note(69, on_time=4e-4)
    assert interrupter.frequency == 440.0
    assert interrupter.duty == pytest.approx(0.176)


def test_interrupter_edges_match_activity():
    """Every edge flips activity, over a run of thousands of bursts."""
    interrupter = Interrupter(on_time=1.0 / 2048, frequency=512.0)
    duration = 8.0
    edges = interrupter.edges(duration)
    assert len(edges) == 2 * int(duration * interrupter.frequency)
    assert interrupter.duty == 0.25
    np.testing.assert_allclose(np.diff(edges[1::2]), interrupter.period)
    assert not interrupter.active(edges[0::2]).any()
    assert interrupter.active(edges[1::2]).all()
    assert alternates(interrupter, edges)
    with pytest.raises(ValueError):
        Interrupter(on_time=1e-3, frequency=1000.0)
    with pytest.raises(ValueError):
        Interrupter(on_time=1e-3, frequency=0.0)


def test_melody_schedule():
    """Two notes and the silent gap between them."""
    melody = Melody(notes=((0.0, 0.02, 69), (0.05, 0.02, 81)), on_time=1e-3)
    edges = melody.edges(0.08)
    assert alternates(melody, edges)
    assert melody.active(0.0)
    assert not melody.active(0.03)
    assert not melody.active(0.075)
    assert melody.active(0.05)
    first, second = edges[edges < 0.02], edges[edges >= 0.05]
    np.testing.assert_allclose(np.diff(first[1::2]), 1.0 / note_frequency(69))
    np.testing.assert_allclose(np.diff(second[0::2]), 1.0 / note_frequency(81))
    assert len(first) == 2 * math.ceil(0.02 * note_frequency(69)) - 1
    assert len(second) == 2 * math.ceil(0.02 * note_frequency(81))
    assert np.all(np.diff(edges) > 0.0)
    empty = Melody(notes=(), on_time=1e-3)
    assert not empty.active(0.0)
    assert len(empty.edges(1.0)) == 0
    with pytest.raises(ValueError):
        Melody(notes=((0.0, 1.0, 69),), on_time=1.0)


def test_ramp_voltage():
    """Linear to the end of the rise, held after; zero rise is a flat bus."""
    ramp = Ramp(final=400.0, initial=100.0, rise=2e-3)
    elapsed = np.array([0.0, 5e-4, 1e-3, 2e-3, 5e-3])
    np.testing.assert_allclose(
        ramp.voltage(elapsed), [100.0, 175.0, 250.0, 400.0, 400.0]
    )
    np.testing.assert_allclose(Ramp(final=350.0).voltage(elapsed), 350.0)
    assert float(Ramp(final=350.0).voltage(0.0)) == 350.0


def test_gate_sequence():
    """Seed pulse, delay, dead time, supersession and the forced end of burst."""
    sequencer = GateSequencer(GATED)
    assert sequencer.gate == 0 and not sequencer.enabled
    assert sequencer.next_time() == math.inf
    sequencer.crossing(0.0, -1)
    assert sequencer.next_time() == math.inf
    np.testing.assert_allclose(replay(GATED, SCRIPT), HISTORY, rtol=1e-12)


def test_gate_sequence_without_dead_time():
    """Zero dead time reverses in one transition, one delay after the crossing."""
    driver = Driver(lead=PhaseLead(0.0), delay=DELAY, interrupter=GATED.interrupter)
    script = ((0.0, "burst", True), (5e-6, "crossing", -1), (1e-5, "crossing", 1))
    expected = ((0.0, 1), (5e-6 + DELAY, -1), (1e-5 + DELAY, 1))
    np.testing.assert_allclose(replay(driver, script), expected, rtol=1e-12)


def test_ungated_driver_runs_free():
    """With no interrupter the driver is always enabled and seeded at construction."""
    driver = Driver(lead=PhaseLead(0.0), bus=340.0)
    sequencer = driver.sequencer()
    assert sequencer.enabled and sequencer.gate == 1
    assert sequencer.bus_voltage(1.0) == 340.0
    assert len(driver.edges(1.0)) == 0
    sequencer.crossing(1e-6, -1)
    assert sequencer.fire(sequencer.next_time()) == -1
    np.testing.assert_allclose(GATED.edges(0.025), GATED.interrupter.edges(0.025))


def test_bus_voltage_follows_each_burst():
    """The ramp is measured from the current burst start and resets on the next."""
    driver = Driver(
        lead=PhaseLead(0.0),
        interrupter=GATED.interrupter,
        ramp=Ramp(final=400.0, rise=2e-3),
    )
    sequencer = driver.sequencer()
    sequencer.burst(0.0, True)
    assert sequencer.bus_voltage(1e-3) == pytest.approx(200.0)
    assert sequencer.bus_voltage(3e-3) == pytest.approx(400.0)
    sequencer.burst(1e-2, False)
    sequencer.burst(2e-2, True)
    assert sequencer.burst_start == 2e-2
    assert sequencer.bus_voltage(2e-2 + 1e-3) == pytest.approx(200.0)
