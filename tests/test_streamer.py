"""Streamer branch in the state space, and the length dynamics that size it."""

import math

import numpy as np
import pytest
import yaml
from scipy.integrate import solve_ivp

from thirdlight.circuit import Bridge, Switch, Tank, from_modes, tune, with_streamer
from thirdlight.circuit.devices import IGBT
from thirdlight.circuit.devices import index as state_index
from thirdlight.discharge import Breakout, Streamer
from thirdlight.machine import Machine
from thirdlight.secondary import Modes

IDEAL = Bridge(igbt=Switch(0.0, 0.0), diode=Switch(0.0, 0.0))
FORWARD = state_index(IGBT, 1.0)
UNIT_L = 1.0
UNIT_F = 1.0 / (2.0 * math.pi)
BARE = Breakout(field=np.ones((1, 1)), critical=np.array([1.0]))

with open("examples/drsstc.yaml", encoding="utf-8") as _handle:
    SPEC = yaml.safe_load(_handle)


def unit_network(l_m=4.0, modes=1):
    """Normalised primary tuned to ``modes`` identical top-referred modes."""
    f = np.full(modes, UNIT_F)
    inductance = np.full(modes, l_m)
    equivalents = Modes(
        f=f,
        v=np.zeros((modes, 0)),
        i=np.zeros((modes, 0)),
        l_m=inductance,
        c_m=1.0 / (inductance * (2.0 * math.pi * f) ** 2),
        z=np.zeros(0),
    )
    tank = Tank(tune(UNIT_L, UNIT_F))
    return from_modes(equivalents, [0.2] * modes, [np.inf] * modes, UNIT_L, tank, IDEAL)


def channel(**changes):
    """Streamer with round constants, on an electrode of unit field per volt."""
    spec = {
        "breakout": BARE,
        "frequency": 1.0e5,
        "growth": 1.0e-2,
        "cooling": 1.0e-3,
    }
    spec.update(changes)
    return Streamer(**spec)


def small(**changes):
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
            "interrupter": {"on_time": 4e-6, "frequency": 20000.0},
        },
    }
    spec.update(changes)
    return Machine.from_dict(spec)


def test_the_length_update_solves_the_growth_ode_exactly():
    """Both regimes of (|v| - E l)_+ are linear, so each step is one exponential."""
    streamer = channel()

    def reference(length, voltage, span):
        return solve_ivp(
            lambda _, state: streamer.growth
            * max(abs(voltage) - streamer.gradient * state[0], 0.0)
            - state[0] / streamer.cooling,
            (0.0, span),
            [length],
            method="DOP853",
            rtol=1e-12,
            atol=1e-16,
        ).y[0, -1]

    for length, voltage, span in [
        (1e-3, 3.0e5, 1e-5),
        (0.5, 3.0e5, 2e-4),
        (1.0, 0.0, 5e-4),
        (2.0, 1.0e5, 1e-4),
    ]:
        assert streamer.advance(length, voltage, 1.0, span) == pytest.approx(
            reference(length, voltage, span), rel=1e-9
        )


def test_a_channel_starts_only_once_the_electrode_reaches_the_peek_threshold():
    """Initiation needs the surface field; a channel that exists carries its own tip."""
    streamer = channel()
    assert streamer.advance(0.0, 5.0e5, 0.999, 1e-6) == 0.0
    assert streamer.advance(0.0, 5.0e5, 1.0, 1e-6) > 0.0
    assert streamer.advance(1e-3, 5.0e5, 0.0, 1e-6) > 1e-3


def test_the_channel_settles_at_the_length_its_top_voltage_sustains():
    """Equilibrium is a fixed point of the update and sits below V / E."""
    streamer = channel()
    settled = streamer.equilibrium(4.0e5)
    assert settled < 4.0e5 / streamer.gradient
    assert streamer.advance(settled, 4.0e5, 1.0, 1e-4) == pytest.approx(settled)
    assert streamer.advance(0.0, 0.0, 1.0, 1e-4) == 0.0
    assert streamer.advance(1.0, 0.0, 1.0, streamer.cooling) == pytest.approx(
        math.exp(-1.0)
    )


def test_the_capacitance_is_quantised_geometrically_from_an_immaterial_floor():
    """Below the floor the branch neither loads nor detunes, so its level is held."""
    streamer = channel()
    admittance = (
        2.0 * math.pi * streamer.frequency * streamer.resistance * streamer.capacitance
    )
    assert admittance * streamer.minimum == pytest.approx(streamer.floor)
    assert streamer.level(0.0) == streamer.level(0.5 * streamer.minimum) == 0
    for length in (0.001, 0.05, 0.4, 3.0):
        level = streamer.level(length)
        exact = streamer.capacitance * max(length, streamer.minimum)
        assert streamer.capacitance_at(level) == pytest.approx(
            exact, rel=0.5 * streamer.tolerance + 1e-12
        )


def test_the_branch_obeys_the_equations_it_stands_for():
    """dv_m/dt = (i_m - i_s) / c_m for every mode, and dv_s/dt = i_s / C_s."""
    base = unit_network(modes=3)
    resistance, capacitance = 2.2e5, 1.5e-12
    net = with_streamer(base, resistance, capacitance)
    rng = np.random.default_rng(7)
    x = rng.normal(size=net.size)
    u = np.array([0.0, 0.3, 0.0])
    rate = net.a[FORWARD] @ x + net.b[FORWARD] @ u
    current = net.streamer_current(x)
    modes = slice(net.loops + 1, 2 * net.loops)
    assert current == pytest.approx(
        (net.top_voltage(x) - net.streamer_voltage(x)) / resistance
    )
    assert rate[modes] == pytest.approx(
        (net.currents(x)[1:] - current - u[1]) / net.capacitances[1:]
    )
    assert rate[-1] == pytest.approx(current / capacitance)
    assert net.a[FORWARD][: net.loops, :-1] == pytest.approx(
        base.a[FORWARD][: net.loops]
    )


def test_the_branch_only_dissipates():
    """A passive R-C branch can only take energy out of the resonator."""
    net = with_streamer(unit_network(), 4.0, 0.5)
    rng = np.random.default_rng(11)
    for x in rng.normal(size=(20, net.size)):
        assert (
            net.streamer_current(x) * (net.top_voltage(x) - net.streamer_voltage(x))
            >= 0.0
        )


def test_the_branch_integrates_like_solve_ivp():
    """One more row of a piecewise linear state space, exact between events."""
    net = with_streamer(unit_network(), 4.0, 0.05)
    start = np.zeros(net.size)
    start[net.modes + 1] = 1.0
    u = np.array([0.0, 0.0, 0.0])
    span = 3.0
    reference = solve_ivp(
        lambda _, x: net.a[FORWARD] @ x + net.b[FORWARD] @ u,
        (0.0, span),
        start,
        method="DOP853",
        rtol=1e-13,
        atol=1e-15,
    ).y[:, -1]
    from thirdlight.solver import Propagator  # pylint: disable=import-outside-toplevel

    prop = Propagator.build(net.a[FORWARD], net.b[FORWARD], span)
    assert prop.advance(start, u) == pytest.approx(reference, rel=1e-9, abs=1e-12)


def test_a_seeded_channel_loads_the_resonator():
    """The branch damps the ring-up and takes real power out of the top node."""
    machine = small()
    streamer = machine.streamer(growth=2.0, cooling=2e-5)
    hot = machine.run(6e-6, streamer=streamer, length0=0.05)
    cold = machine.run(6e-6)
    assert np.abs(hot.top_voltage).max() < np.abs(cold.top_voltage).max()
    assert hot.streamer_power.max() > 0.0
    assert np.all(hot.streamer_power >= 0.0)
    assert cold.streamer_power == pytest.approx(np.zeros(len(cold)))
    assert cold.network.streamer_current(cold.x) == pytest.approx(np.zeros(len(cold)))
    assert cold.network.streamer_voltage(cold.x) == pytest.approx(np.zeros(len(cold)))
    assert hot.network.streamer[0] == streamer.resistance


@pytest.mark.parametrize("divisor,tol", [(1, 1e-4), (4, 1e-5)])
def test_the_energy_ledger_closes_around_the_streamer(divisor, tol):
    """Bus energy in equals dissipation plus storage, the channel included."""
    machine = small()
    streamer = machine.streamer(growth=2.0, cooling=2e-5)
    result = machine.run(
        6e-6, step=machine.step / divisor, streamer=streamer, length0=0.05
    )
    stored = result.energy[-1] - result.energy[0]
    residual = result.input_energy - result.dissipation - stored
    assert abs(residual) < tol * abs(result.input_energy)


def test_a_capacitance_change_never_creates_energy():
    """Growth carries the channel's charge; cooling carries it away. Both lose."""
    machine = small()
    streamer = machine.streamer(growth=2.0, cooling=2e-5)
    for seed in (0.0, 0.05):
        result = machine.run(6e-6, streamer=streamer, length0=seed)
        assert np.all(np.diff(result.loss) >= 0.0)
        assert result.channel == pytest.approx(
            [streamer.capacitance_at(streamer.level(l)) for l in result.length]
        )


@pytest.mark.parametrize("tolerance", [0.05, 0.005])
def test_the_quantised_capacitance_does_not_move_the_answer(tolerance):
    """A level is a set fraction of capacitance, well inside the model's own spread."""
    machine = small()
    streamer = machine.streamer(growth=2.0, cooling=2e-5)
    reference = machine.run(6e-6, streamer=streamer, length0=0.05)
    coarse = machine.run(
        6e-6,
        streamer=Streamer(**{**streamer.__dict__, "tolerance": tolerance}),
        length0=0.05,
    )
    assert coarse.length[-1] == pytest.approx(reference.length[-1], rel=5e-3)
    assert coarse.input_energy == pytest.approx(reference.input_energy, rel=5e-3)


def test_a_run_breaks_out_on_its_own_and_grows_a_channel():
    """The shrunken coil has no voltage gain, so its bus stands in for the turns."""
    machine = small(
        driver={
            **SPEC["driver"],
            "bus": 3.0e4,
            "interrupter": {"on_time": 4e-6, "frequency": 20000.0},
        }
    )
    streamer = machine.streamer(growth=2.0, cooling=2e-5)
    result = machine.run(6e-6, streamer=streamer)
    live = np.flatnonzero(result.length > 0.0)
    assert live.size > 0
    assert result.length[-1] > 0.01
    reached = machine.breakout().margin(machine.network.voltages(result.x)[:, 1:])
    # A length is recorded against the state that produced it, one interval on.
    assert reached[live[0]] >= 1.0
    assert reached[: live[0]].max() < 1.0


def test_a_channel_that_reverses_the_current_inside_a_step_still_advances():
    """Regression: the run once looped on a commutation instant it could not leave."""
    machine = small(
        driver={
            **SPEC["driver"],
            "bus": 1.0e4,
            "interrupter": {"on_time": 4e-6, "frequency": 20000.0},
        }
    )
    streamer = machine.streamer(growth=3.05957, cooling=3.79915e-6)
    result = machine.run(6e-6, streamer=streamer)
    assert np.all(np.diff(result.t) >= 0.0)
    assert result.t[-1] == pytest.approx(6e-6)
    assert len(result) < 20000
