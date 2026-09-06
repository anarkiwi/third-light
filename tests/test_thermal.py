"""Switching-energy fits, commutation attribution and the component loss ledger."""

import copy
import math

import numpy as np
import pytest
import yaml
from test_stepping import driver, modal, network, period

from thirdlight import thermal
from thirdlight.circuit import Bridge, Switch, Tank, from_modes, tune
from thirdlight.circuit.devices import IGBT, Energy, polarity
from thirdlight.circuit.devices import index as state_index
from thirdlight.control import Driver, PhaseLead, Ramp
from thirdlight.em.losses import capacitor_esr
from thirdlight.machine import Machine
from thirdlight.solver import simulate
from thirdlight.solver.stepping import Result

V_BUS = 340.0
V_TEST = 600.0
TJ_TEST = 125.0
PEAK = (50.0, 3.0)
FORWARD = state_index(IGBT, 1.0)
REVERSE = state_index(IGBT, -1.0)

with open("examples/drsstc.yaml", encoding="utf-8") as _handle:
    SPEC = yaml.safe_load(_handle)


def fit(coefficients, alpha=0.0):
    """Switching-energy fit at the datasheet test point."""
    return Energy(
        coefficients=coefficients, alpha=alpha, v_test=V_TEST, tj_test=TJ_TEST
    )


def switched(on=(0.0, 1.0e-5), off=(0.0, 2.0e-5), rr=(0.0, 5.0e-6), full=True):
    """Full or half bridge whose devices carry linear switching-energy fits."""
    return Bridge(
        igbt=Switch(1.2, 0.012, turn_on=fit(on), turn_off=fit(off)),
        diode=Switch(1.0, 0.010, recovery=fit(rr)),
        full=full,
    )


def small(**changes):
    """The example machine shrunk to a size the interpreted coverage pass can run."""
    spec = copy.deepcopy(SPEC)
    spec["secondary"]["turns"] = 60
    spec["sections"] = 20
    spec["top_load_sections"] = 8
    spec.update(changes)
    return Machine.from_dict(spec)


def sinusoid(net, halves=4, per_half=2000):
    """A synthetic result: i_p and i_m sinusoids commutated at every current zero.

    The bridge is gated exactly on the zero crossings, so every interval has an
    IGBT conducting with the current, which is what the closed forms integrate.
    """
    phase = np.linspace(0.0, halves * math.pi, halves * per_half + 1)
    x = np.zeros((len(phase), net.size))
    for loop, peak in enumerate(PEAK):
        x[:, loop] = peak * np.sin(phase)
    sign = np.sign(np.sin(0.5 * (phase[1:] + phase[:-1])))
    sign = np.append(sign, sign[-1])
    u = np.zeros((len(phase), 3))
    u[:, 0] = -net.bridge.devices * net.bridge.igbt.v0 * sign
    u[:, 2] = V_BUS
    zero = np.zeros(len(phase))
    return Result(
        t=phase / (2.0 * math.pi * net.frequencies[0]),
        x=x,
        gate=sign.astype(np.int8),
        state=np.where(sign > 0.0, FORWARD, REVERSE).astype(np.int8),
        u=u,
        network=net,
        length=zero,
        channel=zero,
        loss=zero,
    )


def tanked(resistance=0.4, dissipation_factor=0.0, loss_tangent=0.0, quality=400.0):
    """One-mode network with the tank tuned to it, at the given loss parameters."""
    l_p, f, l_m = 1e-4, 1e5, 6e-2
    tank = Tank(
        tune(l_p, f), resistance=resistance, dissipation_factor=dissipation_factor
    )
    return from_modes(
        modal([f], [l_m]),
        [0.2],
        [quality],
        l_p,
        tank,
        switched(),
        loss_tangent=loss_tangent,
    )


def overlap(current, voltage, rise, fall, points=2001):
    """Transition loss of i ramping to I at V, then v falling to 0 at I."""
    t = np.linspace(0.0, rise + fall, points)
    i = current * np.clip(t / rise, 0.0, 1.0)
    v = voltage * np.clip((rise + fall - t) / fall, 0.0, 1.0)
    return np.trapezoid(v * i, t)


def commutated(result):
    """Attributed events, with the bridge polarity either side of each."""
    hit = thermal.commutations(result)[0]
    return (
        result.losses().switching,
        polarity(result.state[hit - 1]),
        polarity(result.state[hit]),
    )


@pytest.mark.parametrize("current", [40.0, 300.0])
@pytest.mark.parametrize("voltage", [200.0, V_TEST])
def test_a_linear_fit_is_the_overlap_integral_of_the_transition(current, voltage):
    """E = V I (t_ri + t_fv)/2 is what a fit first order in I, linear in V, gives.

    That integral is the derivation the app-note form of [20] and [21] comes
    from, so it validates the machinery rather than a fitted number.
    """
    rise, fall = 60e-9, 90e-9
    device = Switch(1.2, 0.012, turn_on=fit((0.0, V_TEST * 0.5 * (rise + fall))))
    assert device.E_on(current, voltage=voltage) == pytest.approx(
        overlap(current, voltage, rise, fall), rel=1e-12
    )
    assert device.E_off(current, voltage=voltage) == 0.0
    assert device.E_rr(current, voltage=voltage) == 0.0


def test_the_renesas_worked_example_lands_inside_five_percent():
    """RBN75H125S1FP4 at 813 V, 13 A peak, 72 C and 10 kHz: [20] §5 reports 16.9 W.

    Its E_on 21 mJ and E_off 6 mJ at 75 A, 600 V and 150 C, with Kv 1.3 and
    0.003/K, are the fit; only the sqrt2/pi sinusoidal average of its inverter
    phase, which this model carries no equivalent of, is supplied here.
    """
    fits = {"alpha": 3e-3, "v_test": 600.0, "tj_test": 150.0, "exponent": 1.3}
    device = Switch(
        1.0,
        0.022,
        turn_on=Energy((0.0, 0.021 / 75.0), **fits),
        turn_off=Energy((0.0, 0.006 / 75.0), **fits),
    )
    current = 13.0 / math.sqrt(2.0)
    commutation = device.E_on(current, 72.0, 813.0) + device.E_off(current, 72.0, 813.0)
    assert 1e4 * math.sqrt(2.0) / math.pi * commutation == pytest.approx(16.9, rel=0.05)


def test_switching_energy_is_linear_in_the_blocking_voltage():
    """A fit holds at its own test voltage and scales from there, linearly at Kv 1."""
    device = switched().igbt
    reference = device.E_on(120.0, voltage=V_TEST)
    assert reference == pytest.approx(device.E_on(120.0), rel=1e-14)
    assert device.E_on(120.0, voltage=2.0 * V_TEST) == pytest.approx(
        2.0 * reference, rel=1e-14
    )
    assert device.E_on(120.0, voltage=0.0) == 0.0


@pytest.mark.parametrize("tj", [25.0, 125.0, 150.0])
def test_the_temperature_coefficient_reproduces_itself(tj):
    """E(Tj) = E(Tj_test) (1 + alpha dTj), and no argument extrapolates nothing."""
    device = Switch(1.2, 0.012, turn_on=fit((0.0, 1e-5, 2e-9), alpha=4e-3))
    plain = device.E_on(200.0)
    assert plain == pytest.approx(1e-5 * 200.0 + 2e-9 * 200.0**2, rel=1e-14)
    assert device.E_on(200.0, tj) == pytest.approx(
        plain * (1.0 + 4e-3 * (tj - TJ_TEST)), rel=1e-14
    )


def test_a_device_without_fits_costs_nothing_to_switch():
    """Every fit defaults to zero, so an older design loads and runs unchanged."""
    device = Switch(1.2, 0.012)
    for energy in (device.E_on, device.E_off, device.E_rr):
        assert energy(500.0, 150.0, 1200.0) == 0.0


@pytest.mark.parametrize("full", [True, False])
def test_a_bridge_commutates_every_device_in_series_with_the_tank(full):
    """Two devices commutate in a full bridge and one in a half, each at the whole bus."""
    bridge = switched(full=full)
    devices = 2 if full else 1
    assert bridge.commutation(150.0, TJ_TEST, V_TEST) == pytest.approx(
        [
            devices * bridge.igbt.E_on(150.0, TJ_TEST, V_TEST),
            devices * bridge.igbt.E_off(150.0, TJ_TEST, V_TEST),
            devices * bridge.diode.E_rr(150.0, TJ_TEST, V_TEST),
        ]
    )


def test_conduction_loss_matches_the_closed_form_for_a_sinusoid():
    """v0 + r i carrying I sin over whole half cycles costs (2/pi) v0 I + r I^2/2.

    The modal loop is the same integral at its own resistance, and the primary
    loop's is what is left of the tank resistance once the devices are taken out.
    """
    net = tanked()
    result = sinusoid(net)
    span = result.t[-1]
    bridge = net.bridge
    ledger = result.losses()
    assert ledger.igbt == pytest.approx(
        bridge.devices
        * (
            2.0 * bridge.igbt.v0 * PEAK[0] / math.pi
            + bridge.igbt.r * PEAK[0] ** 2 / 2.0
        )
        * span,
        rel=1e-6,
    )
    assert ledger.diode == 0.0
    assert ledger.primary == pytest.approx(0.4 * PEAK[0] ** 2 / 2.0 * span, rel=1e-6)
    assert ledger.winding[0] == pytest.approx(
        net.resistances[FORWARD][1] * PEAK[1] ** 2 / 2.0 * span, rel=1e-6
    )
    assert ledger.former == 0.0
    assert ledger.channel == 0.0
    assert ledger.total == pytest.approx(result.dissipation, rel=1e-12)


def test_the_dissipation_factor_splits_the_esr_out_without_moving_the_total():
    """DF says how much of the loop resistance the capacitor is, and adds none."""
    plain, lossy = (tanked(dissipation_factor=df) for df in (0.0, 2e-3))
    assert plain.resistances == pytest.approx(lossy.resistances)
    assert plain.esr == 0.0
    assert lossy.esr == pytest.approx(
        capacitor_esr(lossy.capacitances[0], lossy.frequencies[0], 2e-3)
    )
    split, whole = (sinusoid(net).losses() for net in (lossy, plain))
    assert split.esr > 0.0
    assert split.primary + split.esr == pytest.approx(whole.primary, rel=1e-12)
    assert split.total == pytest.approx(whole.total, rel=1e-12)


def test_a_capped_dissipation_factor_never_makes_the_rest_of_the_loop_negative():
    """A factor inconsistent with the loop resistance reports the whole of it."""
    net = tanked(resistance=1e-3, dissipation_factor=0.5)
    assert net.esr == pytest.approx(1e-3)
    ledger = sinusoid(net).losses()
    assert ledger.primary == pytest.approx(0.0, abs=1e-12)
    assert ledger.esr > 0.0


def test_the_former_loss_tangent_damps_each_mode_by_its_own_reactance():
    """tan d across c_m is tan d/(omega c_m) in series with it at that frequency."""
    plain, lossy = (tanked(loss_tangent=d) for d in (0.0, 5e-3))
    reactance = 1.0 / (2.0 * math.pi * lossy.frequencies[0] * lossy.capacitances[1])
    added = 5e-3 * reactance
    assert lossy.dielectric[0] == pytest.approx(added, rel=1e-12)
    assert lossy.resistances[FORWARD][1] - plain.resistances[FORWARD][1] == (
        pytest.approx(added, rel=1e-12)
    )
    result = sinusoid(lossy)
    ledger, span = result.losses(), result.t[-1]
    assert ledger.former == pytest.approx(added * PEAK[1] ** 2 / 2.0 * span, rel=1e-6)
    assert ledger.winding[0] == pytest.approx(
        plain.resistances[FORWARD][1] * PEAK[1] ** 2 / 2.0 * span, rel=1e-6
    )
    assert ledger.total == pytest.approx(result.dissipation, rel=1e-12)


def test_zero_current_switching_costs_nothing():
    """Gated on the current zeros, every commutation is at zero current."""
    net = network(bridge=switched())
    result = simulate(net, driver(), 40e-6, period(net) / 256.0)
    events, before, after = commutated(result)
    assert len(events) > 6
    assert np.all(events.current == 0.0)
    assert np.all(before != after)
    assert events.total == 0.0


def test_a_phase_lead_turns_off_into_current_and_turns_on_at_the_zero():
    """Leading the crossing takes the IGBTs out at current; the incoming pair is soft."""
    net = network(bridge=switched())
    span = period(net)
    omega = 2.0 * math.pi / span
    result = simulate(
        net, driver(lead=math.tan(0.2 * math.pi) / omega), 40e-6, span / 256.0
    )
    events, before, _ = commutated(result)
    assert len(events) > 6
    assert np.all(events.off > 0.0)
    assert np.all(events.on == 0.0)
    assert np.all(events.rr == 0.0)
    assert np.all(np.sign(events.current) == before)
    assert np.all(events.voltage == V_BUS)
    assert events.total == pytest.approx(events.off.sum())


def test_a_gate_delay_turns_on_into_current_and_recovers_the_opposite_diode():
    """Lagging the crossing hands the current to the incoming pair against the bus."""
    net = network(bridge=switched())
    span = period(net)
    result = simulate(net, driver(delay=0.1 * span), 40e-6, span / 256.0)
    events, _, after = commutated(result)
    assert len(events) > 4
    assert np.all(events.on > 0.0)
    assert np.all(events.rr > 0.0)
    assert np.all(events.off == 0.0)
    assert np.all(np.sign(events.current) == after)
    current = np.abs(events.current)
    assert events.on == pytest.approx(
        net.bridge.devices * net.bridge.igbt.E_on(current, voltage=events.voltage)
    )
    assert events.rr == pytest.approx(
        net.bridge.devices * net.bridge.diode.E_rr(current, voltage=events.voltage)
    )


def test_every_attributed_instant_is_a_gate_edge_or_a_pinned_current_zero():
    """A hard commutation lands on a gate edge; a soft one on a current zero."""
    net = network(bridge=switched())
    span = period(net)
    result = simulate(
        net, driver(delay=0.05 * span, dead_time=0.02 * span), 60e-6, span / 256.0
    )
    events = result.losses().switching
    edges = set(result.t[np.flatnonzero(np.diff(result.gate.astype(int))) + 1])
    zeros = set(result.t[result.primary_current == 0.0])
    assert len(events) > 8
    for instant, current in zip(events.t, events.current):
        assert instant in (edges if current != 0.0 else zeros)


def test_a_ramped_bus_scales_each_commutation_by_the_voltage_it_stands_at():
    """Every event carries the bus of its own instant, which a QCW ramp moves."""
    net = network(bridge=switched())
    span = period(net)
    ramp = Ramp(final=V_BUS, initial=0.1 * V_BUS, rise=40e-6)
    program = Driver(lead=PhaseLead(0.0), delay=0.1 * span, ramp=ramp)
    result = simulate(net, program, 40e-6, span / 256.0)
    events = result.losses().switching
    assert events.voltage[-1] > 3.0 * events.voltage[0]
    assert events.voltage == pytest.approx(ramp.voltage(events.t))
    assert events.on == pytest.approx(
        net.bridge.devices
        * net.bridge.igbt.turn_on.coefficients[1]
        * np.abs(events.current)
        * events.voltage
        / V_TEST
    )


def test_the_ledger_closes_on_the_dissipation_of_a_run_with_a_streamer():
    """The split ties to the already-validated energy ledger of the run."""
    machine = small()
    streamer = machine.streamer(growth=2.0, cooling=2e-5)
    result = machine.run(6e-6, streamer=streamer, length0=0.05)
    ledger = result.losses(tj=110.0)
    assert ledger.channel > 0.0
    assert ledger.conduction == ledger.igbt + ledger.diode
    assert ledger.total > 0.0
    assert ledger.total == pytest.approx(result.dissipation, rel=1e-12)


def test_the_example_carries_switching_fits_and_a_tank_dissipation_factor():
    """The schema keys reach the built machine, and a run reports them."""
    machine = small()
    assert machine.tank.dissipation_factor == SPEC["tank"]["dissipation_factor"]
    assert machine.bridge.igbt.turn_on.v_test == 600.0
    assert machine.bridge.diode.recovery.coefficients == (0.0, 5.0e-5, -3.0e-8)
    assert machine.network.esr > 0.0
    assert np.all(np.asarray(machine.bridge.commutation(200.0, 150.0, V_BUS)) > 0.0)
