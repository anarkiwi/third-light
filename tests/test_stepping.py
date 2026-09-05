"""Event-driven stepping of the bridge, tank and modes, and its validation."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from thirdlight.circuit import IGBT, OPEN, Bridge, Switch, Tank, from_modes, tune
from thirdlight.control import Driver, Interrupter, PhaseLead, Ramp
from thirdlight.secondary import Modes
from thirdlight.solver import simulate

V_BUS = 340.0
IDEAL = Bridge(igbt=Switch(0.0, 0.0), diode=Switch(0.0, 0.0))
REAL = Bridge(igbt=Switch(1.2, 0.012), diode=Switch(1.0, 0.010))


def modal(frequencies, inductances):
    """Modes carrying only the top-referred equivalents the network needs."""
    f = np.asarray(frequencies, dtype=float)
    l_m = np.asarray(inductances, dtype=float)
    zero = np.zeros((len(f), 1))
    return Modes(
        f=f, v=zero, i=zero, l_m=l_m, c_m=1.0 / (l_m * (2.0 * math.pi * f) ** 2), z=zero
    )


def network(quality=(400.0,), k=(0.2,), l_p=1e-4, f=(1e5,), l_m=(6e-2,), bridge=REAL):
    """One- or two-mode network with the tank tuned to the first mode."""
    return from_modes(modal(f, l_m), k, quality, l_p, Tank(tune(l_p, f[0])), bridge)


def driver(lead=0.0, delay=0.0, dead_time=0.0, bus=V_BUS, **kwargs):
    """Self-oscillating driver with the lead given as tau, not an angle."""
    return Driver(
        lead=PhaseLead(lead), delay=delay, dead_time=dead_time, bus=bus, **kwargs
    )


def period(net):
    """Period of the lower coupled split, the pole a ZCS driver locks to."""
    values = np.linalg.eigvals(net.a[0])
    return 2.0 * math.pi / np.abs(values.imag).min()


def transitions(result, value=None):
    """Times at which the gate command becomes ``value``, or any polarity."""
    changed = np.concatenate(
        ([0], np.flatnonzero(np.diff(result.gate.astype(int))) + 1)
    )
    gate = result.gate[changed]
    keep = gate != 0 if value is None else gate == value
    return result.t[changed][keep]


def commutations(result):
    """Times at which the primary current is pinned to zero by a commutation."""
    return result.t[result.primary_current == 0.0]


def offsets(result):
    """Distance from each gate edge to the nearest commutation.

    The seed pulse that opens a burst is dropped: no crossing asked for it.
    """
    edges, zeros = transitions(result)[1:], commutations(result)
    return np.abs(edges[:, None] - zeros[None, :]).min(axis=1)


def test_every_interval_matches_an_independent_stiff_solver():
    """Each constant-mode interval reproduces solve_ivp on the same A, B and held u.

    The reference is DOP853 at rtol 1e-13, so this is the design's numerical
    parity check on the propagator and the state-space assembly together.
    """
    net = network()
    result = simulate(net, driver(), 40e-6, period(net) / 256.0)
    error = 0.0
    for i in range(len(result) - 1):
        span = result.t[i + 1] - result.t[i]
        if span == 0.0:
            continue
        a, b = net.a[result.device[i]], net.b
        u = np.array([result.drive[i], 0.0])
        exact = solve_ivp(
            lambda _, x, a=a, b=b, u=u: a @ x + b @ u,
            (0.0, span),
            result.x[i],
            method="DOP853",
            rtol=1e-13,
            atol=1e-16,
        ).y[:, -1]
        scale = max(np.abs(exact).max(), 1e-12)
        error = max(error, np.abs(exact - result.x[i + 1]).max() / scale)
    assert error < 1e-9


def test_a_lossless_network_conserves_energy_over_a_ring_down():
    """No resistance and no bus: the bridge only redirects a fixed stored energy."""
    net = network(quality=(np.inf,), bridge=IDEAL)
    x0 = np.zeros(net.size)
    x0[net.modes + 1] = 1000.0
    result = simulate(net, driver(bus=0.0), 200e-6, period(net) / 256.0, x0=x0)
    energy = result.energy
    assert len(result) > 4000
    assert np.abs(energy / energy[0] - 1.0).max() < 1e-10


def test_zero_crossing_switching_without_delay_or_lead():
    """Gate transitions land exactly on the primary current zero crossings."""
    net = network()
    result = simulate(net, driver(), 40e-6, period(net) / 256.0)
    assert commutations(result).size > 8
    assert offsets(result).max() == 0.0


def test_phase_lead_cancels_the_gate_delay():
    """tau = tan(omega t_d)/omega puts the delayed gate edge back on the zero crossing.

    The UD2 design rule: without the lead the edge lags by the propagation delay,
    with it the residual is the small-signal error of the first-order lead alone.
    """
    net = network()
    step = period(net) / 256.0
    delay = 0.08 * period(net)
    omega = 2.0 * math.pi / period(net)
    lagged, led = (
        simulate(net, driver(lead=tau, delay=delay), 40e-6, step)
        for tau in (0.0, math.tan(omega * delay) / omega)
    )
    assert offsets(lagged).mean() == pytest.approx(delay, rel=0.05)
    assert offsets(led).mean() < 0.05 * delay


def test_dead_time_holds_the_bridge_off_between_polarities():
    """Every reversal passes through gate 0 for exactly the dead time."""
    net = network()
    dead = 0.02 * period(net)
    result = simulate(
        net,
        driver(delay=0.01 * period(net), dead_time=dead),
        40e-6,
        period(net) / 256.0,
    )
    changed = np.flatnonzero(np.diff(result.gate.astype(int)) != 0) + 1
    gates, times = result.gate[changed], result.t[changed]
    off = np.flatnonzero(gates == 0)
    assert off.size > 4
    assert times[off + 1] - times[off] == pytest.approx(dead, rel=1e-9)


def test_the_bridge_blocks_once_the_current_reaches_zero_off_a_burst():
    """After a burst the freewheeling current stops at zero and the tank freezes."""
    net = network()
    gating = Interrupter(on_time=20e-6, frequency=1e4)
    result = simulate(net, driver(interrupter=gating), 99e-6, period(net) / 256.0)
    off = result.t > 20e-6
    assert np.all(result.gate[off] == 0)
    blocked = off & (result.device == OPEN)
    assert blocked.sum() > 0.5 * off.sum()
    assert np.abs(result.primary_current[blocked]).max() == 0.0
    run = np.flatnonzero(blocked)
    run = run[: np.flatnonzero(np.diff(run) != 1)[0] + 1]
    assert run.size > 8
    assert np.ptp(result.x[run][:, net.modes + 1]) == 0.0
    assert np.abs(result.top_voltage[blocked]).max() > 0.0


def test_the_interrupter_restarts_a_burst_from_the_seed_pulse():
    """Two bursts each begin with a positive gate command and ring up again."""
    net = network()
    gating = Interrupter(on_time=15e-6, frequency=2.5e4)
    result = simulate(net, driver(interrupter=gating), 60e-6, period(net) / 256.0)
    starts = transitions(result, 1)
    assert starts.min() == 0.0
    assert np.abs(starts - 40e-6).min() < period(net) / 128.0
    peak = [
        np.abs(result.primary_current[(result.t > a) & (result.t < b)]).max()
        for a, b in ((0.0, 15e-6), (40e-6, 55e-6))
    ]
    assert min(peak) > 0.5 * max(peak)


def test_a_ramped_bus_ramps_the_primary_current():
    """A QCW bus ramp gives an envelope that grows with the ramp, not the step."""
    net = network()
    ramp = Ramp(final=V_BUS, initial=0.05 * V_BUS, rise=40e-6)
    flat = simulate(net, driver(), 40e-6, period(net) / 256.0)
    ramped = simulate(
        net,
        Driver(lead=PhaseLead(0.0), ramp=ramp, interrupter=Interrupter(40e-6, 1e4)),
        40e-6,
        period(net) / 256.0,
    )
    conducting = np.abs(ramped.drive[ramped.device == IGBT])
    assert conducting.max() <= V_BUS
    assert conducting[0] < 0.2 * conducting[-1]
    envelope = np.maximum.accumulate(np.abs(ramped.primary_current))
    assert envelope[-1] < np.abs(flat.primary_current).max()
    assert np.interp(20e-6, ramped.t, envelope) < 0.5 * envelope[-1]


def test_a_load_current_at_the_top_node_damps_the_secondary():
    """A resistive top-node load bleeds modal energy without touching the tank."""
    net = network()
    step = period(net) / 256.0
    free, loaded = (
        simulate(net, driver(), 40e-6, step, load=load)
        for load in (None, lambda _, v: v / 2.2e5)
    )
    assert np.abs(loaded.top_voltage).max() < 0.9 * np.abs(free.top_voltage).max()


def test_two_modes_run_and_keep_the_state_layout():
    """A two-mode network exposes both modal voltages at the same top node."""
    net = network(quality=(400.0, 500.0), k=(0.2, -0.1), f=(1e5, 3e5), l_m=(6e-2, 5e-3))
    result = simulate(net, driver(), 20e-6, period(net) / 256.0)
    assert result.x.shape[1] == net.size == 6
    assert result.top_voltage == pytest.approx(result.x[:, 4] + result.x[:, 5])
    assert np.abs(result.x[:, 5]).max() > 0.0


def test_an_initial_state_is_not_written_through():
    """The caller's array is copied, and the run starts from it."""
    net = network()
    x0 = np.zeros(net.size)
    x0[net.modes + 1] = 10.0
    result = simulate(net, driver(bus=0.0), 1e-6, period(net) / 256.0, x0=x0)
    assert result.x[0] == pytest.approx(x0)
    assert x0[net.modes + 1] == 10.0
