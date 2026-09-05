"""Bridge conduction states, tank tuning and the primary/modal state space."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from thirdlight.circuit import (
    DIODE,
    IGBT,
    OPEN,
    Bridge,
    Switch,
    Tank,
    from_design,
    from_modes,
    tune,
)
from thirdlight.em import inductance, losses
from thirdlight.geometry import Design, Primary, Solenoid, Toroid
from thirdlight.secondary import Modes, coupling, resonance

IDEAL = Bridge(igbt=Switch(0.0, 0.0), diode=Switch(0.0, 0.0))
REAL = Bridge(igbt=Switch(1.6, 0.02), diode=Switch(1.1, 0.015))
V_BUS = 400.0

# Normalised units: L_p = C_p = 1 puts w_0 at 1 rad/s and every state at O(1), so
# one absolute tolerance serves the whole vector.
UNIT_L = 1.0
UNIT_F = 1.0 / (2.0 * math.pi)

SMALL = Design(
    secondary=Solenoid(
        radius=0.04, length=0.2, turns=60, wire_diameter=3e-4, base=0.02
    ),
    primary=Primary(inner_radius=0.06, turns=4.0, pitch=0.01),
    top_load=Toroid(major_radius=0.06, minor_radius=0.02, height=0.24),
    sections=40,
    top_load_sections=8,
)


def modal(frequencies, l_m):
    """Modes carrying only the top-referred equivalents the state space reads."""
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    l_m = np.atleast_1d(np.asarray(l_m, dtype=float))
    empty = np.zeros((len(f), 0))
    return Modes(
        f=f,
        v=empty,
        i=empty,
        l_m=l_m,
        c_m=1.0 / (l_m * (2.0 * math.pi * f) ** 2),
        z=np.zeros(0),
    )


def tuned(k, l_m=4.0, quality=np.inf, bridge=IDEAL, resistance=0.0):
    """Single mode of inductance ``l_m``, primary tuned to it with coupling ``k``."""
    modes = modal(UNIT_F, l_m)
    tank = Tank(tune(UNIT_L, UNIT_F), resistance=resistance)
    return from_modes(modes, [k], [quality], UNIT_L, tank, bridge)


def free_response(net, span, points=4001):
    """Lossless ring-down from all energy in the primary capacitor."""
    start = np.zeros(net.size)
    start[net.modes + 1] = 1.0
    sol = solve_ivp(
        lambda t, x: net.a[IGBT] @ x,
        (0.0, span),
        start,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        dense_output=True,
    )
    return sol, np.linspace(0.0, span, points)


def primary_energy(net, x):
    """Energy held by the primary alone, (1/2) C_p v_Cp^2 + (1/2) L_p i_p^2."""
    return 0.5 * (
        net.capacitances[0] * net.voltages(x)[..., 0] ** 2
        + net.inductances[0, 0] * net.primary_current(x) ** 2
    )


def oscillatory(matrix):
    """Eigenvalues of positive imaginary part, and their eigenvectors, by frequency."""
    values, vectors = np.linalg.eig(matrix)
    keep = values.imag > 0.0
    values, vectors = values[keep], vectors[:, keep]
    order = np.argsort(values.imag)
    return values[order], vectors[:, order]


def eigen_quality(values):
    """Q = |Im lambda| / (2 |Re lambda|)."""
    return values.imag / (-2.0 * values.real)


def energy_quality(net, values, vectors, index=IGBT):
    """Q = w W / P from each eigenvector's stored energy and its dissipation."""
    loops = net.modes + 1
    current, voltage = vectors[:loops], vectors[loops:]
    stored = 0.5 * (
        np.real(np.einsum("ij,ik,kj->j", current.conj(), net.inductances, current))
        + (net.capacitances[:, None] * np.abs(voltage) ** 2).sum(axis=0)
    )
    loss = np.real(
        np.einsum("ij,i,ij->j", current.conj(), net.resistances[index], current)
    )
    return values.imag * stored / loss


def test_tune_inverts_the_resonance_formula():
    for inductances in (1e-6, 100e-6, 3.4e-3):
        for frequency in (5e4, 3e5, 5e6):
            for ratio in (0.9, 1.0, 1.05):
                capacitance = tune(inductances, frequency, ratio)
                assert 1.0 / (
                    2.0 * math.pi * math.sqrt(inductances * capacitance)
                ) == pytest.approx(ratio * frequency, rel=1e-14)


@pytest.mark.parametrize("k", [0.05, 0.2, 0.384615, 0.6])
@pytest.mark.parametrize("l_m", [4.0, 1e-3])
def test_lossless_split_frequencies(k, l_m):
    """A tuned pair splits to f_0 / sqrt(1 -+ k), whatever the turns ratio."""
    net = tuned(k, l_m=l_m)
    values, _ = oscillatory(net.a[IGBT])
    assert np.abs(values.real).max() < 1e-12 * np.abs(values.imag).max()
    assert values.imag / (2.0 * math.pi) == pytest.approx(
        np.sort([UNIT_F / math.sqrt(1.0 + k), UNIT_F / math.sqrt(1.0 - k)]), rel=1e-10
    )


@pytest.mark.parametrize("pair", [(3, 2), (5, 4)])
def test_complete_energy_transfer_at_de_queiroz_coupling(pair):
    """k = (n^2 - m^2)/(n^2 + m^2) empties the primary at w_0 t = pi m sqrt(1 + k).

    The mode frequencies are then w_0/sqrt(1 +- k) = m W, n W, so the primary
    voltage cos(m W t) + cos(n W t) = 2 cos((n+m)Wt/2) cos((n-m)Wt/2) and its
    derivative, the primary current, vanish together only where that product has
    a double root. Both factors can vanish at one t only when n - m is odd, which
    is de Queiroz's condition; see :func:`test_equal_parity_ratios_fall_short`.
    """
    n, m = pair
    k = (n * n - m * m) / (n * n + m * m)
    net = tuned(k)
    instant = math.pi * m * math.sqrt(1.0 + k)
    sol, times = free_response(net, 1.6 * instant)
    initial = net.energy(sol.y[:, 0])

    def fraction(t):
        return primary_energy(net, sol.sol(t)) / initial

    coarse = primary_energy(net, sol.sol(times).T) / initial
    j = int(np.argmin(coarse))
    found = minimize_scalar(
        fraction,
        bounds=(times[max(j - 1, 0)], times[min(j + 1, times.size - 1)]),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert found.fun < 1e-6
    assert found.x == pytest.approx(instant, rel=1e-6)


@pytest.mark.parametrize("pair", [(5, 3), (3, 1)])
def test_equal_parity_ratios_fall_short(pair):
    """Equal-parity n:m leaves the primary holding energy: no double root exists."""
    n, m = pair
    net = tuned((n * n - m * m) / (n * n + m * m))
    sol, times = free_response(net, 4.0 * math.pi * m)
    residual = primary_energy(net, sol.sol(times).T) / net.energy(sol.y[:, 0])
    assert residual.min() > 1e-3


def test_lossless_free_response_conserves_energy():
    net = tuned(0.384615)
    sol, times = free_response(net, 40.0)
    stored = net.energy(sol.sol(times).T)
    assert stored / stored[0] == pytest.approx(1.0, rel=1e-9)


def test_energy_splits_between_the_stores_and_reaches_the_top_node():
    net = tuned(0.6)
    state = np.array([2.0, -1.0, 3.0, 5.0])
    assert net.size == 4
    assert net.primary_current(state) == 2.0
    assert np.array_equal(net.currents(state), [2.0, -1.0])
    assert np.array_equal(net.voltages(state), [3.0, 5.0])
    assert net.top_voltage(state) == 5.0
    assert net.energy(state) == pytest.approx(
        0.5 * net.currents(state) @ net.inductances @ net.currents(state)
        + 0.5 * (net.capacitances * net.voltages(state) ** 2).sum()
    )
    assert net.energy(np.stack([state, np.zeros(4)])) == pytest.approx(
        [net.energy(state), 0.0]
    )


def test_top_load_current_forces_every_mode_and_spares_the_tank():
    """Column 1 of B drives dv_m/dt by -i_load/c_m and leaves v_Cp untouched."""
    net = from_modes(
        modal([1e5, 3e5], [1e-4, 1e-5]),
        [0.15, -0.1],
        [300.0, 400.0],
        1e-4,
        Tank(1e-7),
        IDEAL,
    )
    assert net.modes == 2
    assert net.size == 6
    assert np.array_equal(net.b[:3, 1], np.zeros(3))
    assert net.b[3, 1] == 0.0
    assert net.b[4:, 1] == pytest.approx(-1.0 / net.capacitances[1:])
    assert net.b[:3, 0] == pytest.approx(np.linalg.inv(net.inductances)[:, 0])
    assert np.array_equal(net.b[3:, 0], np.zeros(3))
    assert net.top_voltage(np.arange(6.0)) == 4.0 + 5.0


def test_inductance_matrix_is_arrowhead_and_the_tank_overrides_geometry():
    net = from_modes(
        modal([1e5, 3e5], [1e-4, 1e-5]),
        [0.15, -0.1],
        [300.0, 400.0],
        1e-4,
        Tank(1e-7, resistance=0.5, inductance=2e-4),
        REAL,
    )
    assert net.inductances[0, 0] == 2e-4
    assert net.inductances[1, 2] == 0.0
    assert net.inductances[0, 1:] == pytest.approx(
        np.array([0.15, -0.1]) * np.sqrt(2e-4 * np.array([1e-4, 1e-5]))
    )
    assert net.inductances == pytest.approx(net.inductances.T)
    modal_r = 2.0 * math.pi * net.frequencies * np.array([1e-4, 1e-5]) / [300.0, 400.0]
    assert net.resistances[IGBT] == pytest.approx(np.concatenate(([0.54], modal_r)))
    assert net.resistances[DIODE] == pytest.approx(np.concatenate(([0.53], modal_r)))


@pytest.mark.parametrize("gate", [-1, 0, 1])
@pytest.mark.parametrize("current", [-7.0, 7.0])
def test_bridge_conduction_states_and_drive(gate, current):
    """IGBTs conduct with the command, diodes against it, and gate 0 freewheels."""
    net = tuned(0.2, bridge=REAL)
    index, sigma, sign = net.state(gate, current)
    s = math.copysign(1.0, current)
    assert sign == s
    if gate == 0:
        assert (index, sigma) == (DIODE, -s)
    else:
        assert sigma == float(gate)
        assert index == (IGBT if s == gate else DIODE)
    ideal = sigma * V_BUS
    drive = net.drive(gate, current, V_BUS)
    assert drive == pytest.approx(ideal - 2.0 * REAL.conducting(index).v0 * s)
    assert (drive - ideal) * s < 0.0
    assert (gate != 0) or drive * s < 0.0


def test_a_bridge_at_zero_current_is_open_and_carries_no_drive():
    """Nothing conducts at i_p = 0: the primary row of A vanishes and the drive is 0."""
    net = tuned(0.2, bridge=REAL)
    for gate in (-1, 0, 1):
        assert net.state(gate, 0.0) == (OPEN, 0.0, 0.0)
        assert net.drive(gate, 0.0, V_BUS) == 0.0
    assert np.all(net.a[OPEN][0] == 0.0)
    assert REAL.conducting(OPEN) is None


def test_the_open_bridge_freezes_the_tank_and_uncouples_the_modes():
    """i_p and v_Cp are held while each mode rings down on its own l_m, c_m and r_m."""
    net = tuned(0.2, bridge=REAL)
    loops = net.modes + 1
    assert np.all(net.a[OPEN][loops] == 0.0)
    block = net.a[OPEN][1:loops, 1:loops]
    assert block == pytest.approx(np.diag(np.diagonal(block)))
    assert net.resistances[OPEN] == pytest.approx(
        np.concatenate(([0.0], net.resistances[IGBT][1:]))
    )
    x = np.zeros(net.size)
    x[loops:] = 1.0
    assert (net.a[OPEN] @ x)[0] == 0.0


def test_half_bridge_halves_the_swing_and_drops_one_device():
    half = Bridge(igbt=REAL.igbt, diode=REAL.diode, full=False)
    assert (half.gain, half.devices) == (0.5, 1)
    assert (REAL.gain, REAL.devices) == (1.0, 2)
    full_net = tuned(0.2, bridge=REAL, resistance=0.1)
    half_net = tuned(0.2, bridge=half, resistance=0.1)
    for gate, current in ((1, 3.0), (-1, 3.0), (0, -3.0)):
        _, sigma, sign = half_net.state(gate, current)
        device = half.conducting(half_net.state(gate, current)[0])
        assert half_net.drive(gate, current, V_BUS) == pytest.approx(
            0.5 * sigma * V_BUS - device.v0 * sign
        )
        assert full_net.drive(gate, current, V_BUS) == pytest.approx(
            sigma * V_BUS - 2.0 * device.v0 * sign
        )
    assert half_net.resistances[IGBT][0] == pytest.approx(0.1 + REAL.igbt.r)
    assert full_net.resistances[IGBT][0] == pytest.approx(0.1 + 2.0 * REAL.igbt.r)


def test_decoupled_branches_ring_down_at_their_own_quality_factor():
    """Uncoupled, every branch is a plain RLC: |Im| / (2|Re|) = Q sqrt(1 - 1/(4 Q^2))."""
    quality = np.array([150.0, 250.0, 600.0])
    l_p, c_p = 1e-4, 1e-7
    resistance = math.sqrt(l_p / c_p) / quality[0]
    net = from_modes(
        modal([1e5, 3e5], [1e-4, 1e-5]),
        [0.0, 0.0],
        quality[1:],
        l_p,
        Tank(c_p, resistance=resistance),
        IDEAL,
    )
    values, vectors = oscillatory(net.a[IGBT])
    damped = quality * np.sqrt(1.0 - 0.25 / quality**2)
    assert eigen_quality(values) == pytest.approx(damped, rel=1e-12)
    assert energy_quality(net, values, vectors) == pytest.approx(
        eigen_quality(values), rel=1e-4
    )


def test_from_design_state_space_tracks_the_coupled_split_and_the_modal_q():
    """Mode 1 splits about f_1; each pair's damping is its own energy balance.

    The weakly coupled mode-2 pair keeps the unloaded modal Q to under a percent;
    the mode-1 pair runs higher because half its energy sits in the lossless
    primary.
    """
    eigen = resonance(SMALL, modes=2)
    k = coupling(SMALL, eigen)
    quality = losses.quality_factor(SMALL, eigen)
    l_p = float(inductance.inductance_matrix(SMALL.primary_rings()).sum())
    net = from_design(SMALL, Tank(tune(l_p, eigen.f[0])), REAL, modes=2)
    assert net.a.shape == (3, 6, 6)
    assert net.b.shape == (6, 2)
    assert np.isfinite(net.a).all() and np.isfinite(net.b).all()
    assert net.inductances[0, 0] == pytest.approx(l_p, rel=1e-12)
    assert net.frequencies == pytest.approx(eigen.f, rel=1e-12)

    values, vectors = oscillatory(net.a[IGBT])
    assert values.size == 3
    assert np.all(values.real < 0.0)
    split = eigen.f[0] / np.sqrt(1.0 + np.array([k[0], -k[0]]))
    assert values.imag[:2] / (2.0 * math.pi) == pytest.approx(split, rel=0.01)
    assert values.imag[2] / (2.0 * math.pi) == pytest.approx(eigen.f[1], rel=0.01)
    damping = eigen_quality(values)
    assert damping == pytest.approx(energy_quality(net, values, vectors), rel=1e-6)
    assert damping[2] == pytest.approx(quality[1], rel=0.03)
    assert damping[0] == pytest.approx(2.0 * quality[0], rel=0.05)
