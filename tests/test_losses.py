"""AC resistance, modal Q and the lumped loss terms, and their validation."""

import math

import numpy as np
import pytest
from scipy.constants import mu_0
from scipy.special import ber, bei, beip, berp, ive, jv

from thirdlight.em import inductance, losses
from thirdlight.em.losses import (
    ac_ratio,
    capacitor_esr,
    dc_resistance,
    dielectric_conductance,
    field_reach,
    kelvin_argument,
    mode_quality,
    proximity_ratio,
    quality_factor,
    resistivity,
    resonant_resistance,
    section_resistance,
    skin_depth,
    skin_ratio,
)
from thirdlight.geometry import Design, Primary, Solenoid, Toroid
from thirdlight.secondary import ladder, resonance

SECONDARY = Solenoid(
    radius=0.076, length=0.5, turns=1000, wire_diameter=4e-4, base=0.05
)
DRSSTC = Design(
    secondary=SECONDARY,
    primary=Primary(inner_radius=0.115, turns=5.5, pitch=0.012, base=0.02),
    top_load=Toroid(major_radius=0.15, minor_radius=0.05, height=0.62),
    sections=100,
    top_load_sections=16,
)
Q_GRID = np.geomspace(0.01, 100.0, 1201)
# Medhurst 1947 Table VIII: phi = R_hf(coil) / R_hf(the same length of straight wire at
# the same frequency), by d/s (wire diameter over pitch) and coil length over diameter.
# Scans at g6yb.com/g3ynh/zdocs/refs/Medhurst/ , part 2 p88; his high-frequency limit.
MEDHURST_INFINITE = {0.1: 1.05, 0.2: 1.19, 0.3: 1.40, 0.5: 1.93, 0.8: 2.81, 1.0: 3.41}
MEDHURST_TEN = {0.1: 1.04, 0.2: 1.17, 0.3: 1.35, 0.5: 1.83, 0.8: 2.65, 1.0: 3.23}
MEDHURST_LENGTH_08 = (2.74, 2.83, 2.97, 3.10, 3.20, 3.17, 2.74, 2.60, 2.60, 2.62, 2.65)
# Butterworth, Experimental Wireless and The Wireless Engineer, Apr 1926 p207 Table I:
# z -> (1 + F, G) in R_ac = R(1 + F + u G (d/c)^2), z = pi d sqrt(2 f/rho) = this q.
BUTTERWORTH = {
    0.5: (1.000, 0.00097),
    1.0: (1.005, 0.01519),
    2.0: (1.078, 0.1724),
    3.0: (1.318, 0.4049),
    5.0: (2.043, 0.755),
    10.0: (3.799, 1.641),
    20.0: (7.328, 3.409),
    50.0: (17.93, 8.713),
    100.0: (35.61, 17.55),
}
# scipy's Cephes Kelvin routines change branch near q = 10 and shed seven digits.
CEPHES_DIP = (5.0, 15.0)


def kelvin_skin(q):
    """R_ac/R_dc from scipy's Kelvin functions, the textbook ber/bei form."""
    return (
        0.5 * q * (ber(q) * beip(q) - bei(q) * berp(q)) / (berp(q) ** 2 + beip(q) ** 2)
    )


def kelvin_proximity(q):
    """2 pi times Ferreira's G_R, using ber2 + j bei2 = J2(q exp(3 j pi/4))."""
    order2 = jv(2, q * np.exp(0.75j * np.pi))
    return (
        -4.0
        * math.pi**2
        * q
        * (order2.real * berp(q) + order2.imag * beip(q))
        / (ber(q) ** 2 + bei(q) ** 2)
    )


def scaled_bessel(q):
    """(z/2) I0(z)/I1(z) from exponentially scaled ive, independent of ber/bei."""
    z = q * np.exp(0.25j * np.pi)
    return 0.5 * z * ive(0, z) / ive(1, z)


def medhurst_phi(spacing, q=200.0, radius=0.05):
    """The model in Medhurst's normalisation: R_ac over the asymptotic straight-wire R.

    His denominator is rho l/(pi d delta) = R_dc q/(2 sqrt2), and his table is the
    f -> infinity limit, so the model is evaluated deep in the asymptotic regime.
    """
    diameter = 1e-3
    coil = Solenoid(radius, 1000.0 * diameter / spacing, 1000, diameter)
    frequency = 2.0 * q**2 * resistivity() / (math.pi * mu_0 * diameter**2)
    return ac_ratio(coil, frequency) / (q / (2.0 * math.sqrt(2.0)))


def relative(values, reference):
    """Elementwise relative deviation of ``values`` from ``reference``."""
    return np.abs(np.asarray(values) / np.asarray(reference) - 1.0)


def outside_dip():
    """Mask of ``Q_GRID`` away from scipy's Kelvin branch change."""
    return (Q_GRID < CEPHES_DIP[0]) | (Q_GRID > CEPHES_DIP[1])


def test_skin_ratio_matches_the_scipy_kelvin_form():
    """1.3e-9 over q = 0.01..100, and 4.4e-13 away from scipy's own branch change."""
    error = relative([skin_ratio(q) for q in Q_GRID], [kelvin_skin(q) for q in Q_GRID])
    assert error.max() < 2e-9
    assert error[outside_dip()].max() < 1e-12


def test_the_residual_against_scipy_is_scipys():
    """Ours and an independent ive evaluation agree to 1.3e-13 where scipy does not."""
    ours = np.array([skin_ratio(q) for q in Q_GRID])
    scaled = np.array([scaled_bessel(q).real for q in Q_GRID])
    assert relative(ours, scaled).max() < 1e-12
    assert relative([kelvin_skin(q) for q in Q_GRID], scaled).max() > 1e-9


def test_both_branches_are_continuous_across_the_crossover():
    """The series and asymptotic forms agree to 1e-12 either side of q = 26."""
    edge = losses._CROSSOVER  # pylint: disable=protected-access
    step = np.spacing(edge) * 4.0
    assert skin_ratio(edge - step) == pytest.approx(skin_ratio(edge + step), rel=1e-12)
    assert proximity_ratio(edge - step) == pytest.approx(
        proximity_ratio(edge + step), rel=1e-12
    )


def test_skin_ratio_low_frequency_series():
    """R_ac/R_dc -> 1 + q^4/192, the next term a fixed negative multiple of q^8.

    q = d/(delta sqrt2), so this is 1 + (a/delta)^4/48; the design's 1 + q^4/48
    holds only for q = a/delta and is four times too large in the q it defines.
    """
    q = np.geomspace(0.1, 0.4, 15)
    ratio = np.array([skin_ratio(v) for v in q])
    assert relative(ratio - 1.0, q**4 / 192.0).max() < 1.1e-4
    assert relative(ratio - 1.0, q**4 / 48.0).min() > 0.749
    residual = (ratio - 1.0 - q**4 / 192.0) / q**8
    assert np.all(residual < 0.0)
    assert relative(residual, residual[-1]).max() < 1e-3


def test_skin_ratio_high_frequency_series():
    """R_ac/R_dc -> q/(2 sqrt2) + 1/4 + 3 sqrt2/(32 q), the residual falling as 1/q^3."""
    q = np.geomspace(50.0, 1000.0, 25)
    series = q / (2.0 * math.sqrt(2.0)) + 0.25 + 3.0 * math.sqrt(2.0) / (32.0 * q)
    ratio = np.array([skin_ratio(v) for v in q])
    assert relative(ratio, series).max() < 1e-7
    residual = (ratio - series) * q**3
    assert relative(residual[-5:], -63.0 / (256.0 * math.sqrt(2.0))).max() < 0.01


def test_proximity_ratio_matches_ferreiras_kelvin_form():
    """G/(2 pi) is Ferreira's G_R to 1.6e-9, and to 6.8e-13 off scipy's branch change."""
    error = relative(
        [proximity_ratio(q) for q in Q_GRID], [kelvin_proximity(q) for q in Q_GRID]
    )
    assert error.max() < 2e-9
    assert error[outside_dip()].max() < 1e-12


def test_proximity_ratio_limits():
    """G -> pi^2 q^4/4 at low frequency and 2 pi^2 (sqrt2 q - 1) at high frequency.

    The first is the elementary uniform-field eddy loss pi sigma omega^2 B^2 a^4/8
    per unit length; the second is surface loss over the field-facing half.
    """
    low = np.geomspace(0.01, 0.25, 25)
    ratio = np.array([proximity_ratio(v) for v in low])
    assert relative(ratio, math.pi**2 * low**4 / 4.0).max() < 2e-4
    high = np.geomspace(100.0, 1000.0, 25)
    series = 2.0 * math.pi**2 * (math.sqrt(2.0) * high - 1.0)
    ratio = np.array([proximity_ratio(v) for v in high])
    assert relative(ratio, series).max() < 2e-5
    residual = (ratio - series) * high
    assert relative(residual[5:], residual[-1]).max() < 1e-3


def test_skin_depth_and_kelvin_argument():
    """delta = sqrt(2 rho/(omega mu0)) halves per doubling of sqrt(f); q = d/(delta sqrt2)."""
    rho = resistivity()
    assert skin_depth(1e6, rho) == pytest.approx(
        math.sqrt(rho / (math.pi * 1e6 * 4e-7 * math.pi)), rel=1e-12
    )
    assert skin_depth(4e5, rho) == pytest.approx(0.5 * skin_depth(1e5, rho), rel=1e-12)
    assert kelvin_argument(1e-3, 1e5, rho) == pytest.approx(
        1e-3 / (math.sqrt(2.0) * skin_depth(1e5, rho)), rel=1e-12
    )


def test_resistivity_tracks_temperature():
    """Annealed copper, 1.7241e-8 ohm m at 20 C, alpha = 3.93e-3 /K."""
    assert resistivity() == pytest.approx(1.7241e-8, rel=1e-12)
    assert resistivity(120.0) / resistivity(20.0) == pytest.approx(1.393, rel=1e-12)
    assert dc_resistance(SECONDARY, temperature=120.0) / dc_resistance(
        SECONDARY
    ) == pytest.approx(1.393, rel=1e-12)


def test_proximity_vanishes_as_the_turns_spread_out():
    """R_ac/R_dc -> the isolated-wire skin ratio as pitch/diameter grows."""
    q = kelvin_argument(SECONDARY.wire_diameter, 3e5, resistivity())
    excess = np.array(
        [
            ac_ratio(Solenoid(0.076, length, 1000, 4e-4), 3e5) - skin_ratio(q)
            for length in (0.5, 1.0, 2.0, 5.0, 20.0, 100.0)
        ]
    )
    assert np.all(np.diff(excess) < 0.0)
    assert excess[-1] / excess[0] == pytest.approx(2.5e-5, rel=0.05)
    assert excess[-1] / skin_ratio(q) < 1e-4


def test_proximity_grows_as_the_turns_are_packed_closer():
    """The proximity term scales exactly as (d/p)^2 at fixed wire and frequency."""
    q = kelvin_argument(4e-4, 3e5, resistivity())
    pitch = np.array([4e-3, 2e-3, 1e-3, 5e-4])
    excess = np.array(
        [
            ac_ratio(Solenoid(0.076, p * 1000, 1000, 4e-4), 3e5) - skin_ratio(q)
            for p in pitch
        ]
    )
    assert np.all(np.diff(excess) > 0.0)
    assert relative(excess * pitch**2, excess[0] * pitch[0] ** 2).max() < 1e-12


def test_resistance_rises_with_frequency():
    """R is monotone in f and approaches the sqrt(f) surface-loss law."""
    f = np.geomspace(1e4, 1e7, 40)
    r = np.array([dc_resistance(SECONDARY) * ac_ratio(SECONDARY, v) for v in f])
    assert np.all(np.diff(r) > 0.0)
    assert np.all(r > dc_resistance(SECONDARY))
    slope = np.diff(np.log(r[-6:])) / np.diff(np.log(f[-6:]))
    assert np.all(np.diff(slope) < 0.0)
    assert slope[-1] == pytest.approx(0.5, abs=0.02)


def test_resistance_falls_with_diameter_until_proximity_dominates():
    """At fixed f and pitch R falls as 1/d^2, then rises as the d^4 proximity term wins."""
    d = np.geomspace(5e-5, 5e-4, 40)
    coils = [Solenoid(0.076, 0.5, 1000, v) for v in d]
    r = np.array([dc_resistance(c) * ac_ratio(c, 3e5) for c in coils])
    turn = int(np.argmin(r))
    assert 0 < turn < len(d) - 1
    assert np.all(np.diff(r[: turn + 1]) < 0.0)
    assert np.all(np.diff(r[turn:]) > 0.0)


def test_section_resistance_lines_up_with_the_ladder():
    """The R vector indexes the same rungs as the ladder's L, and sums to the whole coil."""
    for sections in (37, 100):
        rungs = ladder(DRSSTC, sections)
        r = section_resistance(SECONDARY, 2e5, sections)
        assert r.shape == (len(rungs.L),)
        assert np.all(r > 0.0)
        turns = np.bincount(
            inductance.turn_groups(len(SECONDARY.discretise()), sections)
        )
        assert relative(r / turns, r[0] / turns[0]).max() < 1e-12
        assert r.sum() == pytest.approx(
            dc_resistance(SECONDARY) * ac_ratio(SECONDARY, 2e5), rel=1e-12
        )


def test_mode_quality_reduces_to_omega_over_i_r_i():
    """Modes are normalised to i L i = 1, so the numerator of Q is unity."""
    modes = resonance(DRSSTC, modes=2)
    inductances = inductance.section_inductance_matrix(SECONDARY, DRSSTC.sections)
    r = section_resistance(SECONDARY, modes.f[0], DRSSTC.sections)
    assert modes.i[0] @ inductances @ modes.i[0] == pytest.approx(1.0, rel=1e-8)
    assert mode_quality(inductances, modes.i[0], modes.f[0], r) == pytest.approx(
        2.0 * math.pi * modes.f[0] / (r @ modes.i[0] ** 2), rel=1e-8
    )


def test_resonant_resistance_converges_in_one_iteration():
    """The damping shift is 1/(8 Q^2); a second pass moves f_res by under 1e-9 %."""
    modes = resonance(DRSSTC, modes=1)
    once = resonant_resistance(DRSSTC, modes)
    twice = resonant_resistance(DRSSTC, modes, iterations=2)
    assert len(once) == DRSSTC.sections
    assert once.f < modes.f[0]
    assert 1.0 - once.f / modes.f[0] == pytest.approx(0.125 / once.q**2, rel=1e-6)
    assert abs(twice.f / once.f - 1.0) < 1e-11
    assert relative(twice.r, once.r).max() < 1e-6


def test_quality_factor_rises_with_mode_order():
    """Q of each mode uses that mode's own frequency and current profile."""
    modes = resonance(DRSSTC, modes=3)
    quality = quality_factor(DRSSTC, modes)
    assert len(quality) == 3
    hot = quality_factor(DRSSTC, modes, temperature=120.0)
    assert quality[0] / 1.393 < hot[0] < quality[0]


def test_capacitor_esr_and_dielectric_conductance():
    """ESR = DF/(omega C) and G = omega C tan(delta), reciprocal in the same quantities."""
    omega = 2.0 * math.pi * 3e5
    assert capacitor_esr(0.15e-6, 3e5, 2e-3) == pytest.approx(
        2e-3 / (omega * 0.15e-6), rel=1e-12
    )
    assert dielectric_conductance(20e-12, 3e5, 0.02) == pytest.approx(
        omega * 20e-12 * 0.02, rel=1e-12
    )
    assert capacitor_esr(1e-6, 3e5, 0.01) * dielectric_conductance(
        1e-6, 3e5, 0.01
    ) == pytest.approx(1e-4, rel=1e-12)


def test_medhurst_proximity_factor():
    """Exact for sparse windings, over-predicting monotonically as the turns close up.

    Against Medhurst's l/D = infinity column the error is +0.3, +0.9, +3.3, +15.7, +47.7
    and +73.6 % at d/s = 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, and one-signed: a uniform external
    field ignores the mutual screening of adjacent turns, which grows as they touch.
    """
    spacing = np.array(sorted(MEDHURST_INFINITE))
    error = np.array([medhurst_phi(s) / MEDHURST_INFINITE[s] - 1.0 for s in spacing])
    assert np.all(error > 0.0)
    assert np.all(np.diff(error) > 0.0)
    assert error[spacing <= 0.2].max() < 0.01
    assert error[spacing == 0.3][0] < 0.035
    assert error[-1] == pytest.approx(0.736, abs=0.005)
    against_ten = np.array([medhurst_phi(s) / MEDHURST_TEN[s] - 1.0 for s in spacing])
    assert np.all(against_ten > error)


def test_butterworths_tabulated_f_and_g():
    """1 + F and G reproduce Butterworth's own table to its printed rounding.

    5e-4 except at z = 0.5, where his G = 0.00097 carries only two significant
    figures and the 5e-3 residual is that rounding.
    """
    error = np.array(
        [
            [
                relative(skin_ratio(z), f),
                relative(proximity_ratio(z) / (16.0 * math.pi**2), g),
            ]
            for z, (f, g) in BUTTERWORTH.items()
        ]
    )
    assert error[1:].max() < 5e-4
    assert error.max() < 5e-3


def test_the_close_wound_excess_is_butterworths_eddy_current_reaction():
    """u = 4 pi^2 (H_e p/I)^2 = 9.87, Butterworth's l/D = infinity field factor.

    The close-wound infinite solenoid is then 1 + pi^2/2 = 5.93, the 5.94 he quotes
    before the neighbours' eddy-current reaction field reduces it to Medhurst's 3.41.
    """
    coil = Solenoid(0.05, 0.5, 500, 5e-4)
    reduced = field_reach(coil) * coil.pitch / (0.5 * coil.wire_diameter)
    assert 4.0 * math.pi**2 * reduced**2 == pytest.approx(9.87, abs=0.005)
    assert medhurst_phi(1.0, q=1e4) == pytest.approx(5.94, abs=0.01)
    assert medhurst_phi(1.0, q=1e4) / MEDHURST_INFINITE[1.0] == pytest.approx(
        5.94 / 3.41, rel=0.01
    )


def test_the_model_carries_no_length_to_diameter_dependence():
    """Medhurst's phi spans 23 % over l/D at d/s = 0.8; this long-solenoid model is flat."""
    tabulated = np.array(MEDHURST_LENGTH_08)
    assert tabulated.max() / tabulated.min() - 1.0 > 0.23
    flat = [medhurst_phi(0.8, radius=r) for r in (0.02, 0.05, 0.2)]
    assert flat == pytest.approx([flat[0]] * len(flat), rel=1e-12)


def test_the_proximity_factor_approaches_its_high_frequency_limit_from_below():
    """phi -> 1 + pi^2 (d/p)^2/2, the ratio of the two leading asymptotic terms."""
    for spacing in (0.2, 0.5, 1.0):
        limit = 1.0 + math.pi**2 * spacing**2 / 2.0
        phi = np.array([medhurst_phi(spacing, q) for q in (20.0, 60.0, 200.0, 600.0)])
        assert np.all(np.diff(np.abs(phi - limit)) < 0.0)
        assert phi[-1] == pytest.approx(limit, rel=2e-3)


def test_quality_factor_lands_in_the_published_drsstc_band():
    """Q of 367 at 173 kHz, against Denicolai's 326 measured on a 5 kW coil at 66 kHz.

    Thesis eq. 6-69, eltem.fi/lthesis.pdf; Kaizer's DRSSTC guide tabulates 118-344
    calculated for large secondaries and 207-505 for very large ones.
    """
    modes = resonance(DRSSTC, modes=1)
    quality = quality_factor(DRSSTC, modes)[0]
    assert 1.6e5 < modes.f[0] < 1.8e5
    assert 100.0 < quality < 505.0
