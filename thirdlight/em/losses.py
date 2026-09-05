"""Winding AC resistance, and the lumped tank and former loss terms.

Skin and proximity loss are exact cylinder diffusion solutions sharing one kernel,
I0(z)/I1(z) at z = q exp(j pi/4), q = d/(delta sqrt2). Together they are
Butterworth's R(1 + F + u G (d/c)^2) for a long solenoid, at his u = pi^2.
"""

import math
from dataclasses import dataclass

import numba
import numpy as np
from scipy.constants import mu_0

from thirdlight.backend import array_namespace, kernel
from thirdlight.em import inductance

RHO_COPPER_20C = 1.7241e-8
ALPHA_COPPER = 3.93e-3

_ROOT_J = complex(math.sqrt(0.5), math.sqrt(0.5))
_CROSSOVER = 26.0
_SERIES_TERMS = 128
_SERIES_TOL = 1e-18
# I0/I1 ~ sum c_k z^-k, the Hankel expansion of I0 divided by that of I1.
_ASYMPTOTIC = (
    1.0,
    1.0 / 2,
    3.0 / 8,
    3.0 / 8,
    63.0 / 128,
    27.0 / 32,
    1899.0 / 1024,
    81.0 / 16,
    543483.0 / 32768,
    32427.0 / 512,
    72251109.0 / 262144,
)


@dataclass(frozen=True)
class Resistance:
    """Per-section AC resistance, the frequency it holds at, and that mode's Q."""

    r: np.ndarray
    f: float
    q: float

    def __len__(self):
        return len(self.r)


@kernel
def skin_depth(frequency, rho):
    """delta = sqrt(2 rho / (omega mu0)), in metres."""
    return math.sqrt(rho / (math.pi * frequency * mu_0))


@kernel
def kelvin_argument(diameter, frequency, rho):
    """q = d / (delta sqrt 2) = sqrt(2) a / delta, the argument of ber and bei."""
    return diameter / (math.sqrt(2.0) * skin_depth(frequency, rho))


@kernel
def bessel_ratio(q):
    """(z/2) I0(z)/I1(z) at z = q exp(j pi/4), where I0(z) = ber(q) + j bei(q).

    As N/D with x = j q^2/4, N = sum x^k/(k!)^2, D = sum x^k/(k!(k+1)!) this is
    the ber/bei form with z/2 cancelled; it loses 0.127 q digits to cancellation,
    so above the measured crossover q = 26 the asymptotic series is used instead.
    """
    z = q * _ROOT_J
    if q < _CROSSOVER:
        x = 0.25j * q * q
        num = den = 0.0j
        term_n = term_d = 1.0 + 0.0j
        for k in range(_SERIES_TERMS):
            num += term_n
            den += term_d
            if abs(term_n) < _SERIES_TOL * abs(num):
                break
            term_n = term_n * x / ((k + 1) * (k + 1))
            term_d = term_d * x / ((k + 1) * (k + 2))
        return num / den
    total = 0.0j
    power = 1.0 + 0.0j
    for c in _ASYMPTOTIC:
        total += c * power
        power /= z
    return 0.5 * z * total


@kernel
def skin_ratio(q):
    """R_ac/R_dc of an isolated round wire, Re[(z/2) I0(z)/I1(z)].

    Internal impedance Z = rho gamma I0(gamma a)/(2 pi a I1(gamma a)) with
    gamma = (1+j)/delta and gamma a = z. Limits: 1 + q^4/192 and
    q/(2 sqrt2) + 1/4 + 3 sqrt2/(32 q).
    """
    return bessel_ratio(q).real


@kernel
def proximity_ratio(q):
    """G(q) in R_prox/R_dc = G(q) (a H_e/I)^2, for uniform transverse field H_e.

    Exact cylinder eddy loss: inside A_z = D I1(gamma r) sin(theta) with
    D = -2 mu0 H_e/(gamma I0(z)); the Lommel integral of sigma |E|^2/2 leaves
    2 pi^2 q^2 Im(t)/|t|^2, t being :func:`bessel_ratio`. G is 16 pi^2 times
    Butterworth's tabulated G and 2 pi times Ferreira's G_R.
    """
    t = bessel_ratio(q)
    return (
        2.0 * math.pi * math.pi * q * q * t.imag / (t.real * t.real + t.imag * t.imag)
    )


@numba.njit(cache=True)
def _segment_sum(groups, values, sections):
    """Sum ``values`` into ``sections`` bins labelled by ``groups``."""
    out = np.zeros(sections)
    for i in range(groups.shape[0]):
        out[groups[i]] += values[i]
    return out


def resistivity(temperature=20.0, reference=RHO_COPPER_20C, alpha=ALPHA_COPPER):
    """Conductor resistivity at ``temperature`` in C, linear in temperature.

    Defaults are annealed copper at 100 % IACS, 1.7241e-8 ohm m at 20 C, with
    alpha = 3.93e-3 /K.
    """
    return reference * (1.0 + alpha * (temperature - 20.0))


def field_reach(coil):
    """a H_e / I at one turn of a long solenoid, d/(4 p).

    The winding is a sheet of density K = I/p, H stepping from K inside to zero
    outside, so the conductor sits in the mean field K/2. That is Butterworth's
    field factor u = 4 pi^2 (H_e p/I)^2 = pi^2, his l/D = infinity entry.
    """
    return 0.25 * coil.wire_diameter / coil.pitch


def ac_ratio(coil, frequency, rho=None, **kwargs):
    """R_ac/R_dc of a long single-layer solenoid: skin plus proximity."""
    rho = resistivity(**kwargs) if rho is None else rho
    q = kelvin_argument(coil.wire_diameter, frequency, rho)
    return skin_ratio(q) + proximity_ratio(q) * field_reach(coil) ** 2


def dc_resistance(coil, rho=None, **kwargs):
    """Total DC resistance of the winding, rho l_wire / A."""
    rho = resistivity(**kwargs) if rho is None else rho
    return rho * coil.wire_length / (0.25 * math.pi * coil.wire_diameter**2)


def section_resistance(coil, frequency, sections, rho=None, **kwargs):
    """Per-section AC resistance, grouped as :func:`inductance.turn_groups` groups turns.

    The grouping is the one :func:`thirdlight.secondary.ladder` uses, so the
    vector indexes the same rungs as the ladder's L and C.
    """
    rho = resistivity(**kwargs) if rho is None else rho
    turns = coil.discretise()
    length = turns.n * np.hypot(2.0 * np.pi * turns.a, coil.pitch)
    per_turn = rho * length / (0.25 * math.pi * coil.wire_diameter**2)
    return _segment_sum(
        inductance.turn_groups(len(turns), sections),
        per_turn * ac_ratio(coil, frequency, rho),
        sections,
    )


def mode_quality(inductances, current, frequency, r):
    """Q = omega (i L i) / (i R i) for one mode current profile.

    Time-averaged stored energy is the peak magnetic energy (1/2) i L i and mean
    dissipation is (1/2) sum_k R_k i_k^2 in the same normalisation, so Q = omega
    W/P is their quotient; at the Modes normalisation i L i = 1 it is omega/(i R i).
    """
    xp = array_namespace()
    i = xp.asarray(current)
    stored = i @ xp.asarray(inductances) @ i
    return float(2.0 * math.pi * frequency * stored / (xp.asarray(r) @ (i * i)))


def quality_factor(design, modes, **kwargs):
    """Unloaded Q of each mode, each evaluated at its own frequency."""
    sections = modes.v.shape[1]
    inductances = inductance.section_inductance_matrix(design.secondary, sections)
    return np.array(
        [
            mode_quality(
                inductances,
                modes.i[m],
                f,
                section_resistance(design.secondary, f, sections, **kwargs),
            )
            for m, f in enumerate(modes.f)
        ]
    )


def resonant_resistance(design, modes, mode=0, iterations=1, **kwargs):
    """Per-section R at the damped resonance of ``mode``, and that resonance.

    R depends on frequency and the damped resonance f = f0 sqrt(1 - 1/(4 Q^2))
    depends on R, so the two are a fixed point started from the lossless f0. One
    pass suffices: the shift is 1/(8 Q^2) and R varies as sqrt(f) at most.
    """
    sections = modes.v.shape[1]
    inductances = inductance.section_inductance_matrix(design.secondary, sections)
    f = f_0 = modes.f[mode]
    r = quality = None
    for _ in range(iterations):
        r = section_resistance(design.secondary, f, sections, **kwargs)
        quality = mode_quality(inductances, modes.i[mode], f, r)
        f = f_0 * math.sqrt(1.0 - 0.25 / (quality * quality))
    return Resistance(r=r, f=f, q=quality)


def capacitor_esr(capacitance, frequency, dissipation_factor):
    """Series resistance of a capacitor of dissipation factor DF: ESR = DF/(omega C)."""
    return dissipation_factor / (2.0 * math.pi * frequency * capacitance)


def dielectric_conductance(capacitance, frequency, loss_tangent):
    """Parallel conductance of a lossy dielectric: G = omega C tan(delta)."""
    return 2.0 * math.pi * frequency * capacitance * loss_tangent
