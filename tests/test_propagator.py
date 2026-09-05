"""Piecewise-LTI propagators, against scipy.linalg.expm of the augmented matrix."""

import numpy as np
import pytest
from scipy.linalg import block_diag, expm

from thirdlight.solver.propagator import Propagator, derivative

EPS = np.spacing(1.0)
RNG = np.random.default_rng(20240905)
# Dyadic, so the tabulated path resolves them exactly and the semigroup is not
# perturbed by the dropped remainder below step/2^levels.
FRACTIONS = (0.125, 0.375, 0.5, 1023.0 / 1024.0, 1.0)


def augmented(a, b, s):
    """(Phi, Gamma) from expm([[A, B], [0, 0]] s) = [[Phi, Gamma], [0, I]]."""
    a = np.asarray(a, dtype=float)
    b = np.reshape(np.asarray(b, dtype=float), (len(a), -1))
    n, p = b.shape
    block = np.zeros((n + p, n + p))
    block[:n, :n] = a
    block[:n, n:] = b
    grown = expm(block * s)
    return grown[:n, :n], grown[:n, n:]


def relative(value, reference):
    """Frobenius-norm relative deviation."""
    return np.linalg.norm(value - reference) / np.linalg.norm(reference)


def series_rlc(r=5.0, inductance=100e-6, capacitance=100e-9):
    """[i, v_C] of a series RLC driven by a source voltage; f0 = 50 kHz."""
    a = np.array([[-r / inductance, -1.0 / inductance], [1.0 / capacitance, 0.0]])
    return a, np.array([1.0 / inductance, 0.0]), 1.0 / (256 * 50e3)


def coupled_resonators(k=0.2):
    """Two RLC tanks sharing a mutual inductance, both tuned to 53 kHz.

    In energy coordinates y = G' i, w = C^(1/2) v with L = G G', so that the
    reactive blocks are exact negative transposes and A is balanced; the raw
    [i, v] form spans eleven decades and composing its propagators loses those
    digits to cancellation.
    """
    l_1, l_2, c_1, c_2, r_1, r_2 = 60e-6, 60e-3, 0.15e-6, 150e-12, 0.1, 100.0
    mutual = k * np.sqrt(l_1 * l_2)
    lower = np.linalg.inv(np.linalg.cholesky([[l_1, mutual], [mutual, l_2]]))
    cross = lower @ np.diag(1.0 / np.sqrt([c_1, c_2]))
    a = np.block(
        [
            [-lower @ np.diag([r_1, r_2]) @ lower.T, -cross],
            [cross.T, np.zeros((2, 2))],
        ]
    )
    return (
        a,
        np.concatenate([lower @ np.array([1.0, 0.0]), np.zeros(2)]),
        1.0 / (256 * 53e3),
    )


def random_stable(n=8):
    """Random A shifted to put every eigenvalue in the left half plane."""
    a = RNG.normal(size=(n, n))
    shifted = a - np.eye(n) * (np.linalg.eigvals(a).real.max() + 1.0)
    return shifted, RNG.normal(size=(n, 2)), 0.05


def decades():
    """Diagonal A whose |lam h| straddles the series/expm1 crossover, with a null row."""
    lam = -np.array([1e3, 3e2, 1e2, 3e1, 1e1, 0.0])
    return np.diag(lam), np.arange(1.0, 1.0 + len(lam)), 1e-3


CASES = {
    "rlc": series_rlc(),
    "coupled": coupled_resonators(),
    "random": random_stable(),
    "decades": decades(),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("cond_max", [1e8, 0.0])
def test_matches_expm(name, cond_max):
    a, b, step = CASES[name]
    propagator = Propagator.build(a, b, step, cond_max=cond_max)
    assert propagator.exact == (cond_max > 0.0)
    for fraction in FRACTIONS:
        phi, gamma = propagator.at(fraction * step)
        phi_ref, gamma_ref = augmented(a, b, fraction * step)
        assert relative(phi, phi_ref) < 1e-12
        assert relative(gamma, gamma_ref) < 1e-12


@pytest.mark.parametrize("name", ["rlc", "random", "coupled"])
def test_gamma_is_inverse_form(name):
    a, b, step = CASES[name]
    b = np.reshape(b, (len(a), -1))
    phi, gamma = Propagator.build(a, b, step).at(0.375 * step)
    assert relative(gamma, np.linalg.solve(a, (phi - np.eye(len(a))) @ b)) < 1e-12


def test_zero_eigenvalue_integrates_input():
    a, b, step = CASES["decades"]
    propagator = Propagator.build(a, b, step)
    for fraction in FRACTIONS:
        s = fraction * step
        gamma = propagator.at(s)[1]
        assert gamma[-1, 0] == pytest.approx(s * b[-1], rel=1e-15)
        assert relative(gamma, augmented(a, b, s)[1]) < 1e-12


@pytest.mark.parametrize("cond_max", [1e8, 0.0])
def test_semigroup(cond_max):
    a, b, step = CASES["random"]
    propagator = Propagator.build(a, b, step, cond_max=cond_max)
    first, second = 0.375 * step, 0.5 * step
    phi_1, gamma_1 = propagator.at(first)
    phi_2, gamma_2 = propagator.at(second)
    phi, gamma = propagator.at(first + second)
    assert relative(phi_2 @ phi_1, phi) < 1e-12
    assert relative(phi_2 @ gamma_1 + gamma_2, gamma) < 1e-12


def test_defective_falls_back_and_still_matches():
    a, b, step = np.array([[-1.0, 1.0], [0.0, -1.0]]), np.array([1.0, 2.0]), 0.1
    propagator = Propagator.build(a, b, step)
    assert propagator.exact is False
    for fraction in FRACTIONS:
        phi, gamma = propagator.at(fraction * step)
        phi_ref, gamma_ref = augmented(a, b, fraction * step)
        assert relative(phi, phi_ref) < 1e-12
        assert relative(gamma, gamma_ref) < 1e-12


def test_eigen_failure_falls_back(monkeypatch):
    def refuse(_):
        raise np.linalg.LinAlgError("did not converge")

    monkeypatch.setattr(np.linalg, "eig", refuse)
    a, b, step = CASES["rlc"]
    propagator = Propagator.build(a, b, step)
    assert propagator.exact is False
    assert relative(propagator.at(step)[0], augmented(a, b, step)[0]) < 1e-12


def test_free_response_conserves_energy():
    """Undamped modes give an orthogonal Phi; the free response then only drifts by rounding.

    Each step rounds the state by O(eps), so the energy bound over ``steps`` of
    them is 2 steps eps; 1e-13 is below that floor for any run of this length.
    """
    steps = 10_000
    omega = 2.0 * np.pi * np.array([50e3, 130e3, 210e3, 290e3])
    a = block_diag(*(np.array([[0.0, -w], [w, 0.0]]) for w in omega))
    propagator = Propagator.build(a, np.zeros(8), 1.0 / (256 * 50e3))
    phi = propagator.at(propagator.step)[0]
    assert np.linalg.norm(phi.T @ phi - np.eye(8)) < 8.0 * EPS
    x = RNG.normal(size=8)
    energy = x @ x
    for _ in range(steps):
        x = propagator.advance(x, 0.0)
    assert x @ x == pytest.approx(energy, rel=2.0 * steps * EPS)


@pytest.mark.parametrize("cond_max", [1e8, 0.0])
def test_advance_matches_at(cond_max):
    a, b, step = CASES["random"]
    propagator = Propagator.build(a, b, step, cond_max=cond_max)
    x, u = RNG.normal(size=len(a)), RNG.normal(size=2)
    for fraction in FRACTIONS + (None,):
        s = None if fraction is None else fraction * step
        phi, gamma = propagator.at(step if s is None else s)
        assert propagator.advance(x, u, s) == pytest.approx(
            phi @ x + gamma @ u, rel=1e-12
        )


@pytest.mark.parametrize("cond_max", [1e8, 0.0])
def test_scalar_input(cond_max):
    a, b, step = CASES["rlc"]
    propagator = Propagator.build(a, b, step, cond_max=cond_max)
    x = np.array([1.5, -300.0])
    assert propagator.advance(x, 400.0) == pytest.approx(propagator.advance(x, [400.0]))
    assert propagator.advance(x, 400.0, 0.5 * step) == pytest.approx(
        propagator.at(0.5 * step)[0] @ x + propagator.at(0.5 * step)[1] @ [400.0]
    )


def test_derivative_matches_the_propagator_limit():
    a, b, step = CASES["rlc"]
    x, u = np.array([1.5, -300.0]), 400.0
    slope = derivative(a, b, x, u)
    assert slope == pytest.approx(a @ x + b * u)
    tiny = step * 1e-6
    moved = Propagator.build(a, b, step).advance(x, u, tiny)
    assert (moved - x) / tiny == pytest.approx(slope, rel=1e-5)


@pytest.mark.parametrize("s", [-1e-12, 1.0000001])
def test_at_rejects_out_of_range(s):
    propagator = Propagator.build(*CASES["random"])
    with pytest.raises(ValueError):
        propagator.at(s * propagator.step)


def test_build_rejects_nonpositive_step():
    a, b, _ = CASES["rlc"]
    with pytest.raises(ValueError):
        Propagator.build(a, b, 0.0)


@pytest.mark.parametrize("levels", [8, 32])
def test_sub_step_resolution(levels):
    """A non-dyadic s composes to the exact propagator of the nearest step/2^levels."""
    a, b, step = CASES["random"]
    propagator = Propagator.build(a, b, step, cond_max=0.0, levels=levels)
    s = step / np.pi
    resolved = round(s / step * 2**levels) / 2**levels * step
    assert abs(resolved - s) <= step * 2.0 ** -(levels + 1)
    phi, gamma = propagator.at(s)
    phi_ref, gamma_ref = augmented(a, b, resolved)
    assert relative(phi, phi_ref) < 1e-12
    assert relative(gamma, gamma_ref) < 1e-12


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("cond_max", [1e8, 0.0])
def test_a_zero_step_is_exact_not_reconstructed(name, cond_max):
    """Phi(0) is the identity and Gamma(0) zero to the bit, on both paths.

    The event locator brackets a functional against the value it started from and
    a switching instant pins that value to zero, so a rounding-sized Phi(0) x - x
    puts the bracket on the wrong side and commutes the bridge again at once.
    """
    a, b, step = CASES[name]
    propagator = Propagator.build(a, b, step, cond_max=cond_max)
    phi, gamma = propagator.at(0.0)
    assert np.array_equal(phi, np.eye(len(a)))
    assert not gamma.any()
    x = RNG.normal(size=len(a))
    u = RNG.normal(size=np.atleast_2d(np.asarray(b).T).shape[0])
    moved = propagator.advance(x, u, 0.0)
    assert np.array_equal(moved, x)
    assert moved is not x
