"""Batched eigen-coordinate kernels, against the propagator and locator they mirror."""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from test_propagator import CASES, FRACTIONS, series_rlc
from thirdlight.solver import kernels
from thirdlight.solver.propagator import (  # pylint: disable=protected-access
    Propagator,
    _phi_one,
    derivative,
)
from thirdlight.solver.stepping import _crossing  # pylint: disable=protected-access

DESIGNS = 3
SEED = 20260906
XTOL = 1e-14


@pytest.fixture(name="call", params=[False, True], ids=["compiled", "interpreted"])
def dispatch(request):
    """Invoke a kernel through its compiled dispatcher and through its Python body.

    Under NUMBA_DISABLE_JIT there is no dispatcher to unwrap and the two are one.
    """
    if request.param:
        return lambda k, *args: getattr(k, "py_func", k)(*args)
    return lambda k, *args: k(*args)


def split(arrays):
    """Real and imaginary parts of complex arrays stacked with the design last."""
    stacked = np.stack(arrays, axis=-1)
    return np.ascontiguousarray(stacked.real), np.ascontiguousarray(stacked.imag)


class Batch:  # pylint: disable=too-many-instance-attributes
    """Eigen data, state and held input of several designs, packed design-major."""

    def __init__(self, props, x, u):
        self.props = props
        self.n, self.designs = x.shape
        self.p = u.shape[0]
        self.x = np.ascontiguousarray(x)
        self.u = np.ascontiguousarray(u)
        self.lam_re, self.lam_im = split([q.eigen.lam for q in props])
        self.basis_re, self.basis_im = split([q.eigen.basis for q in props])
        self.inv_re, self.inv_im = split([q.eigen.inverse for q in props])
        self.invb_re, self.invb_im = split([q.eigen.inverse_b for q in props])
        self.y_re, self.y_im = self.zeros(), self.zeros()
        self.ub_re, self.ub_im = self.zeros(), self.zeros()
        self.w_re, self.w_im = self.zeros(), self.zeros()

    def zeros(self):
        """A design-major array of one complex component's worth of state."""
        return np.zeros((self.n, self.designs))

    def coordinates(self, call, d):
        """Fill the modal state and input of design ``d``."""
        call(
            kernels.modal,
            self.inv_re,
            self.inv_im,
            self.x,
            self.y_re,
            self.y_im,
            self.n,
            d,
        )
        call(
            kernels.inject,
            self.invb_re,
            self.invb_im,
            self.u,
            self.ub_re,
            self.ub_im,
            self.n,
            self.p,
            d,
        )

    def functional(self, call, c, d):
        """Fill the functional row of design ``d`` against the eigenbasis."""
        call(
            kernels.row,
            self.basis_re,
            self.basis_im,
            c,
            self.w_re,
            self.w_im,
            self.n,
            d,
        )

    def value(self, call, s, d):
        """The packed functional at sub-step ``s`` of design ``d``."""
        return call(
            kernels.value,
            self.lam_re,
            self.lam_im,
            self.w_re,
            self.w_im,
            self.y_re,
            self.y_im,
            self.ub_re,
            self.ub_im,
            s,
            kernels.SERIES,
            self.n,
            d,
        )

    def bisect(self, call, offset, span, sign, start, d):
        """The packed crossing search over [0, span) of design ``d``."""
        return call(
            kernels.bisect,
            self.lam_re,
            self.lam_im,
            self.w_re,
            self.w_im,
            self.y_re,
            self.y_im,
            self.ub_re,
            self.ub_im,
            offset,
            span,
            sign,
            start,
            span * XTOL,
            kernels.SERIES,
            self.n,
            d,
        )


def batch(name, designs=DESIGNS):
    """Designs of a case scaled apart, so no packed array repeats across the axis.

    Seeded per case, so a draw does not depend on what else the worker ran first.
    """
    a, b, step = CASES[name]
    a = np.asarray(a, dtype=float)
    rng = np.random.default_rng([SEED, list(CASES).index(name), designs])
    props = [Propagator.build(a * (1.0 + 0.05 * k), b, step) for k in range(designs)]
    n, p = len(a), props[0].eigen.inverse_b.shape[1]
    packed = Batch(props, rng.normal(size=(n, designs)), rng.normal(size=(p, designs)))
    return packed, rng


def single(a, b, step, x, u):
    """One design's packed batch, and the propagator it was packed from."""
    prop = Propagator.build(a, b, step)
    x, u = np.asarray(x, dtype=float), np.asarray(u, dtype=float)
    return Batch([prop], x[:, None], u[:, None]), prop


@pytest.mark.parametrize("z", [0j, 1 + 2j, -3 - 0.5j, 0.1j, -40 + 3j])
def test_cexp_matches_numpy(call, z):
    assert complex(*call(kernels.cexp, z.real, z.imag)) == pytest.approx(
        np.exp(z), rel=1e-15
    )


@pytest.mark.parametrize(
    "scale", [0.0, 1e-8, 0.5, 1.0 - 1e-12, 1.0, 1.0 + 1e-12, 2.0, 1e3]
)
@pytest.mark.parametrize("angle", [0.0, 0.7, 2.0, math.pi])
def test_phi_one_matches_the_propagator(call, scale, angle):
    z = kernels.SERIES_LIMIT * scale * complex(math.cos(angle), math.sin(angle))
    s = 2.5e-6
    got = complex(*call(kernels.phi_one, z.real, z.imag, s, kernels.SERIES))
    assert got == pytest.approx(_phi_one(np.array([z]), s)[0], rel=1e-14)


@pytest.mark.parametrize("name", list(CASES))
def test_modal_advance_restore_matches_the_propagator(call, name):
    packed, _ = batch(name)
    out_re, out_im, moved = packed.zeros(), packed.zeros(), packed.zeros()
    for d in range(packed.designs):
        packed.coordinates(call, d)
        for fraction in FRACTIONS:
            s = fraction * packed.props[d].step
            call(
                kernels.advance,
                packed.lam_re,
                packed.lam_im,
                packed.y_re,
                packed.y_im,
                packed.ub_re,
                packed.ub_im,
                s,
                kernels.SERIES,
                out_re,
                out_im,
                packed.n,
                d,
            )
            call(
                kernels.restore,
                packed.basis_re,
                packed.basis_im,
                out_re,
                out_im,
                moved,
                packed.n,
                d,
            )
            reference = packed.props[d].advance(packed.x[:, d], packed.u[:, d], s)
            assert moved[:, d] == pytest.approx(
                reference, rel=1e-12, abs=1e-12 * np.linalg.norm(reference)
            )


@pytest.mark.parametrize("name", list(CASES))
def test_value_matches_the_evaluator(call, name):
    packed, rng = batch(name)
    c, offset = rng.normal(size=(packed.n, packed.designs)), 0.25
    for d in range(packed.designs):
        packed.coordinates(call, d)
        packed.functional(call, c, d)
        evaluate = packed.props[d].evaluator(
            packed.x[:, d], packed.u[:, d], (c[:, d], offset)
        )
        floor = 1e-11 * np.linalg.norm(c[:, d]) * np.linalg.norm(packed.x[:, d])
        for fraction in FRACTIONS:
            s = fraction * packed.props[d].step
            got = packed.value(call, s, d) + offset
            assert got == pytest.approx(evaluate(s), rel=1e-11, abs=floor)


@pytest.mark.parametrize("name", list(CASES))
def test_bisect_matches_brentq(call, name):
    packed, rng = batch(name)
    c = rng.normal(size=(packed.n, packed.designs))
    for d in range(packed.designs):
        prop, span = packed.props[d], packed.props[d].step
        x, u = packed.x[:, d], packed.u[:, d]
        offset = -0.5 * float(c[:, d] @ (x + prop.advance(x, u, span)))
        start = float(c[:, d] @ x) + offset
        sign = math.copysign(1.0, start)
        packed.coordinates(call, d)
        packed.functional(call, c, d)
        got = packed.bisect(call, offset, span, sign, start, d)
        evaluate = prop.evaluator(x, u, (c[:, d], offset))
        assert got == pytest.approx(
            brentq(evaluate, 0.0, span, xtol=span * XTOL), abs=4 * span * XTOL
        )


def test_bisect_reports_a_functional_already_across_at_once(call):
    packed, rng = batch("rlc", designs=1)
    c = rng.normal(size=(packed.n, 1))
    packed.coordinates(call, 0)
    packed.functional(call, c, 0)
    span = packed.props[0].step
    start = float(c[:, 0] @ packed.x[:, 0])
    sign = -math.copysign(1.0, start)
    assert packed.bisect(call, 0.0, span, sign, start, 0) == 0.0


def test_bisect_reports_no_crossing_as_infinite(call):
    packed, rng = batch("rlc", designs=1)
    c = rng.normal(size=(packed.n, 1))
    packed.coordinates(call, 0)
    packed.functional(call, c, 0)
    prop, span = packed.props[0], packed.props[0].step
    x, u = packed.x[:, 0], packed.u[:, 0]
    offset = 1.0 - min(float(c[:, 0] @ x), float(c[:, 0] @ prop.advance(x, u, span)))
    ends = (x, prop.advance(x, u, span))
    assert math.isinf(
        packed.bisect(call, offset, span, 1.0, float(c[:, 0] @ x) + offset, 0)
    )
    assert math.isinf(_crossing(prop, ends, u, (c[:, 0], offset), span, 1.0))


def test_bisect_finds_the_zero_a_pinned_functional_comes_back_to(call):
    """A current pinned at zero by a charged capacitor is due where it returns."""
    a, b, _ = series_rlc()
    x, u = np.array([0.0, 1.0]), np.zeros(1)
    packed, prop = single(a, b, 1.0 / 50e3, x, u)
    span, c = 0.75 * prop.step, np.array([1.0, 0.0])
    sign = math.copysign(1.0, derivative(a, b, x, u)[0])
    packed.coordinates(call, 0)
    packed.functional(call, c[:, None], 0)
    got = packed.bisect(call, 0.0, span, sign, 0.0, 0)
    expected = _crossing(prop, (x, prop.advance(x, u, span)), u, (c, 0.0), span, sign)
    assert got > 0.4 * span
    assert got == pytest.approx(expected, abs=4 * span * XTOL)


def test_bisect_reports_a_pinned_functional_that_never_enters_at_once(call):
    """One leaving its zero the other way never enters, so it is due at once."""
    a, b, step = CASES["decades"]
    x, u = np.zeros(len(a)), np.ones(1)
    packed, prop = single(a, b, step, x, u)
    c = np.eye(len(a))[0]
    sign = -math.copysign(1.0, derivative(a, b, x, u)[0])
    packed.coordinates(call, 0)
    packed.functional(call, c[:, None], 0)
    ends = (x, prop.advance(x, u, step))
    assert packed.bisect(call, 0.0, step, sign, 0.0, 0) == 0.0
    assert _crossing(prop, ends, u, (c, 0.0), step, sign) == 0.0
