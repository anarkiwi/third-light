"""Exact propagator of dx/dt = A x + B u with u zero-order held over a step.

x(t+s) = Phi(s) x(t) + Gamma(s) u, Phi(s) = e^{As}, Gamma(s) = int_0^s e^{At} dt B,
evaluable at arbitrary s so an event-driven integrator propagates to the event
instant itself rather than to a quantised sub-step.

A is diagonalised once per switch state where its eigenbasis is well conditioned,
which costs one complex diagonal scaling per evaluation; a defective or
near-defective A falls back to Pade propagators tabulated at step/2^j and
composed by the semigroup rule.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy.linalg import expm

_EPS = np.spacing(1.0)


def _series_limit(order, eps=_EPS):
    """|z| where the two evaluations of (e^z - 1)/z carry equal relative error.

    A complex expm1 formed as exp(z) - 1 keeps absolute error eps against a
    result of size |z|, so relative error eps/|z|; the series truncated after
    ``order`` terms keeps |z|^order/(order+1)!. They cross at ((m+1)! eps)^(1/(m+1)).
    """
    return (math.factorial(order + 1) * eps) ** (1.0 / (order + 1))


def _series_order(eps=_EPS):
    """Fewest terms whose crossover error eps/limit reaches the Horner sum's own floor.

    That floor is the order eps of accumulating the terms; it is met at order 9,
    limit 0.123 and crossover error 8 eps, and more terms buy nothing.
    """
    order = 1
    while _series_limit(order, eps) < 1.0 / order:
        order += 1
    return order


_SERIES_ORDER = _series_order()
_SERIES_LIMIT = _series_limit(_SERIES_ORDER)
_SERIES = np.array([1.0 / math.factorial(k + 1) for k in range(_SERIES_ORDER)])


def _phi_one(z, s):
    """s (e^z - 1)/z elementwise, i.e. (e^{lam s} - 1)/lam at z = lam s, and s at z = 0."""
    g = np.empty_like(z)
    big = np.abs(z) >= _SERIES_LIMIT
    small = ~big
    g[big] = s * np.expm1(z[big]) / z[big]
    g[small] = s * polyval(z[small], _SERIES)
    return g


def _real(m):
    """Real part of a product whose imaginary part cancels, as a contiguous array."""
    return np.ascontiguousarray(m.real)


def derivative(a, b, x, u):
    """A x + B u, with ``b`` 1-D or 2-D and ``u`` scalar or 1-D."""
    x = np.asarray(x)
    return np.asarray(a) @ x + np.reshape(b, (x.size, -1)) @ np.reshape(u, -1)


@dataclass(frozen=True)
class _Eigen:
    """A = basis diag(lam) inverse, with inverse_b = V^-1 B held for the input map."""

    lam: np.ndarray
    basis: np.ndarray
    inverse: np.ndarray
    inverse_b: np.ndarray


@dataclass(frozen=True)
class _Table:
    """Propagator pairs at step/2^j, j = 0..levels, stacked on the leading axis."""

    phi: np.ndarray
    gamma: np.ndarray

    @property
    def levels(self):
        """Deepest tabulated subdivision."""
        return len(self.phi) - 1


def _diagonalise(a, b, cond_max):
    """Eigenbasis of ``a``, or None if eig fails or the basis is too ill conditioned."""
    try:
        lam, basis = np.linalg.eig(a)
    except np.linalg.LinAlgError:
        return None
    if not np.linalg.cond(basis) < cond_max:
        return None
    inverse = np.linalg.inv(basis)
    return _Eigen(lam=lam, basis=basis, inverse=inverse, inverse_b=inverse @ b)


def _tabulate(a, b, step, levels):
    """(Phi, Gamma) at step/2^j from expm([[A, B], [0, 0]] s) = [[Phi, Gamma], [0, I]]."""
    n, p = b.shape
    block = np.zeros((n + p, n + p))
    block[:n, :n] = a
    block[:n, n:] = b
    aug = np.array([expm(block * (step * 2.0**-j)) for j in range(levels + 1)])
    return _Table(phi=aug[:, :n, :n].copy(), gamma=aug[:, :n, n:].copy())


@dataclass(frozen=True)
class Propagator:
    """Exact propagator of a linear time-invariant state space over a step."""

    step: float
    eigen: _Eigen | None = None
    table: _Table | None = None
    nominal: tuple | None = None

    @property
    def exact(self):
        """True when the diagonalised path was accepted."""
        return self.eigen is not None

    @classmethod
    def build(cls, a, b, step, cond_max=1e8, levels=32):
        """Propagator of dx/dt = A x + B u over nominal ``step``.

        ``b`` is a 1-D vector for a single input or (n, p) for p of them. The
        eigenbasis is used only when its 2-norm condition number is below
        ``cond_max``; otherwise ``levels`` + 1 dyadic Pade propagators are
        tabulated, which resolves a sub-step to step/2^(levels+1).
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float).reshape(len(a), -1)
        step = float(step)
        if step <= 0.0:
            raise ValueError(f"step must be positive; got {step}")
        eigen = _diagonalise(a, b, cond_max)
        table = None if eigen is not None else _tabulate(a, b, step, levels)
        built = cls(step=step, eigen=eigen, table=table)
        return replace(built, nominal=built.at(step))

    def _pieces(self, s):
        """Levels of the dyadic expansion of s/step, largest first.

        s/step is rounded to the nearest multiple of 2^-levels rather than
        truncated, so the composed pair is the exact propagator of an s within
        step/2^(levels+1), and an s that is dyadic to begin with lands on its own
        expansion however the quotient rounds.
        """
        levels = self.table.levels
        bits = round(s / self.step * (1 << levels))
        return [j for j in range(levels + 1) if bits >> (levels - j) & 1]

    def _checked(self, s):
        """``s`` if it lies within the step, else a ValueError."""
        if not 0.0 <= s <= self.step:
            raise ValueError(f"s must lie in [0, {self.step}]; got {s}")
        return s

    def at(self, s):
        """Phi(s), Gamma(s) for 0 <= s <= step, both real, and exact at either end."""
        if self._checked(s) == 0.0:
            phi, gamma = self.nominal
            return np.eye(phi.shape[0]), np.zeros_like(gamma)
        if s == self.step and self.nominal is not None:
            return self.nominal
        if self.eigen is not None:
            z = self.eigen.lam * s
            return (
                _real((self.eigen.basis * np.exp(z)) @ self.eigen.inverse),
                _real((self.eigen.basis * _phi_one(z, s)) @ self.eigen.inverse_b),
            )
        phi = np.eye(self.table.phi.shape[1])
        gamma = np.zeros(self.table.gamma.shape[1:])
        for j in self._pieces(s):
            phi = self.table.phi[j] @ phi
            gamma = self.table.phi[j] @ gamma + self.table.gamma[j]
        return phi, gamma

    def evaluator(self, x, u, functional):
        """``s -> c x(s) + d`` over one interval, for a crossing search to bisect.

        A search evaluates one functional at many sub-steps of an interval whose
        state and held input do not move. Where the eigenbasis was accepted, the
        state's coordinates in it and the functional's row against it are
        constant over that interval, so each evaluation is a diagonal scaling and
        a dot rather than the two matrix-vector products :meth:`advance` costs:
        O(n) against O(n^2), and the two O(n^2) products are paid once.

        The value at s = 0 is the one the caller started from, not a
        reconstruction of it, since that is the value a crossing is bracketed
        against.
        """
        c, d = functional
        c = np.asarray(c, dtype=float)
        start = float(c @ np.asarray(x, dtype=float) + d)
        if self.eigen is None:
            return lambda s: start if s == 0.0 else float(c @ self.advance(x, u, s) + d)
        y = self.eigen.inverse @ np.asarray(x, dtype=float)
        ub = self.eigen.inverse_b @ np.reshape(u, -1)
        w = c @ self.eigen.basis

        def value(s):
            if self._checked(s) == 0.0:
                return start
            z = self.eigen.lam * s
            return float((w @ (np.exp(z) * y + _phi_one(z, s) * ub)).real) + d

        return value

    def advance(self, x, u, s=None):
        """State after ``s`` (default the nominal step) with ``u`` held across it.

        Composed on the state vector rather than through :meth:`at`, so a sub-step
        costs matrix-vector work only. A zero step returns the state untouched, so
        an event locator brackets against the value it started from.
        """
        x = np.asarray(x, dtype=float)
        u = np.reshape(u, -1)
        if self._checked(s if s is not None else self.step) == 0.0:
            return x.copy()
        if s is None or s == self.step:
            phi, gamma = self.nominal
            return phi @ x + gamma @ u
        if self.eigen is not None:
            z = self.eigen.lam * self._checked(s)
            state = np.exp(z) * (self.eigen.inverse @ x)
            return (
                self.eigen.basis @ (state + _phi_one(z, s) * (self.eigen.inverse_b @ u))
            ).real
        for j in self._pieces(self._checked(s)):
            x = self.table.phi[j] @ x + self.table.gamma[j] @ u
        return x
