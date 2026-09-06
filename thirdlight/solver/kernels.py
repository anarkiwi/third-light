"""Scalar kernels of the batched eigen-coordinate stepper, one source for both targets.

Compiled by ``numba.njit`` for the CPU and ``numba.cuda.jit`` for the GPU: nothing
is allocated inside a kernel, complex data is carried as separate real and imaginary
float arrays, and the design indexes the last axis of every packed array.
"""

import math

from thirdlight.backend import kernel
from thirdlight.solver.propagator import _SERIES, _SERIES_LIMIT

SERIES = _SERIES
SERIES_LIMIT = _SERIES_LIMIT
_HALVINGS = 60


@kernel
def cexp(re, im):
    """e^(re + i im), as a (real, imaginary) pair."""
    scale = math.exp(re)
    return scale * math.cos(im), scale * math.sin(im)


@kernel
def _coefficient(exp_re, exp_im, z_re, z_im, s, series):
    """s(e^z - 1)/z from an already formed e^z, by series below the crossover."""
    if math.hypot(z_re, z_im) < SERIES_LIMIT:
        g_re = series[series.shape[0] - 1]
        g_im = 0.0
        for k in range(series.shape[0] - 2, -1, -1):
            g_re, g_im = (
                series[k] + g_re * z_re - g_im * z_im,
                g_re * z_im + g_im * z_re,
            )
        return s * g_re, s * g_im
    num_re = s * (exp_re - 1.0)
    num_im = s * exp_im
    den = z_re * z_re + z_im * z_im
    return (num_re * z_re + num_im * z_im) / den, (num_im * z_re - num_re * z_im) / den


@kernel
def phi_one(z_re, z_im, s, series):
    """The held input's coefficient s(e^z - 1)/z, as a (real, imaginary) pair.

    ``series`` carries the Horner coefficients used below ``SERIES_LIMIT``, where
    forming e^z - 1 loses more to cancellation than the truncation costs.
    """
    exp_re, exp_im = cexp(z_re, z_im)
    return _coefficient(exp_re, exp_im, z_re, z_im, s, series)


@kernel
def modal(inv_re, inv_im, x, y_re, y_im, n, d):
    """y = V^-1 x of design ``d``, for real ``x``, into ``y_re``/``y_im``."""
    for i in range(n):
        acc_re = 0.0
        acc_im = 0.0
        for j in range(n):
            acc_re += inv_re[i, j, d] * x[j, d]
            acc_im += inv_im[i, j, d] * x[j, d]
        y_re[i, d] = acc_re
        y_im[i, d] = acc_im


@kernel
def inject(invb_re, invb_im, u, ub_re, ub_im, n, p, d):
    """ub = V^-1 B u of design ``d``, for ``p`` real inputs held over the interval."""
    for i in range(n):
        acc_re = 0.0
        acc_im = 0.0
        for j in range(p):
            acc_re += invb_re[i, j, d] * u[j, d]
            acc_im += invb_im[i, j, d] * u[j, d]
        ub_re[i, d] = acc_re
        ub_im[i, d] = acc_im


@kernel
def row(basis_re, basis_im, c, w_re, w_im, n, d):
    """w = c^T V of design ``d``, for a real functional row ``c``."""
    for j in range(n):
        acc_re = 0.0
        acc_im = 0.0
        for i in range(n):
            acc_re += c[i, d] * basis_re[i, j, d]
            acc_im += c[i, d] * basis_im[i, j, d]
        w_re[j, d] = acc_re
        w_im[j, d] = acc_im


@kernel
def advance(lam_re, lam_im, y_re, y_im, ub_re, ub_im, s, series, out_re, out_im, n, d):
    """y(s) = e^(lam s) y + phi1(lam, s) ub of design ``d``, into ``out_re``/``out_im``."""
    for i in range(n):
        z_re = lam_re[i, d] * s
        z_im = lam_im[i, d] * s
        exp_re, exp_im = cexp(z_re, z_im)
        g_re, g_im = _coefficient(exp_re, exp_im, z_re, z_im, s, series)
        out_re[i, d] = (
            exp_re * y_re[i, d]
            - exp_im * y_im[i, d]
            + g_re * ub_re[i, d]
            - g_im * ub_im[i, d]
        )
        out_im[i, d] = (
            exp_re * y_im[i, d]
            + exp_im * y_re[i, d]
            + g_re * ub_im[i, d]
            + g_im * ub_re[i, d]
        )


@kernel
def restore(basis_re, basis_im, y_re, y_im, x, n, d):
    """x = Re(V y) of design ``d``, into real ``x``."""
    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc += basis_re[i, j, d] * y_re[j, d] - basis_im[i, j, d] * y_im[j, d]
        x[i, d] = acc


@kernel
def value(lam_re, lam_im, w_re, w_im, y_re, y_im, ub_re, ub_im, s, series, n, d):
    """Re(w . (e^(lam s) y + phi1 ub)) of design ``d``: the functional at sub-step ``s``.

    One diagonal scaling and one dot, against the two matrix-vector products an
    advance costs, since :func:`modal` and :func:`row` are paid once per interval.
    """
    total = 0.0
    for i in range(n):
        z_re = lam_re[i, d] * s
        z_im = lam_im[i, d] * s
        exp_re, exp_im = cexp(z_re, z_im)
        g_re, g_im = _coefficient(exp_re, exp_im, z_re, z_im, s, series)
        state_re = (
            exp_re * y_re[i, d]
            - exp_im * y_im[i, d]
            + g_re * ub_re[i, d]
            - g_im * ub_im[i, d]
        )
        state_im = (
            exp_re * y_im[i, d]
            + exp_im * y_re[i, d]
            + g_re * ub_im[i, d]
            + g_im * ub_re[i, d]
        )
        total += w_re[i, d] * state_re - w_im[i, d] * state_im
    return total


@kernel
def bisect(
    lam_re,
    lam_im,
    w_re,
    w_im,
    y_re,
    y_im,
    ub_re,
    ub_im,
    offset,
    span,
    sign,
    start,
    tol,
    series,
    n,
    d,
):
    """First s in [0, span) at which ``value`` + ``offset`` leaves ``sign``'s half plane.

    ``start`` is the caller's own value at s = 0. A functional already on the far
    side is due at once; one on its own zero is not, so the span is first halved
    to where the value still holds ``sign``, and zero returned if it never enters.
    """
    if start * sign < 0.0:
        return 0.0
    high = span
    outer = value(
        lam_re, lam_im, w_re, w_im, y_re, y_im, ub_re, ub_im, high, series, n, d
    )
    if (outer + offset) * sign >= 0.0:
        return math.inf
    low = 0.0
    if start == 0.0:
        s = span
        for _ in range(_HALVINGS):
            inner = value(
                lam_re, lam_im, w_re, w_im, y_re, y_im, ub_re, ub_im, s, series, n, d
            )
            if (inner + offset) * sign > 0.0:
                low = s
                break
            s *= 0.5
        if low == 0.0:
            return 0.0
    while high - low > tol:
        mid = 0.5 * (low + high)
        trial = value(
            lam_re, lam_im, w_re, w_im, y_re, y_im, ub_re, ub_im, mid, series, n, d
        )
        if (trial + offset) * sign > 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)
