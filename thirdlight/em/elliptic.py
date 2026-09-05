"""Complete elliptic integrals K(m), E(m) by the arithmetic-geometric mean.

Parameter convention matches ``scipy.special``: the argument is the parameter
m = k^2, not the modulus k.
"""

import math

from thirdlight.backend import kernel

_MAX_ITER = 16
_TOL = 1e-17


@kernel
def ellipke(m):
    """Return (K(m), E(m)) for 0 <= m < 1 via the AGM recurrence."""
    a = 1.0
    b = math.sqrt(1.0 - m)
    c = math.sqrt(m)
    s = 0.5 * c * c
    p = 1.0
    for _ in range(_MAX_ITER):
        if abs(c) < _TOL:
            break
        a, b, c = 0.5 * (a + b), math.sqrt(a * b), 0.5 * (a - b)
        p *= 2.0
        s += 0.5 * p * c * c
    k = 0.5 * math.pi / a
    return k, k * (1.0 - s)


@kernel
def ellipk(m):
    """Complete elliptic integral of the first kind."""
    return ellipke(m)[0]


@kernel
def ellipe(m):
    """Complete elliptic integral of the second kind."""
    return ellipke(m)[1]
