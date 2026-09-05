"""Primary current-transformer feedback and its phase lead.

The comparator input is an affine functional of the state, s = c x + d, so the
piecewise-LTI integrator locates a zero crossing exactly within a step instead
of bisecting on a resimulated signal.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhaseLead:
    """Primary-current transformer feedback with first-order phase lead.

    The comparator sees s(t) = i_p + tau di_p/dt, which leads i_p by
    arctan(omega tau); UD2-style drivers realise it with a series capacitor in
    the current-transformer burden.
    """

    tau: float

    @classmethod
    def from_angle(cls, degrees, frequency):
        """Lead of ``degrees`` at ``frequency``: tau = tan(radians)/(2 pi f)."""
        return cls(math.tan(math.radians(degrees)) / (2.0 * math.pi * frequency))

    def angle(self, frequency):
        """Lead in degrees at ``frequency``, arctan(2 pi f tau)."""
        return math.degrees(math.atan(2.0 * math.pi * frequency * self.tau))

    def functional(self, a, b, u, index=0):
        """(c, d) giving s = c x + d for dx/dt = A x + B u with i_p = x[index].

        Substituting the state equation into i_p + tau di_p/dt leaves
        c = e_index + tau A[index] and d = tau B[index] u, both constant while
        the bridge state and the held input are.
        """
        c = np.zeros(np.shape(a)[0])
        c[index] = 1.0
        c += self.tau * np.asarray(a)[index]
        row = np.atleast_1d(np.asarray(b)[index])
        return c, self.tau * float(np.dot(row, np.atleast_1d(u)))
