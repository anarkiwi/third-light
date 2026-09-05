"""Medhurst's measured proximity factor and the eddy-current reaction it implies.

Table VIII of [9] part 2 p88 is phi = R_hf(coil) / R_hf(the same length of straight
wire at the same frequency), measured over d/s and l/D, so it carries both the
neighbouring turns' eddy reaction and the finite-length field factor.
"""

import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Ascending in d/s; the paper prints the rows the other way up.
SPACING = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
LENGTH = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, math.inf)
PHI = np.array(
    [
        [1.02, 1.02, 1.03, 1.03, 1.03, 1.03, 1.04, 1.04, 1.04, 1.04, 1.04, 1.05],
        [1.07, 1.08, 1.08, 1.10, 1.10, 1.10, 1.13, 1.15, 1.16, 1.16, 1.17, 1.19],
        [1.16, 1.19, 1.21, 1.22, 1.23, 1.24, 1.28, 1.32, 1.34, 1.34, 1.35, 1.40],
        [1.20, 1.29, 1.33, 1.38, 1.42, 1.45, 1.50, 1.54, 1.56, 1.57, 1.58, 1.65],
        [1.44, 1.48, 1.54, 1.60, 1.64, 1.67, 1.74, 1.78, 1.80, 1.81, 1.83, 1.93],
        [1.74, 1.77, 1.83, 1.89, 1.92, 1.94, 1.98, 2.01, 2.03, 2.08, 2.10, 2.22],
        [2.12, 2.20, 2.28, 2.38, 2.44, 2.47, 2.32, 2.27, 2.29, 2.34, 2.37, 2.51],
        [2.74, 2.83, 2.97, 3.10, 3.20, 3.17, 2.74, 2.60, 2.60, 2.62, 2.65, 2.81],
        [3.73, 3.84, 3.99, 4.11, 4.17, 4.10, 3.36, 3.05, 2.92, 2.90, 2.93, 3.11],
        [5.31, 5.45, 5.65, 5.80, 5.80, 5.55, 4.10, 3.54, 3.31, 3.20, 3.23, 3.41],
    ]
)


def _compact(length_ratio):
    """l/D mapped onto [0, 1] by x/(1 + x), which carries the table's last column to 1."""
    x = np.asarray(length_ratio, dtype=float)
    return np.divide(x, 1.0 + x, out=np.ones_like(x), where=np.isfinite(x))


def _excess(spacing):
    """Uniform-external-field excess over the straight wire, pi^2 (d/s)^2 / 2.

    The high-frequency ratio of the proximity term u G (d/2p)^2 to the skin term of
    :func:`thirdlight.em.losses.ac_ratio`, at Butterworth's u = pi^2.
    """
    return 0.5 * math.pi * math.pi * np.square(spacing)


_REACTION = RegularGridInterpolator(
    (SPACING, _compact(LENGTH)),
    (PHI - 1.0) / _excess(SPACING)[:, None],
    method="pchip",
)


def eddy_reaction(spacing, length_ratio):
    """Measured proximity excess over the uniform-field model's, at d/s and l/D.

    Below the table's d/s = 0.1 the two agree to 1 %, so clamping there is exact to
    the printed precision.
    """
    d, ratio = np.broadcast_arrays(
        np.clip(spacing, SPACING[0], SPACING[-1]),
        np.clip(_compact(length_ratio), 0.0, 1.0),
    )
    value = _REACTION(np.stack((d.ravel(), ratio.ravel()), axis=-1)).reshape(d.shape)
    return value if value.ndim else value[()]


def proximity_factor(spacing, length_ratio):
    """Medhurst's phi interpolated at d/s ``spacing`` and length-to-diameter ratio."""
    return 1.0 + eddy_reaction(spacing, length_ratio) * _excess(spacing)
