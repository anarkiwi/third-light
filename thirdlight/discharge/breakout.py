"""Breakout onset: electrode surface field per unit modal state, and Peek's threshold.

The surface field is linear in the state the time-domain solver already carries.
Ring potentials are the modal shapes scaled by the modal top-node voltages, ring
charges are the potential-coefficient solve applied to those, and Gauss at a
conductor turns charge into field, so the whole chain collapses to one
(rings x modes) matrix built once per design. Nothing is reassembled per step.
"""

from dataclasses import dataclass

import numpy as np
from scipy.constants import epsilon_0

from thirdlight.em.capacitance import field_correction

STANDARD_PRESSURE = 101325.0
STANDARD_TEMPERATURE = 298.15
DISRUPTIVE_FIELD = 3.0e6
PEEK_COEFFICIENT = 0.0301


def relative_density(pressure=STANDARD_PRESSURE, temperature=STANDARD_TEMPERATURE):
    """Air density relative to 25 C at one atmosphere, Peek's delta."""
    return (pressure / STANDARD_PRESSURE) * (STANDARD_TEMPERATURE / temperature)


def critical_field(radius, density=1.0, surface=1.0):
    """Peek's disruptive field at a conductor of curvature ``radius``, in V/m.

    E_c = m delta E_0 (1 + 0.0301 / sqrt(delta r)), the 0.301 cm^-1/2 of Peek's
    law in SI. ``surface`` is his irregularity factor m, 1 for a polished
    conductor. The curvature term is the ionisation layer's finite depth, so the
    field falls to the uniform-field E_0 delta as the electrode grows.
    """
    radius = np.asarray(radius, dtype=float)
    return (
        surface
        * DISRUPTIVE_FIELD
        * density
        * (1.0 + PEEK_COEFFICIENT / np.sqrt(density * radius))
    )


@dataclass(frozen=True)
class Breakout:
    """Surface field per unit modal state, against each ring's Peek threshold."""

    field: np.ndarray
    critical: np.ndarray

    def stress(self, v):
        """Peak surface field over the electrode at modal state ``v``, in V/m."""
        return np.max(np.abs(np.asarray(v) @ self.field.T), axis=-1)

    def margin(self, v):
        """Peak field as a fraction of the local threshold; 1 or more breaks out."""
        return np.max(np.abs(np.asarray(v) @ self.field.T) / self.critical, axis=-1)

    @property
    def voltage(self):
        """Top voltage carried by the first mode alone at which the surface breaks out."""
        return 1.0 / np.max(np.abs(self.field[:, 0]) / self.critical)


def shapes(modes):
    """Node potentials per unit top-node voltage of each mode, as (nodes, modes)."""
    return modes.v.T / modes.v[:, -1]


def correction(design):
    """Per-ring surface field correction of the top-node electrode discretisation."""
    counts = (design.top_load_sections, design.breakout_sections)
    parts = [
        field_correction(part, count) for part, count in zip(design.electrodes, counts)
    ]
    return np.concatenate(parts) if parts else np.ones(1)


def from_modes(design, rungs, modes, density=1.0, surface=1.0):
    """Breakout functional of a design's top-node electrode.

    ``rungs`` supplies the ring charge per unit node potential and ``modes`` the
    node potentials per unit modal state, so their product is the electrode
    charge per unit state; Gauss at a conductor, sigma / eps0 over the ring's own
    band area, makes it a field, corrected for the discretisation where the
    component needs it.
    """
    rows = rungs.electrode
    rings = rungs.rings[rows]
    scale = correction(design) if rungs.top else np.ones(1)
    field = (
        rungs.charge[rows] @ shapes(modes) * (scale / (epsilon_0 * rings.area))[:, None]
    )
    radius = design.top_load_curvature() if rungs.top else rings.rw
    return Breakout(field=field, critical=critical_field(radius, density, surface))
