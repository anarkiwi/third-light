"""Coupled L/C ladder assembly, eigen-solve and modal reduction.

Section k of the grounded secondary carries current i_k and node k charge q_k,
so dq/dt = A i and L di/dt = -A.T v with v = C^-1 q, giving the generalised
symmetric eigenproblem S v = omega^2 C v, S = A L^-1 A.T.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from thirdlight.backend import array_namespace, asnumpy
from thirdlight.em import capacitance, inductance
from thirdlight.geometry import Rings


@dataclass(frozen=True)
class Ladder:
    """Reduced ladder: section inductance, node capacitance and node heights."""

    L: np.ndarray
    C: np.ndarray
    z: np.ndarray

    def __len__(self):
        return len(self.z)


@dataclass(frozen=True)
class Modes:
    """Lowest modes of a ladder, one row per mode.

    ``v`` and ``i`` are normalised to unit modal energy, v C v = i L i = 1, with
    the top node positive. ``l_m`` and ``c_m`` are referred to the top-node
    potential: c_m = 1 / v_top^2, l_m = 1 / (omega^2 c_m).
    """

    f: np.ndarray
    v: np.ndarray
    i: np.ndarray
    l_m: np.ndarray
    c_m: np.ndarray
    z: np.ndarray

    def __len__(self):
        return len(self.f)


def incidence_matrix(sections):
    """Node-charge/section-current incidence A with (A i)_k = i_k - i_{k+1}."""
    return np.eye(sections) - np.eye(sections, k=1)


def node_map(groups, top_rings):
    """0/1 map T from node to ring potentials, v_rings = T v_nodes.

    Secondary ring j belongs to node ``groups[j]``; every top-load ring belongs
    to the top node, which is also the last secondary section.
    """
    sections = int(groups[-1]) + 1
    rows = np.concatenate([groups, np.full(top_rings, sections - 1, dtype=int)])
    node = np.zeros((rows.size, sections))
    node[np.arange(rows.size), rows] = 1.0
    return node


def ladder(design, sections=None):
    """Reduced L/C ladder of a design's secondary plus top load, at turn resolution.

    Series turns share a current, so L is a block sum; rings tied to one node
    share a potential, so C is merged as T.T C T. The merge is exact for the top
    load, one conductor already, and converges for the secondary as N rises.
    """
    sections = design.sections if sections is None else sections
    turns = design.secondary.discretise()
    top = design.top_load_rings()
    groups = inductance.turn_groups(len(turns), sections)
    xp = array_namespace()
    node = xp.asarray(node_map(groups, len(top)))
    rings = capacitance.capacitance_matrix(
        Rings.concat(turns, top), design.ground_plane, design.dielectric()
    )
    return Ladder(
        L=inductance.reduce_sections(inductance.inductance_matrix(turns), groups),
        C=asnumpy(node.T @ rings @ node),
        z=np.bincount(groups, weights=turns.z) / np.bincount(groups),
    )


def stiffness_matrix(inductances):
    """S = A L^-1 A.T, as the Gram matrix of G^-1 A.T with L = G G.T.

    Written this way S is symmetric and positive definite to rounding, whereas
    the literal triple product of unsymmetric factors is neither.
    """
    xp = array_namespace()
    chol = xp.linalg.cholesky(xp.asarray(inductances))
    right = xp.linalg.solve(chol, xp.asarray(incidence_matrix(len(inductances))).T)
    return right.T @ right


def eigenmodes(rungs, modes=4):
    """Lowest ``modes`` eigenmodes of a :class:`Ladder`."""
    xp = array_namespace()
    incidence = incidence_matrix(len(rungs.C))
    # scipy.linalg.eigh has no CuPy equivalent; the generalised solve runs on host arrays.
    lam, vec = eigh(
        asnumpy(stiffness_matrix(rungs.L)),
        asnumpy(rungs.C),
        subset_by_index=(0, modes - 1),
    )
    vec = vec * np.where(vec[-1] < 0.0, -1.0, 1.0)
    omega = np.sqrt(lam)
    drive = xp.asarray(incidence.T @ vec)
    current = asnumpy(xp.linalg.solve(xp.asarray(rungs.L), drive)) / omega
    top = vec[-1] ** 2
    return Modes(
        f=omega / (2.0 * math.pi),
        v=vec.T,
        i=current.T,
        l_m=top / lam,
        c_m=1.0 / top,
        z=rungs.z,
    )


def resonance(design, sections=None, modes=4):
    """Lowest ``modes`` resonances of a design's secondary and top load."""
    return eigenmodes(ladder(design, sections), modes)


def coupling(design, modes):
    """Coupling coefficient between the primary and each mode.

    The mutual to a mode carrying unit current in its top-referred equivalent is
    M_m = sqrt(l_m) phi.i_m, phi being the per-section flux per unit primary
    current; sqrt(l_m) cancels in k_m = M_m / sqrt(L_p l_m).
    """
    turns = design.secondary.discretise()
    primary = design.primary_rings()
    groups = inductance.turn_groups(len(turns), modes.v.shape[1])
    flux = np.bincount(
        groups, weights=inductance.mutual_matrix(primary, turns).sum(axis=0)
    )
    return modes.i @ flux / math.sqrt(inductance.inductance_matrix(primary).sum())
