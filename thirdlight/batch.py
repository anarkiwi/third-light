"""Design-space expansion, the sweep runner and the optimiser glue.

A point of a design space is a spec mapping, not a machine: an axis moves the
geometry the modes come out of, so every point rebuilds, where
:func:`thirdlight.discharge.calibration.sweep` varies the drive over one network.
"""

import copy
import itertools
import math
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context

import numpy as np
import pandas as pd

from thirdlight.discharge import calibration
from thirdlight.machine import Machine

# A design can be rejected two ways: the geometry checks raise, and rings that
# overlap leave the potential-coefficient matrix indefinite, which Cholesky does.
INFEASIBLE = (ValueError, np.linalg.LinAlgError)


def _resolve(spec, path):
    """Mapping holding the last key of a dotted ``path``, and that key.

    Raises :class:`KeyError` naming the first key that is not there, so a
    mistyped axis fails where it is given, not by adding a key nothing reads.
    """
    keys = path.split(".")
    node = spec
    for depth, key in enumerate(keys):
        if not isinstance(node, Mapping) or key not in node:
            raise KeyError(
                f"axis {path!r} is not in the spec: no {'.'.join(keys[: depth + 1])!r}"
            )
        if depth < len(keys) - 1:
            node = node[key]
    return node, keys[-1]


def _set(spec, path, value):
    """Set a dotted ``path`` of a nested mapping in place."""
    node, key = _resolve(spec, path)
    node[key] = value


def expand(spec, axes, product=True):
    """Points of a design space and the spec each one describes.

    ``axes`` maps dotted paths into ``spec`` to the values they take, crossed by
    :func:`itertools.product` or, where ``product`` is false, zipped and so of
    equal length. Each ``variant`` is a deep copy; the caller's spec is untouched.
    """
    names = list(axes)
    columns = [list(axes[name]) for name in names]
    for name in names:
        _resolve(spec, name)
    if product:
        points = itertools.product(*columns)
    else:
        sizes = {len(column) for column in columns}
        if len(sizes) > 1:
            raise ValueError(f"zipped axes must be of one length, not {sorted(sizes)}")
        points = zip(*columns)
    for values in points:
        variant = copy.deepcopy(spec)
        for name, value in zip(names, values):
            _set(variant, name, value)
        yield dict(zip(names, values)), variant


def observables(machine):
    """Frequency, coupling, Q, primary inductance, tank capacitance and breakout voltage.

    Read back off the built network rather than recomputed: the modal loop
    resistance is 2 pi f l_m / Q plus the former's dielectric term, so the
    unloaded Q of the driven mode inverts out of it.
    """
    net = machine.network
    primary, modal = float(net.inductances[0, 0]), float(net.inductances[1, 1])
    resistance = float(net.resistances[0, 1] - net.dielectric[0])
    frequency = machine.frequency
    return {
        "frequency": frequency,
        "coupling": float(net.inductances[0, 1] / math.sqrt(primary * modal)),
        "quality": 2.0 * math.pi * frequency * modal / resistance,
        "primary_inductance": primary,
        "tank_capacitance": float(machine.tank.capacitance),
        "breakout_voltage": float(machine.breakout().voltage),
    }


def performance(machine, streamer=None, **kwargs):
    """Input power, spark length, peak junction and coil temperature, and convergence.

    The first two are the settled interrupter cycle of
    :func:`~thirdlight.discharge.calibration.operating_point`, the rest the
    settled thermal cycle. ``kwargs`` go to :meth:`Machine.temperatures`.
    """
    streamer = machine.streamer() if streamer is None else streamer
    power, length = calibration.operating_point(machine, streamer)
    settled = machine.temperatures(streamer, **kwargs)
    return {
        "power": float(power),
        "length": float(length),
        "junction": float(settled.peak["igbt"]),
        "coil": float(settled.peak["coil"]),
        "converged": settled.converged,
    }


def spark(machine, rule=None, seed=None, **kwargs):
    """Input power, settled spark length and the k = L / sqrt(P) they imply.

    ``rule`` selects the model: none is the scalar channel of
    :meth:`Machine.streamer`, a :class:`~thirdlight.discharge.Growth` the grown
    tree of :meth:`Machine.channel` at ``seed``. ``kwargs`` go to whichever.
    """
    if rule is None:
        model, rng = machine.streamer(**kwargs), None
    else:
        model, rng = machine.channel(rule, **kwargs), np.random.default_rng(seed)
    power, length = calibration.operating_point(machine, model, rng=rng)
    return {
        "power": float(power),
        "length": float(length),
        "coefficient": float(calibration.inches_per_root_watt(power, length)),
    }


def evaluate(variant, observe=observables):
    """One variant's row, empty where the build rejects the design.

    See :data:`INFEASIBLE` for what rejection means.

    Module level and taking the mapping rather than a machine, so a process pool
    can ship it: what a build costs is the network, which is not worth pickling.
    """
    try:
        return observe(Machine.from_dict(variant))
    except INFEASIBLE:
        return {}


def sweep(spec, axes, observe=observables, product=True, workers=None):
    """Reduce every point of a design space to a row of a ``DataFrame``.

    The index is a MultiIndex over the axes in order and the columns are what
    ``observe`` returns, so ``frame.to_xarray()`` is the labelled sweep cube.
    ``workers`` above one spreads the variants over a process pool, spawned
    rather than forked: the parent has already opened the BLAS and Numba pools.
    """
    points, variants = zip(*expand(spec, axes, product))
    if workers is not None and workers > 1:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            rows = list(pool.map(evaluate, variants, itertools.repeat(observe)))
    else:
        rows = [evaluate(variant, observe) for variant in variants]
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(
            [tuple(point.values()) for point in points], names=list(axes)
        ),
    )


@dataclass(frozen=True)
class Objective:
    """A spec, dotted paths and a figure, as a scalar function of a vector.

    ``scipy.optimize.differential_evolution(obj, obj.bounds)`` minimises it. Under
    Optuna it is ``obj([trial.suggest_float(n, *b) for n, b in zip(obj.names,
    obj.bounds)])``, which is why neither library is imported here.
    """

    spec: Mapping
    names: tuple
    bounds: tuple
    figure: Callable
    kwargs: Mapping = field(default_factory=dict)

    def __call__(self, x):
        """The figure at point ``x``, or infinity where the build rejects it.

        An optimiser reads infinity as a wall and walks away from the infeasible
        region rather than stopping at it.
        """
        variant = copy.deepcopy(self.spec)
        for name, value in zip(self.names, x):
            _set(variant, name, float(value))
        try:
            return float(self.figure(Machine.from_dict(variant), **self.kwargs))
        except INFEASIBLE:
            return np.inf


def objective(spec, bounds, figure, **kwargs):
    """Objective over the dotted paths of ``bounds``, each a ``(low, high)`` pair.

    ``figure(machine)`` is MINIMISED, with no sign applied anywhere here: negate
    inside ``figure`` to maximise. ``kwargs`` go to ``figure``.
    """
    spec = copy.deepcopy(spec)
    for name in bounds:
        _resolve(spec, name)
    return Objective(
        spec=spec,
        names=tuple(bounds),
        bounds=tuple((float(low), float(high)) for low, high in bounds.values()),
        figure=figure,
        kwargs=kwargs,
    )
