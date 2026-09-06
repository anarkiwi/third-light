"""Round-trip of the YAML design schema documented in ``docs/schema.md``.

A sweep varies the spec mapping :func:`load` returns rather than a built
:class:`~thirdlight.machine.Machine`, which is not the spec: loading has already
resolved ``tank.tune`` into a capacitance and ``driver.lead_angle`` into a phase
lead, and the machine carries a network, ladder and modes no mapping describes.
"""

from dataclasses import MISSING, fields, is_dataclass

import numpy as np
import yaml

from thirdlight.geometry import Sphere, Toroid

_KINDS = {"top_load": {Toroid: "toroid", Sphere: "sphere"}}


def _emit(name, value):
    """One field's value: dataclasses recursed, discriminated fields tagged.

    ``Design.top_load`` is either shape and needs its ``kind``; ``Design.breakout``
    is a sphere by construction and carries none.
    """
    if not is_dataclass(value):
        return value.item() if isinstance(value, np.generic) else value
    kinds = _KINDS.get(name)
    body = to_dict(value)
    return body if kinds is None else {"kind": kinds[type(value)], **body}


def _emitted(obj):
    """Name/value pairs of every field left away from its declared default."""
    for f in fields(obj):
        value = getattr(obj, f.name)
        if f.default is MISSING or value != f.default:
            yield f.name, _emit(f.name, value)


def to_dict(obj):
    """Mapping of a dataclass, the inverse of ``Design.from_dict``.

    Defaulted fields are omitted, so the emitted spec is minimal, and numpy
    scalars are coerced to Python ones, so ``yaml.safe_dump`` accepts it.
    """
    return dict(_emitted(obj))


def dump(obj, path):
    """Write :func:`to_dict` of ``obj`` to ``path`` as YAML in field order."""
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(to_dict(obj), handle, sort_keys=False)


def load(path):
    """Read a spec mapping from a YAML file."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
