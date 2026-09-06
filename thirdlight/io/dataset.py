"""Labelled output of a run: xarray datasets and a flat frame beside them.

Every series is taken from the :class:`~thirdlight.solver.stepping.Result` that
produced it. The modal rows are the network accessors' columns past the primary
loop, whose current and tank voltage are already carried in their own right.
"""

import numpy as np
import pandas as pd
import xarray as xr

_SIGNALS = (
    ("primary_current", "A"),
    ("tank_voltage", "V"),
    ("top_voltage", "V"),
    ("bus_voltage", "V"),
    ("drive", "V"),
    ("energy", "J"),
    ("length", "m"),
    ("channel", "F"),
    ("streamer_current", "A"),
    ("streamer_power", "W"),
    ("gate", "dimensionless"),
    ("state", "dimensionless"),
)

_MODAL = ("mode_current", "mode_voltage")


def _variables(table):
    """Dataset variables from rows of (name, dims, values, units)."""
    return {
        name: (dims, values, {"units": units}) for name, dims, values, units in table
    }


def to_dataset(result):
    """Run history as an :class:`xarray.Dataset` over time and mode."""
    net = result.network
    table = [
        (name, ("t",), np.asarray(getattr(result, name)), units)
        for name, units in _SIGNALS
    ]
    table += [
        (name, ("t", "mode"), accessor(result.x)[..., 1:], units)
        for name, accessor, units in (
            (_MODAL[0], net.currents, "A"),
            (_MODAL[1], net.voltages, "V"),
        )
    ]
    return xr.Dataset(
        _variables(table),
        coords={"t": ("t", result.t, {"units": "s"}), "mode": np.arange(net.modes)},
        attrs={
            "frequency": float(net.frequencies[0]),
            "duration": float(result.t[-1]),
            "samples": len(result),
            "dissipation": result.dissipation,
            "input_energy": result.input_energy,
        },
    )


def modes_dataset(modes):
    """Modal solution of a :class:`~thirdlight.secondary.Modes` over mode and height."""
    table = (
        ("f", ("mode",), modes.f, "Hz"),
        ("l_m", ("mode",), modes.l_m, "H"),
        ("c_m", ("mode",), modes.c_m, "F"),
        ("v", ("mode", "z"), modes.v, "V"),
        ("i", ("mode", "z"), modes.i, "A"),
    )
    return xr.Dataset(
        _variables(table),
        coords={"mode": np.arange(len(modes)), "z": ("z", modes.z, {"units": "m"})},
    )


def to_frame(result):
    """Run history as a :class:`pandas.DataFrame` indexed by time.

    The modal variables unstack to one column per mode, ``mode_current_0`` on.
    """
    data = to_dataset(result)
    modal = [data[name].to_pandas().add_prefix(f"{name}_") for name in _MODAL]
    return pd.concat([data.drop_dims("mode").to_dataframe(), *modal], axis=1)


def to_parquet(result, path):
    """Write :func:`to_frame` of ``result`` to a Parquet file."""
    to_frame(result).to_parquet(path)
