"""Waveform, mode-shape, field, streamer and channel plots of what a run carries.

Each function draws one view of one data object onto a supplied or fresh Axes
and returns it, so the figure, the backend and whether anything is shown or
saved stay the caller's. See §4 of design.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

RING_POINTS = 64


def _axes(ax, xlabel, ylabel):
    """The Axes given, or a fresh one, labelled with quantity and unit."""
    if ax is None:
        _, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def _axes3d(ax, xlabel, ylabel, zlabel):
    """The Axes given, or a fresh three-dimensional one, labelled on all three."""
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    ax.set_zlabel(zlabel)
    return _axes(ax, xlabel, ylabel)


def _twinned(ax, x, xlabel, left, right):
    """Two ``(label, values)`` series against ``x`` on twinned y axes.

    The twin continues the default property cycle rather than restarting it, and
    the two lines share one legend on the primary axis, which is what returns.
    """
    ax = _axes(ax, xlabel, left[0])
    other = ax.twinx()
    other.set_prop_cycle(plt.rcParams["axes.prop_cycle"][1:])
    other.set_ylabel(right[0])
    lines = ax.plot(x, left[1], label=left[0]) + other.plot(x, right[1], label=right[0])
    ax.legend(lines, [line.get_label() for line in lines])
    return ax


def waveforms(result, ax=None):
    """Primary current and top-load voltage of a run against time."""
    return _twinned(
        ax,
        result.t,
        "time (s)",
        ("primary current (A)", result.primary_current),
        ("top voltage (V)", result.top_voltage),
    )


def mode_shapes(modes, ax=None):
    """Voltage profile of each mode along the secondary, labelled by frequency."""
    ax = _axes(ax, "height (m)", "modal voltage (V)")
    for f, v in zip(modes.f, modes.v):
        ax.plot(modes.z, v, label=f"{f / 1e3:.1f} kHz")
    ax.legend()
    return ax


def surface_field(breakout, voltages, ax=None):
    """Electrode surface field at modal top-node ``voltages``, against Peek onset."""
    field = breakout.profile(voltages)
    index = np.arange(len(field))
    ax = _axes(ax, "electrode ring", "surface field (V/m)")
    ax.plot(index, field, label="surface field")
    ax.plot(index, np.broadcast_to(breakout.critical, field.shape), label="Peek onset")
    ax.legend()
    return ax


def streamer(result, ax=None):
    """Channel length and streamer dissipation of a run against time."""
    return _twinned(
        ax,
        result.t,
        "time (s)",
        ("channel length (m)", result.length),
        ("streamer power (W)", result.streamer_power),
    )


def _circles(rings):
    """Each ring of an electrode as a closed polyline about the axis, (rings, points, 3)."""
    angle = np.linspace(0.0, 2.0 * np.pi, RING_POINTS)
    unit = np.stack([np.cos(angle), np.sin(angle), np.zeros(RING_POINTS)], axis=-1)
    axial = np.stack([np.zeros(len(rings)), np.zeros(len(rings)), rings.z], axis=-1)
    return rings.a[:, None, None] * unit[None] + axial[:, None, :]


def channel(state, ax=None):
    """A grown channel over its electrode in three dimensions, coloured by segment charge.

    The tree is one line collection over the segment endpoints it already
    carries, coloured by the charges 3.4b's mixed solve puts on them, and the
    electrode the rings of the same solve.
    """
    discharge = state.discharge
    nodes, parent = state.tree.nodes, state.tree.parent
    circles = _circles(discharge.rings)
    ax = _axes3d(ax, "x (m)", "y (m)", "z (m)")
    ax.add_collection3d(Line3DCollection(circles, colors="0.7"))
    tree = Line3DCollection(np.stack([nodes[parent[1:]], nodes[1:]], axis=1))
    tree.set_array(discharge.charges()[len(discharge.rings) :])
    ax.add_collection3d(tree)
    ax.figure.colorbar(tree, ax=ax, label="segment charge (C)")
    ax.auto_scale_xyz(*np.concatenate([nodes, circles.reshape(-1, 3)]).T)
    return ax


def losses(ledger, ax=None):
    """Component energy ledger of a run as a horizontal bar chart.

    Switching is a bar of its own: it is additive to the conduction total the
    other bars sum to, not one of its parts.
    """
    entries = {
        "IGBT": ledger.igbt,
        "diode": ledger.diode,
        "primary": ledger.primary,
        "ESR": ledger.esr,
        "winding": float(np.sum(ledger.winding)),
        "former": ledger.former,
        "channel": ledger.channel,
        "switching": ledger.switching.total,
    }
    ax = _axes(ax, "energy (J)", "component")
    ax.barh(list(entries), list(entries.values()))
    return ax


def temperatures(steady, ax=None):
    """Settled cycle-mean port temperature, with the within-cycle swing as an error bar.

    The bar reaches the mean and the whiskers the extremes of the cycle, so their
    span is the ripple and the upper one is what the die actually sees.
    """
    ports = list(steady.mean)
    mean = np.array([steady.mean[port] for port in ports])
    ripple = np.array([steady.ripple[port] for port in ports])
    high = np.array([steady.peak[port] for port in ports]) - mean
    ax = _axes(ax, "port", "temperature (C)")
    ax.bar(ports, mean, yerr=[ripple - high, high])
    return ax
