"""Waveform, mode-shape, field, streamer, channel, loss and temperature plots."""

# The config directory has to be writable before matplotlib is imported, and the
# backend chosen before pyplot binds one, so the imports follow the setup.
# pylint: disable=wrong-import-position

import copy
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from thirdlight import viz
from thirdlight.discharge import Growth
from thirdlight.machine import Machine
from thirdlight.viz.plots import RING_POINTS

TJ = 110.0
SPATIAL = ("channel",)

with open("examples/drsstc.yaml", encoding="utf-8") as _handle:
    SPEC = yaml.safe_load(_handle)


@pytest.fixture(name="machine", scope="module")
def fixture_machine():
    """The example machine shrunk to two modes and a burst a test can run."""
    spec = copy.deepcopy(SPEC)
    spec["secondary"]["turns"] = 60
    spec["sections"] = 20
    spec["top_load_sections"] = 8
    spec["driver"]["interrupter"] = {"on_time": 4.0e-6, "frequency": 20000.0}
    return Machine.from_dict(spec)


@pytest.fixture(name="result", scope="module")
def fixture_result(machine):
    """One short run with a streamer, so the channel plots have data."""
    channel = machine.streamer(growth=2.0, cooling=2e-5)
    return machine.run(6e-6, streamer=channel, length0=0.05)


@pytest.fixture(name="grown", scope="module")
def fixture_grown(machine):
    """A channel grown off the electrode at a held top voltage, by the growth clock."""
    model = machine.channel(Growth(step=0.02, radius=1.0e-3), rng=0)
    voltage = 3.0e5
    span = 12.0 * model.growth.step / (model.velocity * voltage)
    state = model.advance(model.initial(0.0), voltage, 1.0, span)
    assert state.tree.segments > 1
    return state


@pytest.fixture(name="settled", scope="module")
def fixture_settled(machine):
    """The settled interrupter cycle, the slowest object the plots consume."""
    return machine.temperatures(passes=2)


@pytest.fixture(name="plots", scope="module")
def fixture_plots(machine, result, grown, settled):
    """Every public plot bound to its data, as ``name -> f(ax)``."""
    state = machine.network.voltages(result.x[-1])[1:]
    hot = machine.breakout()
    return {
        "waveforms": lambda ax=None: viz.waveforms(result, ax),
        "mode_shapes": lambda ax=None: viz.mode_shapes(machine.eigen, ax),
        "surface_field": lambda ax=None: viz.surface_field(hot, state, ax),
        "streamer": lambda ax=None: viz.streamer(result, ax),
        "channel": lambda ax=None: viz.channel(grown, ax),
        "losses": lambda ax=None: viz.losses(result.losses(tj=TJ), ax),
        "temperatures": lambda ax=None: viz.temperatures(settled, ax),
    }


@pytest.fixture(autouse=True)
def _closed():
    """Close whatever a test opened, so no figure outlives it."""
    yield
    plt.close("all")


def spatial(collection):
    """Three-dimensional segments of a 3-D line collection, which matplotlib keeps private."""
    return np.asarray(collection._segments3d)  # pylint: disable=protected-access


def twinned(ax):
    """Lines of an axes and of the twin it shares its x axis with, in draw order."""
    return [line for axes in ax.figure.axes for line in axes.lines]


@pytest.mark.parametrize("name", viz.__all__)
def test_a_plot_returns_the_axes_it_was_given_and_labels_every_one(name, plots):
    ax = plots[name]()
    assert isinstance(ax, matplotlib.axes.Axes)
    given = plt.figure().add_subplot(projection="3d" if name in SPATIAL else None)
    assert plots[name](given) is given
    for drawn in (ax, given):
        assert drawn.get_xlabel()
        assert all(axes.get_ylabel() for axes in drawn.figure.axes)


def test_the_waveforms_are_the_primary_current_and_the_top_voltage(result, plots):
    current, voltage = twinned(plots["waveforms"]())
    assert np.array_equal(current.get_ydata(), result.primary_current)
    assert np.array_equal(voltage.get_ydata(), result.top_voltage)
    for line in (current, voltage):
        assert np.array_equal(line.get_xdata(), result.t)


def test_the_streamer_plot_is_the_channel_length_and_what_it_dissipates(result, plots):
    length, power = twinned(plots["streamer"]())
    assert np.array_equal(length.get_ydata(), result.length)
    assert np.array_equal(power.get_ydata(), result.streamer_power)
    assert length.get_ydata().max() > 0.0


def test_the_channel_is_its_own_segments_coloured_by_its_own_charges(grown, plots):
    rings, tree = plots["channel"]().collections
    nodes, parent = grown.tree.nodes, grown.tree.parent
    electrode = grown.discharge.rings
    assert np.array_equal(
        spatial(tree), np.stack([nodes[parent[1:]], nodes[1:]], axis=1)
    )
    assert np.array_equal(tree.get_array(), grown.discharge.charges()[len(electrode) :])
    circles = spatial(rings)
    assert circles.shape == (len(electrode), RING_POINTS, 3)
    assert np.hypot(circles[..., 0], circles[..., 1]) == pytest.approx(
        np.repeat(electrode.a[:, None], RING_POINTS, axis=1)
    )
    assert np.array_equal(
        circles[..., 2], np.repeat(electrode.z[:, None], RING_POINTS, axis=1)
    )


def test_one_labelled_profile_is_drawn_per_mode(machine, plots):
    modes = machine.eigen
    lines = plots["mode_shapes"]().lines
    assert len(lines) == len(modes)
    for line, f, v in zip(lines, modes.f, modes.v):
        assert np.array_equal(line.get_xdata(), modes.z)
        assert np.array_equal(line.get_ydata(), v)
        assert float(line.get_label().removesuffix(" kHz")) == pytest.approx(
            f / 1e3, rel=1e-3
        )


def test_the_surface_field_is_drawn_against_its_peek_threshold(machine, result, plots):
    hot = machine.breakout()
    state = machine.network.voltages(result.x[-1])[1:]
    field, critical = plots["surface_field"]().lines
    assert np.array_equal(field.get_ydata(), np.abs(state @ hot.field.T))
    assert field.get_ydata().max() == pytest.approx(hot.stress(state), rel=1e-14)
    assert critical.get_ydata() == pytest.approx(hot.critical, rel=1e-14)
    assert np.array_equal(field.get_xdata(), np.arange(len(hot.critical)))


def test_the_loss_bars_are_the_ledger_with_switching_beside_it(result, plots):
    ledger = result.losses(tj=TJ)
    ax = plots["losses"]()
    ax.figure.canvas.draw()
    widths = [patch.get_width() for patch in ax.patches]
    energy = dict(zip([text.get_text() for text in ax.get_yticklabels()], widths))
    assert energy["IGBT"] == ledger.igbt
    assert energy["diode"] == ledger.diode
    assert energy["primary"] == ledger.primary
    assert energy["ESR"] == ledger.esr
    assert energy["winding"] == pytest.approx(np.sum(ledger.winding), rel=1e-14)
    assert energy["former"] == ledger.former
    assert energy["channel"] == ledger.channel
    assert energy["switching"] == ledger.switching.total
    assert sum(widths) - energy["switching"] == pytest.approx(ledger.total, rel=1e-12)


def test_the_temperature_bars_are_the_settled_means_inside_their_ripple(settled, plots):
    ports = list(settled.mean)
    ax = plots["temperatures"]()
    ax.figure.canvas.draw()
    assert [text.get_text() for text in ax.get_xticklabels()] == ports
    assert [patch.get_height() for patch in ax.patches] == pytest.approx(
        [settled.mean[port] for port in ports], rel=1e-14
    )
    for port, segment in zip(ports, ax.collections[0].get_segments()):
        low, high = segment[0][1], segment[1][1]
        assert high - low == pytest.approx(settled.ripple[port], rel=1e-12)
        assert high == pytest.approx(settled.peak[port], rel=1e-12)
