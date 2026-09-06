"""Design-schema round-trip and the labelled output of a run."""

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from thirdlight import io
from thirdlight.geometry import Design, Former, Primary, Solenoid, Sphere, Toroid
from thirdlight.machine import Machine

DRIVE = ("modes", "tank", "bus", "bridge", "driver", "thermal")
SIGNALS = (
    "primary_current",
    "tank_voltage",
    "top_voltage",
    "bus_voltage",
    "drive",
    "energy",
    "length",
    "channel",
    "streamer_current",
    "streamer_power",
    "gate",
    "state",
)

SECONDARY = Solenoid(
    radius=0.0762, length=0.508, turns=1200.0, wire_diameter=0.0004, base=0.05
)
PRIMARY = Primary(
    inner_radius=0.115, turns=5.5, pitch=0.012, base=0.055, wire_diameter=0.0064
)
TOROID = Toroid(major_radius=0.152, minor_radius=0.0508, height=0.6)
BALL = Sphere(radius=0.15, height=0.6)
POINT = Sphere(radius=0.006, height=0.665)
FORMER = Former(
    outer_radius=0.075, length=0.508, base=0.05, inner_radius=0.072, loss_tangent=0.001
)

with open("examples/drsstc.yaml", encoding="utf-8") as _handle:
    SPEC = yaml.safe_load(_handle)


def design(**parts):
    """A design of the shared secondary and primary, varied by keyword."""
    return Design(secondary=SECONDARY, primary=PRIMARY, **parts)


DESIGNS = {
    "toroid": design(top_load=TOROID, breakout=POINT, former=FORMER),
    "sphere": design(top_load=BALL, breakout=Sphere(radius=0.006, height=0.8)),
    "bare": design(),
    "no breakout": design(top_load=TOROID, former=FORMER),
    "no former": design(top_load=TOROID, breakout=POINT),
    "no ground plane": design(top_load=TOROID, ground_plane=False),
    "band primary": Design(
        secondary=SECONDARY,
        primary=Primary(
            inner_radius=0.2, turns=1.0, band_width=0.075, band_thickness=0.0015
        ),
        top_load=TOROID,
    ),
    "sections": design(
        top_load=BALL,
        breakout=Sphere(radius=0.006, height=0.8),
        former=FORMER,
        sections=64,
        top_load_sections=16,
        breakout_sections=12,
        former_sections=48,
    ),
}


def example(path):
    """The design part of an example file, the drive sections dropped."""
    with open(path, encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    return Design.from_dict({k: v for k, v in spec.items() if k not in DRIVE})


@pytest.fixture(name="machine", scope="module")
def fixture_machine():
    """The example machine shrunk to a size the interpreted coverage pass can run."""
    spec = copy.deepcopy(SPEC)
    spec["secondary"]["turns"] = 60
    spec["sections"] = 20
    spec["top_load_sections"] = 8
    return Machine.from_dict(spec)


@pytest.fixture(name="result", scope="module")
def fixture_result(machine):
    """A few driven cycles of that machine."""
    return machine.run(5.0 / machine.frequency)


@pytest.mark.parametrize("path", ["examples/sstc.yaml", "examples/drsstc.yaml"])
def test_the_examples_round_trip_through_a_mapping(path):
    expected = example(path)
    assert Design.from_dict(io.to_dict(expected)) == expected


@pytest.mark.parametrize("expected", DESIGNS.values(), ids=DESIGNS)
def test_every_shape_of_design_round_trips(expected):
    assert Design.from_dict(io.to_dict(expected)) == expected


def test_defaulted_fields_are_omitted():
    spec = io.to_dict(design(top_load=TOROID))
    assert set(spec) == {"secondary", "primary", "top_load"}
    assert "rise" not in spec["primary"]
    assert "permittivity" not in io.to_dict(design(former=FORMER))["former"]


def test_only_the_top_load_carries_a_kind():
    spec = io.to_dict(design(top_load=TOROID, breakout=POINT))
    assert spec["top_load"]["kind"] == "toroid"
    assert io.to_dict(design(top_load=BALL))["top_load"]["kind"] == "sphere"
    assert spec["breakout"] == {"radius": POINT.radius, "height": POINT.height}


def test_numpy_scalars_are_coerced_for_the_dumper():
    spec = io.to_dict(design(top_load=Sphere(np.float64(0.15), np.float64(0.6))))
    assert not isinstance(spec["top_load"]["radius"], np.generic)
    assert yaml.safe_dump(spec)


@pytest.mark.parametrize("expected", DESIGNS.values(), ids=DESIGNS)
def test_a_dumped_design_loads_back(expected, tmp_path):
    path = tmp_path / "design.yaml"
    io.dump(expected, path)
    assert Design.from_dict(io.load(path)) == expected


def test_the_dataset_carries_every_result_series(result):
    data = io.to_dataset(result)
    np.testing.assert_array_equal(data["t"].values, result.t)
    assert data["t"].attrs["units"] == "s"
    for name in SIGNALS:
        np.testing.assert_array_equal(data[name].values, getattr(result, name))
        assert data[name].dims == ("t",)
        assert data[name].attrs["units"]


def test_the_modal_variables_are_the_columns_past_the_primary(result):
    data = io.to_dataset(result)
    net = result.network
    shape = (len(result), net.modes)
    np.testing.assert_array_equal(data["mode"].values, np.arange(net.modes))
    for name, accessor in (
        ("mode_current", net.currents),
        ("mode_voltage", net.voltages),
    ):
        assert data[name].shape == shape
        assert data[name].dims == ("t", "mode")
        np.testing.assert_array_equal(data[name].values, accessor(result.x)[:, 1:])


def test_the_dataset_attributes_describe_the_run(result):
    attrs = io.to_dataset(result).attrs
    assert attrs["frequency"] == float(result.network.frequencies[0])
    assert attrs["duration"] == result.t[-1]
    assert attrs["samples"] == len(result)
    assert attrs["dissipation"] == result.dissipation
    assert attrs["input_energy"] == result.input_energy


def test_the_modal_dataset_matches_the_eigen_solution(machine):
    modes = machine.eigen
    data = io.modes_dataset(modes)
    np.testing.assert_array_equal(data["z"].values, modes.z)
    np.testing.assert_array_equal(data["mode"].values, np.arange(len(modes)))
    for name in ("f", "l_m", "c_m"):
        assert data[name].dims == ("mode",)
        np.testing.assert_array_equal(data[name].values, getattr(modes, name))
    for name in ("v", "i"):
        assert data[name].shape == (len(modes), len(modes.z))
        np.testing.assert_array_equal(data[name].values, getattr(modes, name))
    assert data["c_m"].attrs["units"] == "F"


def test_the_frame_flattens_the_modal_variables(result):
    frame = io.to_frame(result)
    modes = result.network.modes
    assert set(frame.columns) == set(SIGNALS) | {
        f"{name}_{m}" for name in ("mode_current", "mode_voltage") for m in range(modes)
    }
    assert frame.index.name == "t"
    np.testing.assert_array_equal(frame.index.to_numpy(), result.t)
    np.testing.assert_array_equal(frame["top_voltage"].to_numpy(), result.top_voltage)
    np.testing.assert_array_equal(
        frame["mode_current_1"].to_numpy(), result.network.currents(result.x)[:, 2]
    )


def test_parquet_round_trips_the_frame(result, tmp_path):
    path = tmp_path / "run.parquet"
    io.to_parquet(result, path)
    pd.testing.assert_frame_equal(pd.read_parquet(path), io.to_frame(result))
