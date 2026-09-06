"""Spark length against input power: the operating point map, its law and the fit."""

import math

import numpy as np
import pytest
import yaml

from thirdlight.discharge import calibration
from thirdlight.machine import Machine

with open("examples/drsstc.yaml", encoding="utf-8") as _handle:
    SPEC = yaml.safe_load(_handle)
BUSES = [1.0e4, 2.0e4, 4.0e4]


def small(**changes):
    """The example machine shrunk, driven hard enough for its needle to break out."""
    spec = {
        **SPEC,
        "secondary": {**SPEC["secondary"], "turns": 60},
        "sections": 20,
        "top_load_sections": 8,
        "breakout": {"radius": 2e-4, "height": 0.665},
        "breakout_sections": 6,
        "driver": {
            **SPEC["driver"],
            "bus": 2.0e4,
            "interrupter": {"on_time": 4e-6, "frequency": 20000.0},
        },
    }
    spec.update(changes)
    return Machine.from_dict(spec)


def streamer(machine, growth=2.0, cooling=2e-5):
    """Streamer with constants scaled to the shrunken machine's microsecond bursts."""
    return machine.streamer(growth=growth, cooling=cooling)


def test_freau_law_is_the_published_inches_per_root_watt():
    """L[in] = 1.7 sqrt(P[W]), in metres."""
    assert calibration.freau_length(1000.0) == pytest.approx(
        1.7 * math.sqrt(1000.0) * 0.0254
    )
    assert calibration.freau_length(0.0) == 0.0
    assert calibration.freau_length([100.0, 400.0]) == pytest.approx(
        [0.0254 * 17.0, 0.0254 * 34.0]
    )


def test_an_operating_point_averages_the_burst_over_the_whole_cycle():
    """The gap between bursts draws nothing, so cycle power is burst energy times PRF."""
    machine = small()
    hot = streamer(machine, cooling=2e-6)
    power, length = calibration.operating_point(machine, hot)
    result = calibration.burst(machine, hot)
    assert power == pytest.approx(
        result.input_energy * machine.driver.interrupter.frequency, rel=1e-9
    )
    assert length == pytest.approx(result.length.max(), rel=1e-9)
    assert power > 0.0 and length > 0.0


def test_a_hotter_channel_carries_over_between_bursts():
    """A cooling time comparable with the gap leaves the next burst a head start."""
    machine = small()
    brief = calibration.operating_point(machine, streamer(machine, cooling=2e-6))[1]
    persistent = calibration.operating_point(machine, streamer(machine, cooling=2e-4))[
        1
    ]
    assert persistent > brief


def test_the_model_predicts_a_square_root_law():
    """Length follows top voltage and top voltage follows the root of bang energy.

    Nothing in the length dynamics puts that exponent there; it comes out of the
    circuit, which is why the fitted constants set only where the line sits.
    """
    machine = small()
    power, length = calibration.sweep(machine, streamer(machine), BUSES)
    assert np.all(np.diff(power) > 0.0)
    assert np.all(np.diff(length) > 0.0)
    exponent = np.polyfit(np.log(power), np.log(length), 1)[0]
    assert 0.40 < exponent < 0.55


def test_the_fit_moves_the_sweep_towards_the_law_it_is_given():
    """Two constants against a coefficient, fitted in logarithms to keep both positive."""
    machine = small()
    target = 3.0e-3
    # A cooling time short against the gap leaves each burst its own fixed point.
    start = streamer(machine, growth=0.2, cooling=2e-6)
    bounds = (np.log([1e-3, 1e-7]), np.log([1e2, 1e-5]))
    before = calibration.residuals(machine, start, BUSES[:2], None, target)
    fitted, solution = calibration.fit(
        machine,
        start,
        BUSES[:2],
        coefficient=target,
        bounds=bounds,
        max_nfev=4,
        diff_step=0.3,
    )
    assert solution.cost < 0.5 * float(before @ before) / 2.0
    assert fitted.growth > 0.0 and fitted.cooling > 0.0
    assert fitted.gradient == start.gradient


@pytest.mark.slow
def test_the_calibrated_constants_land_inside_the_published_spark_length_band():
    """Published DRSSTC coils imply k = 1.2 to 2.1 in/sqrt(W); no source states a law.

    The within-coil exponent is the independent check: Steve Ward's DRSSTC-0.5
    table, the one published set that varies power at fixed geometry, gives 0.341
    over its five points, against the 0.27 this machine's sweep predicts.
    """
    machine = Machine.from_yaml("examples/drsstc.yaml")
    power, length = calibration.sweep(
        machine, machine.streamer(), [120.0, 200.0, 350.0]
    )
    coefficient = length / np.sqrt(power) / 0.0254
    assert 190.0 < power[0] and power[-1] < 1400.0
    assert np.all((coefficient > 1.2) & (coefficient < 2.1))
    assert 0.25 < np.polyfit(np.log(power), np.log(length), 1)[0] < 0.40
