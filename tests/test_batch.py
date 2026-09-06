"""Design-space expansion, the sweep runner and the optimiser glue."""

import copy
import itertools
import math

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.optimize import minimize

from thirdlight import batch, secondary
from thirdlight.em import losses
from thirdlight.machine import Machine

with open("examples/drsstc.yaml", encoding="utf-8") as _handle:
    SPEC = yaml.safe_load(_handle)
BASE = {
    **copy.deepcopy(SPEC),
    "secondary": {**SPEC["secondary"], "turns": 60},
    "sections": 20,
    "top_load_sections": 8,
    "breakout": {"radius": 2.0e-4, "height": 0.665},
    "breakout_sections": 6,
    "modes": 2,
    "driver": {
        **SPEC["driver"],
        "bus": 2.0e4,
        "interrupter": {"on_time": 4.0e-6, "frequency": 20000.0},
    },
}
RADII = [0.15, 0.20]


def test_crossing_two_axes_gives_every_combination_in_axis_order():
    axes = {"top_load.major_radius": RADII, "tank.tune": [0.9, 1.0, 1.1]}
    points = [point for point, _ in batch.expand(BASE, axes)]
    assert len(points) == 6
    assert list(points[0]) == list(axes)
    assert {tuple(point.values()) for point in points} == set(
        itertools.product(*axes.values())
    )


def test_zipped_axes_pair_up_and_must_be_of_one_length():
    axes = {"top_load.major_radius": RADII, "tank.tune": [0.9, 1.0]}
    pairs = [point for point, _ in batch.expand(BASE, axes, product=False)]
    assert [tuple(point.values()) for point in pairs] == [(0.15, 0.9), (0.20, 1.0)]
    with pytest.raises(ValueError):
        list(batch.expand(BASE, {**axes, "tank.tune": [0.9]}, product=False))


def test_every_variant_is_an_independent_copy_and_the_spec_is_not_touched():
    before = copy.deepcopy(BASE)
    variants = [variant for _, variant in batch.expand(BASE, {"sections": [8, 12]})]
    assert [variant["sections"] for variant in variants] == [8, 12]
    variants[0]["secondary"]["turns"] = 1
    assert variants[1]["secondary"] == before["secondary"]
    assert BASE == before


@pytest.mark.parametrize(
    "path", ["top_load.radius", "toploads.major_radius", "sections.deep"]
)
def test_an_axis_that_is_not_in_the_spec_fails_at_expansion(path):
    with pytest.raises(KeyError, match=path):
        list(batch.expand(BASE, {path: [1.0]}))


def test_the_observables_are_what_the_machine_already_carries():
    machine = Machine.from_dict(copy.deepcopy(BASE))
    values = batch.observables(machine)
    inductances = machine.network.inductances
    assert values["frequency"] == machine.frequency
    assert values["primary_inductance"] == inductances[0, 0]
    assert values["tank_capacitance"] == machine.tank.capacitance
    assert values["breakout_voltage"] == machine.breakout().voltage
    assert values["coupling"] == pytest.approx(
        secondary.coupling(machine.design, machine.eigen)[0], rel=1e-12
    )
    assert values["quality"] == pytest.approx(
        losses.quality_factor(machine.design, machine.eigen)[0], rel=1e-12
    )


def test_a_sweep_is_indexed_by_its_axes_and_unstacks_into_a_cube():
    axes = {"top_load.major_radius": RADII, "tank.tune": [0.9, 1.0]}
    frame = batch.sweep(BASE, axes)
    assert list(frame.index.names) == list(axes)
    assert list(frame.index) == list(itertools.product(*axes.values()))
    cube = frame.to_xarray()
    assert cube.frequency.dims == tuple(axes)
    assert cube.frequency.shape == (2, 2)
    assert np.all(np.diff(cube.frequency.values, axis=0) < 0.0)
    assert np.all(np.diff(cube.frequency.values, axis=1) == 0.0)


def test_an_infeasible_variant_is_a_row_of_nan_and_the_sweep_goes_on():
    frame = batch.sweep(BASE, {"top_load.minor_radius": [0.05, 0.20]})
    assert len(frame) == 2
    assert frame.iloc[0].notna().all()
    assert frame.iloc[1].isna().all()
    with pytest.raises(ValueError):
        Machine.from_dict(
            {
                **copy.deepcopy(BASE),
                "top_load": {**BASE["top_load"], "minor_radius": 0.20},
            }
        )


def test_overlapping_rings_are_infeasible_and_not_an_aborted_sweep():
    """Wire thicker than the turn pitch leaves P indefinite, which Cholesky raises on."""
    frame = batch.sweep(BASE, {"secondary.wire_diameter": [4.0e-4, 5.0e-2]})
    assert frame.iloc[0].notna().all()
    assert frame.iloc[1].isna().all()
    with pytest.raises(np.linalg.LinAlgError):
        Machine.from_dict(
            {
                **copy.deepcopy(BASE),
                "secondary": {**BASE["secondary"], "wire_diameter": 5.0e-2},
            }
        )


def test_a_process_pool_returns_the_frame_the_loop_does():
    axes = {"top_load.major_radius": RADII}
    pd.testing.assert_frame_equal(
        batch.sweep(BASE, axes, workers=1), batch.sweep(BASE, axes, workers=2)
    )


def test_an_objective_keeps_its_paths_in_order_and_builds_the_point_it_is_given():
    bounds = {"top_load.major_radius": (0.12, 0.30), "tank.tune": (0.8, 1.2)}
    obj = batch.objective(BASE, bounds, lambda machine: machine.frequency)
    assert obj.names == tuple(bounds)
    assert list(obj.bounds) == list(bounds.values())
    point = {"top_load": {**BASE["top_load"], "major_radius": 0.18}}
    by_hand = Machine.from_dict(
        {**copy.deepcopy(BASE), **point, "tank": {**BASE["tank"], "tune": 1.1}}
    )
    assert obj([0.18, 1.1]) == by_hand.frequency


def test_a_point_the_build_rejects_is_a_wall_and_not_a_crash():
    obj = batch.objective(
        BASE, {"top_load.minor_radius": (0.02, 0.30)}, lambda machine: machine.frequency
    )
    assert math.isfinite(obj([0.05]))
    assert obj([0.25]) == np.inf


def test_a_bounded_scipy_minimisation_walks_the_figure_downhill():
    obj = batch.objective(
        BASE, {"top_load.major_radius": (0.12, 0.30)}, lambda machine: machine.frequency
    )
    start = 0.20
    solution = minimize(
        obj,
        [start],
        bounds=obj.bounds,
        method="L-BFGS-B",
        options={"maxiter": 2, "eps": 0.01},
    )
    assert solution.x[0] > start
    assert solution.fun < obj([start])


def test_performance_reports_the_burst_and_what_it_heats():
    machine = Machine.from_dict(copy.deepcopy(BASE))
    row = batch.performance(
        machine, machine.streamer(growth=2.0, cooling=2.0e-6), passes=2
    )
    assert set(row) == {"power", "length", "junction", "coil", "converged"}
    assert math.isfinite(row["power"]) and row["power"] > 0.0
    assert math.isfinite(row["length"]) and row["length"] > 0.0
    assert row["junction"] > row["coil"] > machine.thermal.ambient
    assert isinstance(row["converged"], bool)
