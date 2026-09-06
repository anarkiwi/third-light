"""Loading a complete machine from YAML and running it."""

import copy
import math

import numpy as np
import pytest
import yaml

from thirdlight.control import Melody, Ramp
from thirdlight.geometry import Design
from thirdlight.machine import STEPS_PER_CYCLE, Machine

with open("examples/drsstc.yaml", encoding="utf-8") as _handle:
    SPEC = yaml.safe_load(_handle)
SMALL = copy.deepcopy(SPEC)
SMALL["secondary"]["turns"] = 60
SMALL["sections"] = 20
SMALL["top_load_sections"] = 8


def small(**changes):
    """The example machine shrunk to a size the interpreted coverage pass can run."""
    spec = copy.deepcopy(SMALL)
    spec.update(changes)
    return Machine.from_dict(spec)


def test_the_geometry_only_example_still_loads_as_a_design():
    """A machine spec is a design spec plus drive sections, and the older file works."""
    design = Design.from_yaml("examples/sstc.yaml")
    assert design.secondary.turns == 1200
    assert Machine.from_dict(SMALL).design.secondary.turns == 60


def test_the_tank_is_tuned_to_the_driven_mode():
    """tune: 1.0 places the primary resonance on the first secondary mode."""
    machine = small()
    l_p = machine.network.inductances[0, 0]
    primary = 1.0 / (2.0 * math.pi * math.sqrt(l_p * machine.tank.capacitance))
    assert primary == pytest.approx(machine.frequency, rel=1e-12)
    assert machine.step == pytest.approx(1.0 / (STEPS_PER_CYCLE * machine.frequency))


def test_an_explicit_capacitance_overrides_tuning():
    """Without ``tune`` the tank takes the capacitance as given."""
    machine = small(tank={"capacitance": 47e-9, "resistance": 0.05})
    assert machine.tank.capacitance == 47e-9


def test_the_phase_lead_resolves_against_the_driven_mode():
    """``lead_angle`` is degrees at the first mode, and becomes a tau."""
    machine = small()
    assert machine.driver.lead.angle(machine.frequency) == pytest.approx(12.0)
    assert machine.driver.dead_time == 0.4e-6
    assert machine.driver.interrupter.frequency == 200.0


def test_a_melody_and_a_ramp_load_from_the_schema():
    """MIDI notes drive the interrupter and a ramp sets the bus."""
    machine = small(
        driver={
            "lead_angle": 10.0,
            "bus": 300.0,
            "interrupter": {"on_time": 1e-4, "notes": [[0.0, 0.5, 69], [0.5, 0.5, 72]]},
            "ramp": {"final": 300.0, "initial": 20.0, "rise": 2e-3},
        }
    )
    assert isinstance(machine.driver.interrupter, Melody)
    assert isinstance(machine.driver.ramp, Ramp)
    assert machine.driver.interrupter.notes[0] == (0.0, 0.5, 69)


def test_a_short_run_of_the_shrunken_machine():
    """The whole chain runs: geometry to matrices to modes to state space to waveforms."""
    machine = small()
    result = machine.run(20.0 / machine.frequency)
    assert len(result) > 20 * STEPS_PER_CYCLE
    assert np.abs(result.primary_current).max() > 0.0
    assert np.abs(result.top_voltage).max() > np.abs(result.tank_voltage).max()


@pytest.mark.slow
def test_the_example_drsstc_locks_to_the_lower_coupled_split():
    """A self-oscillating driver runs at the lower pole, not at the bare resonance.

    The offset from the pole is the residual of the gate delay after the phase lead.
    """
    machine = Machine.from_yaml("examples/drsstc.yaml")
    inductances = machine.network.inductances
    k = inductances[0, 1] / math.sqrt(inductances[0, 0] * inductances[1, 1])
    assert 1e5 < machine.frequency < 2e5
    assert 0.1 < k < 0.3
    result = machine.run(120e-6)
    current = result.primary_current
    live = current != 0.0
    crossings = result.t[live][np.flatnonzero(np.diff(np.sign(current[live])))]
    oscillation = 0.5 / np.mean(np.diff(crossings))
    assert oscillation == pytest.approx(
        machine.frequency / math.sqrt(1.0 + k), rel=0.05
    )
    assert np.abs(result.top_voltage).max() > 1e5
    assert np.abs(current).max() > 1e2


def test_a_bus_reservoir_loads_from_the_schema_and_sags():
    """A finite bus becomes the last state and droops as the bridge draws on it."""
    machine = small(bus={"capacitance": 2e-5, "resistance": 1.0}, driver=SPEC["driver"])
    assert machine.bus.reservoir
    assert machine.network.size == 2 * (machine.network.modes + 1) + 1
    result = machine.run(20.0 / machine.frequency)
    assert result.bus_voltage[-1] < machine.driver.bus
    assert result.bus_voltage[-1] > 0.0


def test_a_stiff_bus_is_the_default():
    """Without a bus section the supply voltage is held, and reported as the bus."""
    machine = small()
    assert not machine.bus.reservoir
    result = machine.run(5.0 / machine.frequency)
    assert np.all(result.bus_voltage == machine.driver.bus)


def test_a_machine_reports_the_breakout_functional_of_its_electrode():
    """The field per unit modal state is built from the same ladder as the modes."""
    machine = small(breakout={"radius": 0.006, "height": 0.70})
    hot = machine.breakout()
    assert hot.field.shape == (SMALL["top_load_sections"] + 8, machine.network.modes)
    assert 2e4 < hot.voltage < 5e5
    assert hot.margin(np.zeros(machine.network.modes)) == 0.0
    blunt = small().breakout()
    assert hot.voltage < blunt.voltage
    thin = machine.breakout(density=0.8).voltage
    assert thin < hot.voltage
