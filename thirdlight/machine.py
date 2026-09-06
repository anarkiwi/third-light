"""A complete machine: coil geometry, primary tank, bridge and driver.

The YAML schema of :mod:`thirdlight.geometry` extended with ``tank``, ``bridge``
and ``driver`` sections. Loading resolves the two quantities that are naturally
given against the resonance -- the tuned tank capacitance and the phase lead in
degrees -- so a :class:`Machine` carries a built state space, ready to run.
"""

from dataclasses import dataclass

import yaml

from thirdlight import secondary
from thirdlight.circuit import Bridge, Bus, Network, Switch, Tank, from_modes, tune
from thirdlight.control import Driver, Interrupter, Melody, PhaseLead, Ramp
from thirdlight.discharge import breakout
from thirdlight.em import inductance, losses
from thirdlight.geometry import Design
from thirdlight.solver.stepping import simulate

STEPS_PER_CYCLE = 256


def _bridge(spec):
    """Bridge from a mapping of two device mappings and the topology flag."""
    spec = dict(spec)
    return Bridge(
        igbt=Switch(**spec.pop("igbt")), diode=Switch(**spec.pop("diode")), **spec
    )


def _driver(spec, frequency):
    """Driver from a mapping, with the phase lead given in degrees at ``frequency``."""
    spec = dict(spec)
    lead = PhaseLead.from_angle(spec.pop("lead_angle", 0.0), frequency)
    gating = spec.pop("interrupter", None)
    if gating is not None:
        gating = dict(gating)
        notes = gating.pop("notes", None)
        gating = (
            Interrupter(**gating)
            if notes is None
            else Melody(notes=tuple(map(tuple, notes)), **gating)
        )
    ramp = spec.pop("ramp", None)
    return Driver(
        lead=lead,
        interrupter=gating,
        ramp=None if ramp is None else Ramp(**ramp),
        **spec,
    )


@dataclass(frozen=True)
class Machine:  # pylint: disable=too-many-instance-attributes
    """A coil and its drive electronics, with the state space they form."""

    design: Design
    tank: Tank
    bridge: Bridge
    bus: Bus
    driver: Driver
    network: Network
    ladder: secondary.Ladder
    eigen: secondary.Modes

    def breakout(self, **air):
        """Breakout functional of the top-node electrode; see :mod:`thirdlight.discharge`."""
        return breakout.from_modes(self.design, self.ladder, self.eigen, **air)

    @property
    def frequency(self):
        """Driven resonance, the first secondary mode, Hz."""
        return float(self.network.frequencies[0])

    @property
    def step(self):
        """Nominal integration step, a fixed fraction of the driven period."""
        return 1.0 / (STEPS_PER_CYCLE * self.frequency)

    def run(self, duration, step=None, load=None, x0=None):
        """Simulate ``duration`` seconds, defaulting to :attr:`step`."""
        return simulate(
            self.network, self.driver, duration, step or self.step, load, x0
        )

    @classmethod
    def from_dict(cls, spec):
        """Build from a plain mapping; keys outside the drive sections are geometry."""
        spec = dict(spec)
        modes = spec.pop("modes", 2)
        tank = dict(spec.pop("tank"))
        bus = Bus(**spec.pop("bus", {}))
        bridge = _bridge(spec.pop("bridge"))
        driver = spec.pop("driver", {})
        design = Design.from_dict(spec)
        rungs = secondary.ladder(design)
        eigen = secondary.eigenmodes(rungs, modes)
        primary = float(inductance.inductance_matrix(design.primary_rings()).sum())
        ratio = tank.pop("tune", None)
        if ratio is not None:
            tank["capacitance"] = tune(primary, eigen.f[0], ratio)
        tank = Tank(**tank)
        return cls(
            design=design,
            ladder=rungs,
            eigen=eigen,
            tank=tank,
            bridge=bridge,
            bus=bus,
            driver=_driver(driver, eigen.f[0]),
            network=from_modes(
                eigen,
                secondary.coupling(design, eigen),
                losses.quality_factor(design, eigen),
                primary,
                tank,
                bridge,
                bus,
            ),
        )

    @classmethod
    def from_yaml(cls, path):
        """Load a machine from the YAML schema in ``docs/schema.md``."""
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))
