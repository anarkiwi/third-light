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
from thirdlight.discharge import Streamer, TreeChannel, breakout
from thirdlight.em import inductance, losses
from thirdlight.geometry import Design
from thirdlight.solver.stepping import simulate
from thirdlight.thermal import Stack, equilibrium

STEPS_PER_CYCLE = 256


def _bridge(spec):
    """Bridge from a mapping of two device mappings, and each device's Zth beside it."""
    spec = dict(spec)
    devices = {kind: dict(spec.pop(kind)) for kind in ("igbt", "diode")}
    zth = {kind: device.pop("zth", None) for kind, device in devices.items()}
    bridge = Bridge(
        igbt=Switch.from_dict(devices["igbt"]),
        diode=Switch.from_dict(devices["diode"]),
        **spec,
    )
    return bridge, zth


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
    thermal: Stack = Stack()

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

    def temperatures(self, streamer=None, **kwargs):
        """Settled junction, coil and tank temperatures; see :mod:`thirdlight.thermal`."""
        return equilibrium(self, streamer, **kwargs)

    def streamer(self, **kwargs):
        """Fritz streamer load for this machine, breaking out at the driven frequency.

        Keyword arguments override the calibrated constants of
        :class:`~thirdlight.discharge.Streamer`; those naming air conditions go
        to :meth:`breakout`.
        """
        air = {k: kwargs.pop(k) for k in ("density", "surface") if k in kwargs}
        return Streamer(
            breakout=self.breakout(**air), frequency=self.frequency, **kwargs
        )

    def channel(self, growth, **kwargs):
        """Tree-backed channel model for this machine; see :mod:`thirdlight.discharge.channel`.

        Keyword arguments go to :class:`~thirdlight.discharge.TreeChannel`; those
        naming air conditions go to :meth:`breakout`.
        """
        air = {k: kwargs.pop(k) for k in ("density", "surface") if k in kwargs}
        return TreeChannel.from_design(
            self.design, self.breakout(**air), self.frequency, growth, **kwargs
        )

    def run(
        self,
        duration,
        step=None,
        load=None,
        x0=None,
        streamer=None,
        length0=0.0,
        rng=None,
    ):
        """Simulate ``duration`` seconds, defaulting to :attr:`step`."""
        return simulate(
            self.network,
            self.driver,
            duration,
            step or self.step,
            load,
            x0,
            streamer,
            length0,
            rng,
        )

    @classmethod
    def from_dict(cls, spec):
        """Build from a plain mapping; keys outside the drive sections are geometry."""
        spec = dict(spec)
        modes = spec.pop("modes", 2)
        tank = dict(spec.pop("tank"))
        bus = Bus(**spec.pop("bus", {}))
        bridge, zth = _bridge(spec.pop("bridge"))
        thermal = Stack.from_dict(spec.pop("thermal", {}), **zth)
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
            thermal=thermal,
            driver=_driver(driver, eigen.f[0]),
            network=from_modes(
                eigen,
                secondary.coupling(design, eigen),
                losses.quality_factor(design, eigen),
                primary,
                tank,
                bridge,
                bus,
                0.0 if design.former is None else design.former.loss_tangent,
            ),
        )

    @classmethod
    def from_yaml(cls, path):
        """Load a machine from the YAML schema in ``docs/schema.md``."""
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))
