# Design schema

A design is a YAML mapping loaded by `thirdlight.geometry.Design.from_yaml`.
All lengths are metres, measured from the ground plane at z = 0.

| Key | Meaning |
|---|---|
| `secondary.radius` | winding radius to wire centre |
| `secondary.length` | axial length of the winding |
| `secondary.turns` | total turns |
| `secondary.wire_diameter` | bare conductor diameter |
| `secondary.base` | height of the bottom turn |
| `primary.inner_radius` | radius of the first turn |
| `primary.turns` | turns, fractional allowed |
| `primary.pitch` | radial gain per turn (0 for a helix) |
| `primary.rise` | axial gain per turn (0 for a flat spiral) |
| `primary.base` | height of the first turn |
| `primary.wire_diameter` | conductor or tube diameter |
| `top_load.kind` | `toroid` or `sphere` |
| `top_load.major_radius` | toroid centreline radius |
| `top_load.minor_radius` | toroid tube radius |
| `top_load.radius` | sphere radius |
| `top_load.height` | height of the top-load centre |
| `breakout.radius` | breakout point tip radius; the point is a sphere on the top node |
| `breakout.height` | height of the tip centre, clear of the top load |
| `former.outer_radius` | winding former outer radius; the wire sits outside it |
| `former.inner_radius` | bore radius, 0 for a solid rod |
| `former.length` | axial length of the former |
| `former.base` | height of the former's bottom face |
| `former.permittivity` | relative permittivity, default 2.56 (polystyrene) |
| `ground_plane` | image a conducting plane at z = 0 |
| `sections` | secondary ring sections, 50–400 |
| `top_load_sections` | rings around the top-load surface |
| `breakout_sections` | rings around the breakout point |
| `former_sections` | bands around the former's closed meridian contour |

`thirdlight.geometry.Design.from_yaml` reads the keys above. A file that also
carries the drive sections below is a complete machine, loaded by
`thirdlight.machine.Machine.from_yaml`; the tank capacitance and the phase lead
are resolved against the first secondary mode at load time.

| Key | Meaning |
|---|---|
| `modes` | secondary eigenmodes carried into the time domain, 1–16 |
| `tank.capacitance` | primary series capacitance, F |
| `tank.tune` | in place of `capacitance`: primary resonance as a multiple of mode 1 |
| `tank.resistance` | primary loop resistance including tank ESR, ohm |
| `tank.inductance` | override the primary inductance taken from the geometry |
| `bus.capacitance` | DC bus reservoir, F; omit for a stiff bus held at the driver's voltage |
| `bus.resistance` | rectifier and mains series resistance, ohm; positive with a reservoir |
| `bridge.igbt.v0`, `bridge.igbt.r` | IGBT knee voltage and slope resistance |
| `bridge.diode.v0`, `bridge.diode.r` | anti-parallel diode, same form |
| `bridge.full` | true for a full bridge, false for a half bridge |
| `driver.lead_angle` | current-transformer phase lead in degrees at mode 1 |
| `driver.delay` | comparator plus gate propagation delay, s |
| `driver.dead_time` | dead time inserted at each polarity reversal, s |
| `driver.bus` | flat bus voltage, V, when there is no ramp |
| `driver.ramp.initial`, `.final`, `.rise` | QCW bus envelope measured from each burst start |
| `driver.interrupter.on_time` | burst length, s |
| `driver.interrupter.frequency` | bursts per second |
| `driver.interrupter.notes` | in place of `frequency`: `[[start, duration, midi note], ...]` |

The streamer is not part of the schema. It is built from a loaded machine by
`thirdlight.machine.Machine.streamer()`, which carries the calibrated constants
of `thirdlight.discharge.streamer` and needs the driven frequency the machine
already knows; keyword arguments override any of them.

See `examples/` for complete designs.
