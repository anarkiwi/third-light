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
| `former.outer_radius` | winding former outer radius; the wire sits outside it |
| `former.inner_radius` | bore radius, 0 for a solid rod |
| `former.length` | axial length of the former |
| `former.base` | height of the former's bottom face |
| `former.permittivity` | relative permittivity, default 2.56 (polystyrene) |
| `ground_plane` | image a conducting plane at z = 0 |
| `sections` | secondary ring sections, 50–400 |
| `top_load_sections` | rings around the top-load surface |
| `former_sections` | bands around the former's closed meridian contour |

See `examples/` for complete designs.
