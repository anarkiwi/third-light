# third-light

Multiphysics solid-state Tesla coil (SSTC / DRSSTC / QCW) simulator. Python with
CUDA acceleration (CuPy + Numba CUDA), CPU fallback.

## Install

```sh
pip install -e ".[dev]"          # CPU
pip install -e ".[dev,cuda]"     # + CuPy, CUDA 12
```

Extras: `io` (xarray, pandas, pyarrow) for labelled output, `viz` (matplotlib)
for plots; `dev` carries both.

Backend selection: `THIRDLIGHT_BACKEND=cuda|cpu`, default auto.

## Use

```python
from thirdlight.geometry import Design
from thirdlight.secondary import resonance, coupling
from thirdlight.em.losses import quality_factor

design = Design.from_yaml("examples/sstc.yaml")
modes = resonance(design, modes=4)
modes.f              # resonant frequencies, Hz
modes.v              # modal voltage profiles over modes.z
coupling(design, modes)      # primary-to-mode k
quality_factor(design, modes)  # unloaded Q
```

```python
from thirdlight.machine import Machine

machine = Machine.from_yaml("examples/drsstc.yaml")
result = machine.run(200e-6)   # bridge, tank and modes, event stepped
result.primary_current         # A, sampled at every step and switching instant
result.top_voltage             # V
```

```python
losses = result.losses(tj=110.0)  # component-resolved energy ledger, J
losses.igbt, losses.diode         # device conduction, by kind
losses.primary, losses.esr        # primary loop, tank capacitor ESR
losses.winding, losses.former     # per secondary mode, and the former dielectric
losses.total                      # conduction total, equal to result.dissipation
losses.switching.total            # commutation energy, additive to it
```

```python
from scipy.optimize import differential_evolution

from thirdlight import batch

frame = batch.sweep(spec, {"top_load.major_radius": [0.14, 0.15, 0.16],
                           "tank.tune": [0.95, 1.0, 1.05]}, workers=4)
frame.to_xarray()                     # the same, as a labelled cube
frame["frequency"]                    # per point; an infeasible point is NaN

obj = batch.objective(spec, {"primary.turns": (4.0, 8.0)},
                      lambda m: abs(m.frequency - 3.0e5))
differential_evolution(obj, obj.bounds)   # or an Optuna trial over obj.names
```

```python
from thirdlight.solver import batched

packed = batched.pack([machine, other, third])   # design-major, one dtype
out = batched.run(packed, 200e-6)                # no Python in the interval loop
out.peak_current, out.input_energy, out.dissipation   # per design
```

```python
from thirdlight import io, viz

io.to_dict(design)                    # design back to the YAML schema, defaults omitted
io.dump(design, "design.yaml")
io.to_dataset(result)                 # xarray Dataset over time and mode
io.to_frame(result)                   # the same, flat and indexed by time
io.to_parquet(result, "run.parquet")

viz.waveforms(result)                 # primary current and top voltage
viz.mode_shapes(modes)                # modal voltage profiles along the coil
viz.losses(result.losses())           # component energy ledger
viz.temperatures(hot)                 # settled cycle, mean and swing
```

```python
from thirdlight.discharge import calibration

machine.breakout().voltage        # top voltage that breaks the electrode out, V
streamer = machine.streamer()     # Fritz channel load, calibrated constants
result = machine.run(200e-6, streamer=streamer)
result.length                     # streamer length, m, per sample
result.streamer_power             # channel dissipation, W
calibration.operating_point(machine, streamer)   # cycle mean power, spark length
```

```python
hot = machine.temperatures(streamer)  # settled interrupter cycle, C
hot.peak["igbt"]                      # peak junction temperature, what kills a die
hot.mean["igbt"], hot.ripple["igbt"]  # cycle mean and the swing on top of it
hot.peak["coil"], hot.peak["capacitor"]   # winding and tank capacitor
hot.converged                         # false for a coil in thermal runaway
```

Implemented: EM matrices and the ladder eigen-solve, the dielectric former and
Medhurst proximity correction, the bridge, driver and exponential integrator,
breakout, the streamer load and its length dynamics, switching-energy and
component-resolved loss extraction, the Foster/Cauer thermal networks,
junction temperature and settled interrupter cycle that consume it, and the
schema round trip, labelled output, plots, design-space sweeps and the optimiser
glue, and the batched stepper on both targets (roadmap phases 1, 1a, 2, 3, 4 and
5). Phase 6 -- DBM streamer geometry, acoustics, JavaTC import and 3D
visualisation -- is not built.

## Test

```sh
pytest                                   # host
docker build --target cpu -t tl . && docker run --rm tl
docker build --target lint -t tl-lint . && docker run --rm tl-lint
```

## Docs

- [docs/design.md](docs/design.md): physics models, numerical core, validation plan, references.
- [docs/schema.md](docs/schema.md): YAML design schema.

## License

Apache 2.0, see [LICENSE](LICENSE).
