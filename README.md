# third-light

Multiphysics solid-state Tesla coil (SSTC / DRSSTC / QCW) simulator. Python with
CUDA acceleration (CuPy + Numba CUDA), CPU fallback.

## Install

```sh
pip install -e ".[dev]"          # CPU
pip install -e ".[dev,cuda]"     # + CuPy, CUDA 12
```

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
component-resolved loss extraction, and the Foster/Cauer thermal networks,
junction temperature and settled interrupter cycle that consume it (roadmap
phases 1, 1a, 2, 3 and 4). Batched GPU sweeps are not yet built.

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
