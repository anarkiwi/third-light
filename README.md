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
from thirdlight.discharge import calibration

machine.breakout().voltage        # top voltage that breaks the electrode out, V
streamer = machine.streamer()     # Fritz channel load, calibrated constants
result = machine.run(200e-6, streamer=streamer)
result.length                     # streamer length, m, per sample
result.streamer_power             # channel dissipation, W
calibration.operating_point(machine, streamer)   # cycle mean power, spark length
```

Implemented: EM matrices and the ladder eigen-solve, the dielectric former and
Medhurst proximity correction, the bridge, driver and exponential integrator, and
breakout, the streamer load and its length dynamics (roadmap phases 1, 1a, 2 and
3). Thermal models and batched GPU sweeps are not yet built.

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
