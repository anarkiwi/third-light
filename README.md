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

design = Design.from_yaml("examples/sstc.yaml")
```

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
