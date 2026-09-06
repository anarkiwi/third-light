# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

# Dependency layer: rebuilt only when pyproject.toml changes.
FROM python:${PYTHON_VERSION}-slim AS deps
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p thirdlight && touch thirdlight/__init__.py \
    && pip install --no-cache-dir ".[dev]"
# matplotlib writes a font cache; $HOME is not writable in the test stages.
ENV MPLCONFIGDIR=/tmp/matplotlib

FROM deps AS cpu
COPY . /src
RUN pip install --no-cache-dir --no-deps -e .
ENV THIRDLIGHT_BACKEND=cpu
# First pass compiled, for numerics and timing; second interpreted, so coverage
# can trace inside njit kernels.
CMD ["sh", "-c", "pytest && NUMBA_DISABLE_JIT=1 pytest --cov=thirdlight \
    --cov-report=term-missing --cov-fail-under=85"]

FROM deps AS lint
COPY . /src
RUN pip install --no-cache-dir --no-deps -e .
CMD ["sh", "-c", "black --check --diff . && pylint thirdlight && pylint --disable=missing-function-docstring tests"]

# CUDA layer, exercised by a self-hosted runner or manual dispatch.
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS cuda
ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p thirdlight && touch thirdlight/__init__.py \
    && pip install --no-cache-dir ".[dev,cuda]"
COPY . /src
RUN pip install --no-cache-dir --no-deps -e .
ENV THIRDLIGHT_BACKEND=cuda MPLCONFIGDIR=/tmp/matplotlib
CMD ["pytest"]
