"""Backend selection: array namespace for library linear algebra, and kernel dispatch.

The two mechanisms are separate. Dense linear algebra goes through an array
namespace handle bound to NumPy/SciPy or CuPy. Elementwise numerical kernels are
compiled by ``numba.njit`` for the CPU and ``numba.cuda.jit`` for the GPU from
one source, because Numba's targets share no namespace with either library.
"""

import functools
import os

import numpy as np
import numba

_ENV = "THIRDLIGHT_BACKEND"


@functools.lru_cache(maxsize=1)
def cuda_available():
    """True if a usable CUDA device and CuPy are present."""
    try:
        import cupy  # pylint: disable=import-outside-toplevel

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pylint: disable=broad-except
        return False


def selected():
    """Return the active backend name, "cuda" or "cpu"."""
    choice = os.environ.get(_ENV, "auto").lower()
    if choice not in ("auto", "cpu", "cuda"):
        raise ValueError(f"{_ENV} must be one of auto, cpu, cuda; got {choice!r}")
    if choice == "cuda":
        if not cuda_available():
            raise RuntimeError(f"{_ENV}=cuda but no CUDA device is available")
        return "cuda"
    if choice == "cpu":
        return "cpu"
    return "cuda" if cuda_available() else "cpu"


def array_namespace():
    """Return the array module (``numpy`` or ``cupy``) for the active backend."""
    if selected() == "cuda":
        import cupy  # pylint: disable=import-outside-toplevel

        return cupy
    return np


def asnumpy(x):
    """Return ``x`` as a NumPy array, copying off the device if needed."""
    return np.asarray(x.get() if hasattr(x, "get") else x)


def kernel(func):
    """Compile ``func`` as an ``njit`` scalar kernel usable from CPU kernels."""
    return numba.njit(cache=True)(func)


@functools.lru_cache(maxsize=None)
def device(func):
    """CUDA device-function build of a :func:`kernel`, compiled on first use."""
    from numba import cuda  # pylint: disable=import-outside-toplevel

    return cuda.jit(device=True, inline=True)(getattr(func, "py_func", func))
