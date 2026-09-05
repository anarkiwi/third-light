"""Backend selection and kernel dispatch."""

import numpy as np
import pytest

from thirdlight import backend


@backend.kernel
def _twice(x):
    return 2.0 * x


@pytest.fixture(name="clear_cuda_cache", autouse=True)
def _clear_cuda_cache():
    backend.cuda_available.cache_clear()
    yield
    backend.cuda_available.cache_clear()


def test_kernel_compiles_and_runs():
    assert _twice(2.5) == pytest.approx(5.0)


def test_device_build_does_not_need_a_device():
    assert backend.device(_twice) is backend.device(_twice)


def test_selected_honours_explicit_cpu(monkeypatch):
    monkeypatch.setenv("THIRDLIGHT_BACKEND", "cpu")
    assert backend.selected() == "cpu"
    assert backend.array_namespace() is np


def test_selected_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("THIRDLIGHT_BACKEND", "opencl")
    with pytest.raises(ValueError, match="must be one of"):
        backend.selected()


def test_selected_cuda_without_device_raises(monkeypatch):
    monkeypatch.setenv("THIRDLIGHT_BACKEND", "cuda")
    monkeypatch.setattr(backend, "cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        backend.selected()


def test_auto_follows_availability(monkeypatch):
    monkeypatch.delenv("THIRDLIGHT_BACKEND", raising=False)
    monkeypatch.setattr(backend, "cuda_available", lambda: False)
    assert backend.selected() == "cpu"
    assert backend.array_namespace() is np


def test_auto_selects_cuda_when_available(monkeypatch):
    monkeypatch.delenv("THIRDLIGHT_BACKEND", raising=False)
    monkeypatch.setattr(backend, "cuda_available", lambda: True)
    assert backend.selected() == "cuda"


def test_array_namespace_returns_cupy_on_cuda(monkeypatch):
    cupy = pytest.importorskip("cupy")
    monkeypatch.setattr(backend, "selected", lambda: "cuda")
    assert backend.array_namespace() is cupy


def test_cuda_available_is_false_without_cupy(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "cupy", None)
    assert backend.cuda_available() is False


def test_asnumpy_passes_through_and_unwraps():
    array = np.arange(4.0)
    assert backend.asnumpy(array) is not None
    assert np.array_equal(backend.asnumpy(array), array)

    class Device:
        """Stand-in for a device array."""

        def get(self):
            """Host copy."""
            return array

    assert np.array_equal(backend.asnumpy(Device()), array)
