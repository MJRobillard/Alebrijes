from __future__ import annotations

import sys
import types

from alebrijes.diagnostics import torch_build_warning


def _fake_torch(cuda_version: str | None, cuda_available: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        __version__="2.6.0+cu118" if cuda_version else "2.13.0+cpu",
        version=types.SimpleNamespace(cuda=cuda_version),
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
    )


def test_warns_when_cuda_build_has_no_usable_gpu(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_version="11.8", cuda_available=False))

    warning = torch_build_warning()

    assert warning is not None
    assert "CUDA 11.8" in warning
    assert "download.pytorch.org/whl/cpu" in warning


def test_no_warning_for_cpu_only_build(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_version=None, cuda_available=False))

    assert torch_build_warning() is None


def test_no_warning_when_cuda_build_has_a_real_gpu(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_version="11.8", cuda_available=True))

    assert torch_build_warning() is None
