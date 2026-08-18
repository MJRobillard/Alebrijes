"""Cheap runtime checks surfaced to the user, not just logged and lost.

First one: detecting a CUDA-compiled torch build running on a machine with
no usable GPU — confirmed for real on a Windows test machine (torch
2.6.0+cu118, `torch.cuda.is_available()` False). `pip install -e ".[dev]"`
pulls whatever build the default PyPI index resolves to, with no preference
for CPU vs. CUDA, so it's easy to end up here even though this project's
whole Phase 4 sizing assumed the ~190MB CPU-only wheel. Not fatal — CUDA
builds still run CPU inference correctly — just needless size and load-time
overhead the correct install avoids.
"""

from __future__ import annotations


def torch_build_warning() -> str | None:
    import torch

    if torch.version.cuda is not None and not torch.cuda.is_available():
        return (
            f"torch {torch.__version__} was built for CUDA {torch.version.cuda}, but no usable GPU was "
            "found on this machine — it's running CPU inference through a CUDA build, which is larger "
            "and slower to load than necessary. Reinstall the CPU-only build: "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )
    return None
