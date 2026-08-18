"""Points huggingface_hub at bundled model weights when running as a
packaged (PyInstaller) app, so pocket-tts never needs network access or a
Hugging Face account at runtime — the weights are fetched once at build
time (see PHASES.md Phase 4) using a project-owned, gate-accepted token,
then bundled into the app the same way torch/spacy/ffmpeg are.

Must run before anything imports huggingface_hub (transitively, before
anything imports pocket_tts) — huggingface_hub reads its cache-location env
vars at import time, not lazily per call. Both CLI and GUI entry points
call this first thing, before any other alebrijes import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_bundled_hf_cache() -> None:
    if not getattr(sys, "frozen", False):
        return  # source/dev install: use the developer's own HF login (see README)

    base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    bundled_cache = base / "hf_cache"
    if not bundled_cache.exists():
        return

    os.environ["HF_HOME"] = str(bundled_cache)
    # Forces huggingface_hub to use only the bundled cache and never attempt
    # a network call, even if something's missing from the bundle -- a
    # silent network fetch would be both a privacy regression (this app's
    # whole positioning is "local and private") and pointless for most
    # users, who won't have accepted pocket-tts's HF gate themselves.
    os.environ["HF_HUB_OFFLINE"] = "1"
