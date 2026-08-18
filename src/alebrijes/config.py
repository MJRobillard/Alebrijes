"""Single source of truth for paths and defaults used across pipeline stages.

pocket-tts-agent scattered these as argparse defaults duplicated across four
scripts (tts-menu, ci-tts, ci-tts-from-epub, epub-reader-gui). Here every stage
takes a `Config` instead of reading its own defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    voice_cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "alebrijes" / "voice_prompts")
    synth_cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "alebrijes" / "synth")
    output_dir: Path = field(default_factory=lambda: Path("output"))
    max_workers: int | None = None  # None -> synth.parallel.recommended_worker_count picks one
