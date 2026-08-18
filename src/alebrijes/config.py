"""Single source of truth for paths and defaults used across pipeline stages.

pocket-tts-agent scattered these as argparse defaults duplicated across four
scripts (tts-menu, ci-tts, ci-tts-from-epub, epub-reader-gui). Here every stage
takes a `Config` instead of reading its own defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# pocket-tts's built-in voices, usable directly as a CastEntry.voice_state_path
# with no cloning required. Single source of truth for cast/build.py's
# placeholder assignment and the GUI's voice-picker dropdown.
BUILTIN_VOICES = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]


@dataclass
class Config:
    voice_cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "alebrijes" / "voice_prompts")
    synth_cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "alebrijes" / "synth")
    output_dir: Path = field(default_factory=lambda: Path("output"))
    # Where `alebrijes voices add` saves cloned voice states, and where the
    # GUI's voice picker looks for them. Not committed to the repo (Phase 0):
    # empty on a fresh clone until the user runs `alebrijes voices add`.
    voices_dir: Path = field(default_factory=lambda: Path.home() / ".local" / "share" / "alebrijes" / "voices")
    max_workers: int | None = None  # None -> synth.parallel.recommended_worker_count picks one
