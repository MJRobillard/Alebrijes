"""Synthesis stage: renders (label, text, voice_state) jobs to audio.

Ported from pocket-tts-agent's `agent.py` and `parallel_synthesis.py` — this is
the one part of the old repo that was already working and tested, so it moved
over with import-path updates only, not a rewrite.
"""

from .agent import TTSAgent, create_voice_state
from .parallel import (
    build_m4b,
    merge_wav_files,
    recommended_worker_count,
    run_parallel_synthesis,
)

__all__ = [
    "TTSAgent",
    "create_voice_state",
    "build_m4b",
    "merge_wav_files",
    "recommended_worker_count",
    "run_parallel_synthesis",
]
