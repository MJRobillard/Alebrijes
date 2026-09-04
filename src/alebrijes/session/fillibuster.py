"""A folder of pre-rendered, generic filler audio, consumed as a FIFO queue
while a checkpoint's grading/chat call is in flight (see `llm.ollama.
OllamaClient.keep_warm` for the other half of hiding that latency).

Deliberately dependency-free and standalone: this module doesn't import
anything from `session/__init__.py` or `session/buffer.py`, and nothing
there imports this yet -- wiring it into a live session is a later,
separate step (chapter-boundary handoff mode). Built and tested on its own
first.

No manifest file: FIFO order is tracked via each file's own mtime, sorted
ascending -- `next_fillibuster` picks the oldest file, then rotates it to
the back by bumping its mtime past the current newest. This is intentional:
one less piece of state to get out of sync with the directory's actual
contents (add/remove a .wav file by hand and the order just works), at the
cost of relying on the filesystem's mtime granularity/reliability, which is
fine for a local queue like this.

Content is generic/topic-level, not tied to a specific question, so a file
is reusable indefinitely -- "consumed" only means "moved to the back of the
line," never deleted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Fillibuster:
    audio: np.ndarray
    sample_rate: int
    path: Path


def _ordered_wav_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.wav"), key=lambda p: p.stat().st_mtime)


def next_fillibuster(directory: Path) -> Optional[Fillibuster]:
    """Pops the oldest (by mtime) .wav file in `directory`, rotates it to
    the back of the queue, and returns its decoded audio. `None` if the
    directory doesn't exist or holds no .wav files -- callers treat that as
    "no filler available, play silence instead," never as an error."""
    if not directory.exists():
        return None

    files = _ordered_wav_files(directory)
    if not files:
        return None

    chosen = files[0]

    from scipy.io import wavfile as _wavfile

    sample_rate, audio = _wavfile.read(chosen)

    # Rotate to the back: past the current newest mtime, not just "now" --
    # guards against two calls landing in the same filesystem mtime tick
    # (coarse on some filesystems/platforms) and picking the same file twice
    # in a row.
    newest = max(p.stat().st_mtime for p in files)
    bumped = newest + 1
    os.utime(chosen, (bumped, bumped))

    return Fillibuster(audio=np.asarray(audio), sample_rate=sample_rate, path=chosen)
