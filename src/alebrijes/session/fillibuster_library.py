"""A repo-committed library of pre-rendered "side conversation" audio clips
the listener can pick, live, to play as filler while a checkpoint's
wrong-answer/chat LLM reply is in flight.

Entirely client-driven (see `frontend/quiz.html`): the server's only job is
listing what's available (this module) -- playback, per-clip position
memory, and start/stop timing all live in the browser, reusing the same
`/api/book/audio`/`/api/book/transcript` endpoints the main narration audio
already goes through. No live synthesis happens anywhere in this path,
matching how the main book itself is played (a pre-rendered `.m4b`, never
synthesized on demand during a session).

Each entry is produced by the ordinary `alebrijes render` pipeline like any
other book: a `<id>.m4b` with its own `<id>.transcript.json` sidecar (its
`book_title` is this entry's display title -- no separate metadata file).
A `<id>.quiz.yaml` may also exist if the source script happened to carry its
own `QUIZ:` markers, but the fillibuster picker itself doesn't use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alebrijes.transcript import load as load_transcript


@dataclass
class FillibusterMeta:
    id: str
    title: str
    path: str  # the .m4b's path -- feed straight into /api/book/audio's `path` query param


def list_fillibusters(directory: Path) -> list[FillibusterMeta]:
    """Every `<id>.m4b` with a companion `<id>.transcript.json` in
    `directory`, sorted by id. `[]` if the directory doesn't exist or holds
    no complete pairs -- an `.m4b` with no transcript sidecar is silently
    skipped (no title to show, and it wouldn't have been produced by
    `alebrijes render` in the first place)."""
    if not directory.exists():
        return []
    metas: list[FillibusterMeta] = []
    for m4b_path in sorted(directory.glob("*.m4b")):
        transcript_path = m4b_path.with_suffix(".transcript.json")
        if not transcript_path.exists():
            continue
        transcript = load_transcript(transcript_path)
        metas.append(FillibusterMeta(id=m4b_path.stem, title=transcript.book_title, path=str(m4b_path)))
    return metas
