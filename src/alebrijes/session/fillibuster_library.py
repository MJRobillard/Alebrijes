"""A repo-committed library of full scripts the listener can choose to have
read aloud, live, while waiting on an LLM call (a checkpoint reaction or a
wrong-answer/chat reply) -- replaces `session.fillibuster`'s old mechanism
(rotating short, generic pre-rendered `.wav` filler clips) entirely.

Each entry is a `<id>.txt` (plain prose, read start to finish in the
session's own narrator voice via the same live-synthesis pipeline real
narration/responses use -- no `NAME:`/`QUIZ:` script syntax; this isn't a
book, just text to read) plus a companion `<id>.yaml` (`title`/
`description` metadata for the picker on the quiz start form).

The quiz's own script + quiz.yaml stay upload-at-run-time and are handled
entirely separately (`ingest.script`, `lesson.py`) -- this module is only
for the repo-committed filler-content library.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class FillibusterMeta:
    id: str
    title: str
    description: str = ""


def list_fillibusters(directory: Path) -> list[FillibusterMeta]:
    """Every `<id>.txt` with a companion `<id>.yaml` in `directory`, sorted
    by id. `[]` if the directory doesn't exist or holds no complete pairs --
    a `.txt` with no metadata sibling is silently skipped rather than
    guessed at, since title/description drive what the listener sees in the
    picker."""
    if not directory.exists():
        return []
    metas: list[FillibusterMeta] = []
    for txt_path in sorted(directory.glob("*.txt")):
        yaml_path = txt_path.with_suffix(".yaml")
        if not yaml_path.exists():
            continue
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        metas.append(
            FillibusterMeta(
                id=txt_path.stem,
                title=data.get("title", txt_path.stem),
                description=data.get("description", ""),
            )
        )
    return metas


def load_fillibuster_text(directory: Path, fillibuster_id: str) -> str:
    """Raw text of one library entry by id. Raises `FileNotFoundError` if
    missing -- unlike `list_fillibusters`'s quiet skip, a caller resolving an
    id the listener actually selected wants to know if it's gone, not have
    it silently vanish from the read order."""
    path = directory / f"{fillibuster_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Fillibuster script not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_fillibuster_sentences(directory: Path, fillibuster_ids: list[str]) -> list[str]:
    """Concatenates the given library entries, in the given (listener-
    chosen) order, into one flat list of sentences -- seeds
    `InteractiveSession.fillibuster_sentences` at session start. Reuses
    `synth.streaming.SentenceChunker`'s sentence-boundary splitting (the
    same one LLM output goes through) rather than a separate splitter, so a
    fillibuster script and a live LLM reply sound the same going into
    synthesis."""
    from alebrijes.synth.streaming import SentenceChunker

    sentences: list[str] = []
    for fillibuster_id in fillibuster_ids:
        text = load_fillibuster_text(directory, fillibuster_id)
        chunker = SentenceChunker()
        sentences.extend(chunker.feed(text))
        tail = chunker.flush()
        if tail:
            sentences.append(tail)
    return sentences
