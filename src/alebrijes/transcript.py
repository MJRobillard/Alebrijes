"""Transcript sidecar: per-span timing, written alongside a rendered .m4b.

Rendering already knows each span's audio duration (`synthesize_chapter`
computes it before concatenating spans into a chapter's WAV) but discarded
it once concatenated. This module captures that instead, so a player can map
playback position -> span (for a live "now narrating" transcript view) and a
future RAG index can filter retrieval to what's been heard so far.

JSON rather than YAML (unlike `cast.py`): this file is dense, line-per-span
data meant for a player/index to load programmatically, not hand-edited.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from alebrijes.segment import SpanKind


@dataclass
class ChapterMark:
    index: int
    title: str
    start_ms: int
    end_ms: int


@dataclass
class SpanTiming:
    span_id: str
    chapter_index: int
    paragraph_index: int
    character: str
    kind: SpanKind
    text: str
    start_ms: int
    end_ms: int


@dataclass
class Transcript:
    book_title: str
    chapters: list[ChapterMark] = field(default_factory=list)
    spans: list[SpanTiming] = field(default_factory=list)


def load(path: Path) -> Transcript:
    data = json.loads(path.read_text())
    chapters = [ChapterMark(**chapter) for chapter in data.get("chapters", [])]
    spans = [
        SpanTiming(**{**span, "kind": SpanKind[span["kind"]]}) for span in data.get("spans", [])
    ]
    return Transcript(book_title=data["book_title"], chapters=chapters, spans=spans)


def save(transcript: Transcript, path: Path) -> None:
    payload = {
        "book_title": transcript.book_title,
        "chapters": [
            {"index": c.index, "title": c.title, "start_ms": c.start_ms, "end_ms": c.end_ms}
            for c in transcript.chapters
        ],
        "spans": [
            {
                "span_id": s.span_id,
                "chapter_index": s.chapter_index,
                "paragraph_index": s.paragraph_index,
                "character": s.character,
                "kind": s.kind.name,
                "text": s.text,
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
            }
            for s in transcript.spans
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
