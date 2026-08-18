"""Segment stage: split a paragraph into narration vs. quoted-dialogue spans.

Each span keeps its chapter/paragraph position so downstream stages (attribute,
cast) can trace a rendered line back to exactly where it came from in the book.

The dialogue/narration split itself is ported and generalized from
pocket-tts-agent's examples/multi_ci_tts.py (QUOTE_RE + _is_probable_dialogue),
which was already genuinely rule-based and book-agnostic — unlike that file's
speaker attribution, which called out to an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto

from alebrijes.ingest import Paragraph

_QUOTE_RE = re.compile(r"[\"“](.*?)[\"”]", re.DOTALL)

_SPEECH_MARKERS = (
    " said",
    " asked",
    " replied",
    " blurted",
    " shouted",
    " yelled",
    " voice",
    " told",
    " wrote",
    " chanting",
    " whispered",
    " muttered",
    " cried",
    " exclaimed",
    " murmured",
    " called",
    " answered",
    " interrupted",
    " continued",
    " added",
)

_CONTEXT_WINDOW = 260


class SpanKind(Enum):
    NARRATION = auto()
    DIALOGUE = auto()


@dataclass
class Span:
    text: str
    kind: SpanKind
    chapter_index: int
    paragraph_index: int


def _clean(chunk: str) -> str:
    return re.sub(r"\s+", " ", chunk).strip()


def _is_probable_dialogue(quote_text: str, before: str, after: str) -> bool:
    # Quoted single words/short phrases in narration are usually emphasis,
    # not speech (e.g. the "twilight" in "influenced by the 'twilight'").
    if len(quote_text.split()) <= 2:
        return False

    around = f"{before} {after}"
    if any(marker in around for marker in _SPEECH_MARKERS):
        return True

    # A colon often introduces direct speech even with no speech verb nearby.
    return before.rstrip().endswith(":")


def segment_paragraph(paragraph: Paragraph, chapter_index: int) -> list[Span]:
    text = paragraph.text
    spans: list[Span] = []

    matches = list(_QUOTE_RE.finditer(text))
    if not matches:
        clean = _clean(text)
        if clean:
            spans.append(Span(clean, SpanKind.NARRATION, chapter_index, paragraph.index))
        return spans

    cursor = 0
    for match in matches:
        narration = _clean(text[cursor : match.start()])
        if narration:
            spans.append(Span(narration, SpanKind.NARRATION, chapter_index, paragraph.index))

        quote_text = _clean(match.group(1))
        if quote_text:
            before = text[max(0, match.start() - _CONTEXT_WINDOW) : match.start()].lower()
            after = text[match.end() : match.end() + _CONTEXT_WINDOW].lower()
            kind = SpanKind.DIALOGUE if _is_probable_dialogue(quote_text, before, after) else SpanKind.NARRATION
            spans.append(Span(quote_text, kind, chapter_index, paragraph.index))

        cursor = match.end()

    tail = _clean(text[cursor:])
    if tail:
        spans.append(Span(tail, SpanKind.NARRATION, chapter_index, paragraph.index))

    return spans
