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
    span_id: str


def _clean(chunk: str) -> str:
    return re.sub(r"\s+", " ", chunk).strip()


def _is_probable_dialogue(quote_text: str, before: str, after: str, *, opens_paragraph: bool = False) -> bool:
    # Quoted single words/short phrases in narration are usually emphasis,
    # not speech (e.g. the "twilight" in "influenced by the 'twilight'").
    if len(quote_text.split()) <= 2:
        return False

    # A paragraph that opens with a quote is a dialogue turn in fiction, even
    # with no speech tag anywhere near it -- the untagged back-and-forth that
    # the speech-marker check below misses (PDNC's "implicit" quotes: <1%
    # detected without this; see evals/attribution).
    if opens_paragraph:
        return True

    around = f"{before} {after}"
    if any(marker in around for marker in _SPEECH_MARKERS):
        return True

    # A colon often introduces direct speech even with no speech verb nearby.
    return before.rstrip().endswith(":")


def _span_id(chapter_index: int, paragraph_index: int, offset: int) -> str:
    return f"{chapter_index}.{paragraph_index}.{offset}"


def segment_paragraph(paragraph: Paragraph, chapter_index: int) -> list[Span]:
    text = paragraph.text
    spans: list[Span] = []

    def _append(span_text: str, kind: SpanKind) -> None:
        span_id = _span_id(chapter_index, paragraph.index, len(spans))
        spans.append(Span(span_text, kind, chapter_index, paragraph.index, span_id))

    matches = list(_QUOTE_RE.finditer(text))
    if not matches:
        clean = _clean(text)
        if clean:
            _append(clean, SpanKind.NARRATION)
        return spans

    opens_paragraph = matches[0].start() == len(text) - len(text.lstrip())
    cursor = 0
    for match in matches:
        narration = _clean(text[cursor : match.start()])
        if narration:
            _append(narration, SpanKind.NARRATION)

        quote_text = _clean(match.group(1))
        if quote_text:
            before = text[max(0, match.start() - _CONTEXT_WINDOW) : match.start()].lower()
            after = text[match.end() : match.end() + _CONTEXT_WINDOW].lower()
            is_dialogue = _is_probable_dialogue(quote_text, before, after, opens_paragraph=opens_paragraph)
            kind = SpanKind.DIALOGUE if is_dialogue else SpanKind.NARRATION
            _append(quote_text, kind)

        cursor = match.end()

    tail = _clean(text[cursor:])
    if tail:
        _append(tail, SpanKind.NARRATION)

    return spans
