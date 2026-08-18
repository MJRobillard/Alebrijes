"""Segment stage: split a paragraph into narration vs. quoted-dialogue spans.

Each span keeps its chapter/paragraph position so downstream stages (attribute,
cast) can trace a rendered line back to exactly where it came from in the book.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from alebrijes.ingest import Paragraph


class SpanKind(Enum):
    NARRATION = auto()
    DIALOGUE = auto()


@dataclass
class Span:
    text: str
    kind: SpanKind
    chapter_index: int
    paragraph_index: int


def segment_paragraph(paragraph: Paragraph, chapter_index: int) -> list[Span]:
    raise NotImplementedError("Narration/dialogue span splitting not yet implemented.")
