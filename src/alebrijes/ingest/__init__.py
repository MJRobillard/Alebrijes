"""Ingest stage: turn a source file into a normalized chapter/paragraph tree.

Every format-specific parser (epub, pdf, txt) produces this same shape, so
every later stage (segment, attribute, ...) only needs to know one document
model, not one per input format.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Paragraph:
    text: str
    index: int  # position within the chapter, 0-based


@dataclass
class Chapter:
    title: str
    index: int  # position within the book, 0-based
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass
class Book:
    title: str
    chapters: list[Chapter] = field(default_factory=list)
    # "prose" (default, epub/pdf/txt): attribution is guessed by segment +
    # attribute. "script" (ingest/script.py's NAME: text format): attribution
    # is already explicit in the source, so cast/build.py and render.py skip
    # the guessing entirely -- see attribute/script.py.
    source_format: str = "prose"
    # span_id -> quiz question id, from ingest/script.py's `QUIZ: <id>`
    # marker lines. Only ever populated for source_format == "script"; see
    # lesson.py and session/__init__.py for how these drive LEARNING mode's
    # interrogation checkpoints.
    quiz_checkpoints: dict[str, str] = field(default_factory=dict)
