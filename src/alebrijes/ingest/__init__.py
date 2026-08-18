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
