"""Ingest stage: turn a source file into a normalized chapter/paragraph tree.

Every format-specific parser (epub, pdf, txt) produces this same shape, so
every later stage (segment, attribute, ...) only needs to know one document
model, not one per input format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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


def parse_book(book_path: Path) -> Book:
    """Parses `book_path` with the ingest module for its extension.

    Kept in one place so the CLI and the server support exactly the same
    set of formats and fail the same way on an unsupported one, rather than
    each hand-rolling this dispatch.

    A `.txt` file gets sniffed before being handed to the prose parser: one
    already written in `ingest/script.py`'s strict `NAME: text` form (just
    saved with a `.txt` extension instead of `.script`) is routed there
    instead, so its explicit speakers are used deterministically rather than
    guessed at by segment/attribute -- see `attribute/rules.py`'s "NAME:"
    labeled-line fallback for the case where that guess still has to happen
    (e.g. a `NAME:`-labeled snippet embedded inside otherwise-ordinary prose).
    """
    suffix = book_path.suffix.lower()
    if suffix == ".epub":
        from alebrijes.ingest.epub import parse
    elif suffix == ".pdf":
        from alebrijes.ingest.pdf import parse
    elif suffix == ".txt":
        from alebrijes.ingest.script import looks_like_script
        from alebrijes.ingest.script import parse as parse_script
        from alebrijes.ingest.txt import parse as parse_txt

        raw = book_path.read_text(encoding="utf-8-sig", errors="replace")
        parse = parse_script if looks_like_script(raw) else parse_txt
    elif suffix == ".script":
        from alebrijes.ingest.script import parse
    else:
        raise NotImplementedError(
            f"Ingest for '{suffix}' files is not supported. Supported formats: .epub, .pdf, .txt, .script."
        )
    return parse(book_path)
