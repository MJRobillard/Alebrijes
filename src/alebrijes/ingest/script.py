"""Script-format parsing: source file -> `alebrijes.ingest.Book`.

For a book written (or converted) as an explicit script rather than prose,
where every line already says who's speaking, e.g.:

    # Chapter One
    NARRATOR: The room was quiet.
    COLIN: Let's take another path.
    SABLE: You shouldn't be here.

Every non-blank line must be `NAME: text`, name in ALL CAPS (the strict
casing is what lets a colon inside dialogue text not be mistaken for a
speaker prefix — "Wait: is that true?" doesn't match, `NARRATOR:` does).
`NARRATOR` (case-insensitive) is a convention, not special syntax — it's
just the name that becomes narration instead of dialogue; see
attribute/script.py. A line starting with a single `#` and a space starts
a new chapter, Markdown-heading style; content before the first one (if
any) becomes an implicit first chapter.

This sets `Book.source_format = "script"`, which is what tells
`cast/build.py` and `render.py` to skip `segment`/`attribute`'s guessing
entirely and use the exact, deterministic parse instead.
"""

from __future__ import annotations

import re
from pathlib import Path

from alebrijes.ingest import Book, Chapter, Paragraph

_CHAPTER_RE = re.compile(r"^#\s+(.+)$")
_LINE_RE = re.compile(r"^[A-Z][A-Z0-9' ]*:")


def parse(path: Path) -> Book:
    if not path.exists():
        raise FileNotFoundError(f"Script file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    book_title = path.stem

    chapters: list[Chapter] = []
    chapter_title: str | None = None
    paragraphs: list[Paragraph] = []

    def flush() -> None:
        if not paragraphs:
            return
        title = chapter_title or f"Chapter {len(chapters) + 1}"
        chapters.append(Chapter(title=title, index=len(chapters), paragraphs=list(paragraphs)))
        paragraphs.clear()

    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        chapter_match = _CHAPTER_RE.match(line)
        if chapter_match:
            flush()
            chapter_title = chapter_match.group(1).strip()
            continue

        if not _LINE_RE.match(line):
            raise ValueError(
                f"{path}:{line_number}: expected 'NAME: text' (name in ALL CAPS), got: {line!r}"
            )
        paragraphs.append(Paragraph(text=line, index=len(paragraphs)))

    flush()

    if not chapters:
        raise ValueError(f"No script lines found in '{path}'.")

    return Book(title=book_title, chapters=chapters, source_format="script")
