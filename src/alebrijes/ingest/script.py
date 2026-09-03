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

A `QUIZ: <id>` line attaches a lesson practice question (by id, looked up
from a companion `<name>.quiz.yaml` — see `lesson.py`) to the position of
the line immediately before it. It's a directive, not a speaker — `QUIZ` is
reserved and can't be used as a character name — and produces no spoken
span of its own.

This sets `Book.source_format = "script"`, which is what tells
`cast/build.py` and `render.py` to skip `segment`/`attribute`'s guessing
entirely and use the exact, deterministic parse instead.
"""

from __future__ import annotations

import re
from pathlib import Path

from alebrijes.ingest import Book, Chapter, Paragraph

_CHAPTER_RE = re.compile(r"^#\s+(.+)$")
_QUIZ_RE = re.compile(r"^QUIZ:\s*(\S+)\s*$")
_LINE_RE = re.compile(r"^[A-Z][A-Z0-9' ]*:")


def parse(path: Path) -> Book:
    if not path.exists():
        raise FileNotFoundError(f"Script file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    book_title = path.stem

    chapters: list[Chapter] = []
    chapter_title: str | None = None
    paragraphs: list[Paragraph] = []
    quiz_checkpoints: dict[str, str] = {}

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

        quiz_match = _QUIZ_RE.match(line)
        if quiz_match:
            if not paragraphs:
                raise ValueError(
                    f"{path}:{line_number}: QUIZ marker has no preceding line in this chapter to attach to"
                )
            # The chapter this paragraph belongs to hasn't been flushed
            # yet -- `len(chapters)` is exactly the index it'll get when it
            # is, same span_id scheme as attribute/script.py.
            span_id = f"{len(chapters)}.{paragraphs[-1].index}.0"
            quiz_checkpoints[span_id] = quiz_match.group(1)
            continue

        if not _LINE_RE.match(line):
            raise ValueError(
                f"{path}:{line_number}: expected 'NAME: text' (name in ALL CAPS), got: {line!r}"
            )
        paragraphs.append(Paragraph(text=line, index=len(paragraphs)))

    flush()

    if not chapters:
        raise ValueError(f"No script lines found in '{path}'.")

    return Book(
        title=book_title, chapters=chapters, source_format="script", quiz_checkpoints=quiz_checkpoints
    )
