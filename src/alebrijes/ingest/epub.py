"""EPUB parsing: source file -> `alebrijes.ingest.Book`.

Not yet ported. pocket-tts-agent's `examples/ci_tts_from_epub.py` (in the old
`~/projects/audio` repo) already has working chapter- and percent-range
extraction logic — this module's job in the next work session is to lift that
logic out from under its CLI argument parsing and CLI-specific I/O, and
express it as `parse(path: Path) -> Book` using the shared `Book`/`Chapter`/
`Paragraph` shape from `alebrijes.ingest`, so it can be unit tested against a
fixture epub without invoking any CLI or synthesizing any audio.
"""

from __future__ import annotations

from pathlib import Path

from alebrijes.ingest import Book


def parse(path: Path) -> Book:
    raise NotImplementedError(
        "EPUB ingest not yet ported from pocket-tts-agent's examples/ci_tts_from_epub.py"
    )
