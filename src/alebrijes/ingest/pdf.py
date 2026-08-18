"""PDF parsing: source file -> `alebrijes.ingest.Book`.

Ports the header/footer-stripping and page-marker heuristics from
pocket-tts-agent's `examples/ci_tts_from_pdf.py` (pypdf-based), and adds
mitigations for pypdf's well-documented rough edges and the common
real-world PDF failure modes:

- **Missing spaces / bad reading order**: a long-standing, still-open pypdf
  issue for some PDFs (github.com/py-pdf/pypdf#1328). Extraction uses pypdf's
  `extraction_mode="layout"`, which was built specifically to improve
  spacing and column ordering, falling back to plain mode if layout mode
  isn't available on the installed pypdf version. Best-effort, not a fix —
  pypdf's own docs acknowledge this is a fundamentally hard problem given how
  PDF stores text.
- **Ligatures** (ﬁ, ﬀ, ﬃ as single glyphs): normalized to ASCII via
  `unicodedata.normalize("NFKC", ...)`.
- **Word-wrap hyphenation** ("exam-" / "ple" -> "example"): joined by
  `_dehyphenate_lines`, a heuristic — see its docstring for the real
  ambiguity it can't fully resolve (a wrapped word vs. a genuine hyphenated
  compound that happens to break at a line end).
- **Encrypted PDFs**: many real-world "encrypted" PDFs only set permission
  flags with a blank user password, so an empty-password decrypt is tried
  before giving up with a clear error.
- **Scanned/image-only PDFs**: detected (no extractable text on any page)
  and reported clearly rather than silently producing an empty book — OCR is
  out of scope.
- **Corrupt/unreadable PDFs**: pypdf's `PdfReadError` is reported as a clear
  error rather than propagating a library-internal exception.

Chapter detection reuses the same "Chapter N" / "N:" heading heuristic as
`alebrijes.ingest.epub`, checked only against each page's first line (a PDF
page boundary makes the epub module's stronger "table of contents lists many
matches" guard unnecessary here, though a ToC page whose first line happens
to look like a heading is a known, accepted false-positive risk). Falls back
to a single chapter when no headings are found, same as the epub module.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

from alebrijes.ingest import Book, Chapter, Paragraph

_CHAPTER_HEADER_RE = re.compile(r"^\s*(?:chapter|chap\.?)\s*(\d+)\b|^\s*(\d+)\s*[:.]\s+", re.IGNORECASE)
_HYPHEN_WRAP_RE = re.compile(r"[a-z]-$")
_TERMINAL_PUNCTUATION_RE = re.compile(r"[.!?:;]$")


def _looks_like_page_marker(line: str) -> bool:
    normalized = line.strip().lower()
    if not normalized:
        return False
    return bool(re.fullmatch(r"(page\s*)?\d+", normalized) or re.fullmatch(r"\d+\s*(/|of)\s*\d+", normalized))


def _strip_headers_and_footers(page_lines: list[list[str]]) -> list[list[str]]:
    """Drops running headers/footers (repeated across pages) and page numbers."""
    top_lines = [lines[0] for lines in page_lines if lines]
    bottom_lines = [lines[-1] for lines in page_lines if lines]
    repeated_top = {line for line, count in Counter(top_lines).items() if count > 1}
    repeated_bottom = {line for line, count in Counter(bottom_lines).items() if count > 1}

    cleaned: list[list[str]] = []
    for lines in page_lines:
        lines = list(lines)
        while lines and (lines[0] in repeated_top or _looks_like_page_marker(lines[0])):
            lines.pop(0)
        while lines and (lines[-1] in repeated_bottom or _looks_like_page_marker(lines[-1])):
            lines.pop()
        cleaned.append(lines)
    return cleaned


def _dehyphenate_lines(lines: list[str]) -> list[str]:
    """Joins likely word-wrap hyphenation ("exam-" / "ple" -> "example").

    Heuristic only: a lowercase letter immediately before a trailing hyphen,
    followed by a lowercase-starting next line, triggers a merge. This can't
    perfectly distinguish a wrapped word from a genuine hyphenated compound
    that happens to break at a line end (e.g. "well-known" split across
    lines) — an accepted limitation, not a guarantee, same spirit as pypdf's
    own documented ligature/spacing caveats.
    """
    merged: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines) and _HYPHEN_WRAP_RE.search(line) and lines[i + 1][:1].islower():
            merged.append(line[:-1] + lines[i + 1])
            i += 2
            continue
        merged.append(line)
        i += 1
    return merged


def _group_into_paragraphs(lines: list[str]) -> list[str]:
    """Groups wrapped lines into paragraphs, splitting after terminal punctuation.

    PDF text extraction doesn't reliably preserve blank-line paragraph
    breaks the way EPUB's block-level HTML tags do, so this uses the same
    "a line ending in . ! ? : ;  closes a paragraph" heuristic
    pocket-tts-agent's `_ensure_terminal_pause` was already built on.
    """
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        current.append(line)
        if _TERMINAL_PUNCTUATION_RE.search(line):
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def _ensure_decrypted(reader, path: Path) -> None:
    if not getattr(reader, "is_encrypted", False):
        return
    try:
        result = reader.decrypt("")
    except Exception:
        result = 0
    if not result:
        raise ValueError(
            f"'{path}' is password-protected and the empty password didn't unlock it. "
            "Decrypt it first (e.g. with `qpdf --decrypt`) before importing."
        )


def _extract_raw_text(page) -> str:
    try:
        return page.extract_text(extraction_mode="layout") or ""
    except TypeError:
        # Older pypdf versions without the extraction_mode parameter.
        return page.extract_text() or ""


def _extract_page_lines(reader) -> list[list[str]]:
    pages: list[list[str]] = []
    for page in reader.pages:
        raw = _extract_raw_text(page)
        cleaned = [unicodedata.normalize("NFKC", re.sub(r"[ \t]+", " ", line)).strip() for line in raw.splitlines()]
        pages.append([line for line in cleaned if line])
    return pages


def _detect_chapter_starts(pages: list[list[str]]) -> dict[int, int]:
    chapter_to_page: dict[int, int] = {}
    for idx, lines in enumerate(pages):
        if not lines:
            continue
        match = _CHAPTER_HEADER_RE.match(lines[0])
        if not match:
            continue
        num_str = next((g for g in match.groups() if g), None)
        if num_str:
            chapter_to_page.setdefault(int(num_str), idx)
    return chapter_to_page


def _chapter_title(first_line: str, fallback: str) -> str:
    match = _CHAPTER_HEADER_RE.match(first_line)
    if match:
        return first_line[match.end() :].strip(" :.-") or fallback
    return first_line or fallback


def _pdf_title(reader, fallback: str) -> str:
    try:
        title = reader.metadata.title if reader.metadata else None
    except Exception:
        title = None
    if title:
        return unicodedata.normalize("NFKC", re.sub(r"\s+", " ", title)).strip()
    return fallback


def parse(path: Path) -> Book:
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
    except PdfReadError as exc:
        raise ValueError(f"'{path}' is not a valid/readable PDF: {exc}") from exc

    _ensure_decrypted(reader, path)

    pages = _extract_page_lines(reader)
    pages = _strip_headers_and_footers(pages)
    pages = [_dehyphenate_lines(lines) for lines in pages]

    if not any(pages):
        raise ValueError(
            f"No extractable text found in '{path}'. This PDF may be scanned/image-only, "
            "which requires OCR — not supported yet."
        )

    book_title = _pdf_title(reader, fallback=path.stem)
    chapter_to_page = _detect_chapter_starts(pages)

    if not chapter_to_page:
        all_lines = [line for lines in pages for line in lines]
        paragraphs = [Paragraph(text=text, index=i) for i, text in enumerate(_group_into_paragraphs(all_lines))]
        return Book(title=book_title, chapters=[Chapter(title=book_title, index=0, paragraphs=paragraphs)])

    ordered = sorted(chapter_to_page.items())
    chapters: list[Chapter] = []
    for position, (_chapter_num, start_page) in enumerate(ordered):
        stop_page = ordered[position + 1][1] if position + 1 < len(ordered) else len(pages)
        chapter_lines = [line for lines in pages[start_page:stop_page] for line in lines]
        title = _chapter_title(pages[start_page][0], fallback=book_title) if pages[start_page] else book_title
        paragraphs = [Paragraph(text=text, index=i) for i, text in enumerate(_group_into_paragraphs(chapter_lines))]
        chapters.append(Chapter(title=title, index=position, paragraphs=paragraphs))

    return Book(title=book_title, chapters=chapters)
