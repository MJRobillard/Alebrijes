"""Plain text parsing: source file -> `alebrijes.ingest.Book`.

Plain `.txt` has no structural markup to lean on, so this is deliberately
simpler than the epub/pdf modules:

- Paragraphs are split on blank lines when the file has them (the common
  convention, and the author's actual intent when present).
- A file with no blank lines at all (a single wall of text — common for
  scraped/copy-pasted content) falls back to sentence-count-based reflow,
  ported from pocket-tts-agent's `examples/reflow_txt_paragraphs.py`, so TTS
  isn't handed one enormous unbroken paragraph.
- Chapter detection reuses the same "Chapter N" / "N:" heading heuristic as
  `alebrijes.ingest.epub`/`.pdf`, restricted to short blocks so an ordinary
  sentence that happens to start with a number isn't mistaken for a heading.
- Encoding: tries UTF-8 first (with BOM handling), falls back to latin-1
  (which never fails to decode, so this is a "best effort" rather than a
  guarantee of correctness for genuinely mixed/unknown encodings).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from alebrijes.ingest import Book, Chapter, Paragraph

# Arabic or Roman chapter numbers -- Project Gutenberg texts overwhelmingly
# use "CHAPTER I." (found by evals/attribution: all 28 PDNC novels parsed as
# a single chapter before Roman numerals were accepted).
_CHAPTER_HEADER_RE = re.compile(r"^\s*(?:chapter|chap\.?)\s*(\d+|[ivxlcdm]+)\b|^\s*(\d+)\s*[:.]\s+", re.IGNORECASE)
_MAX_HEADING_LENGTH = 60
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])(?:[\]\)\"]+)?\s+")
_MIN_SENTENCES_PER_PARAGRAPH = 5
_MAX_SENTENCES_PER_PARAGRAPH = 7


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _clean_block(block: str) -> str:
    return unicodedata.normalize("NFKC", re.sub(r"\s+", " ", block)).strip()


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(normalized) if part.strip()]


def _group_sentences(sentences: list[str], min_sentences: int, max_sentences: int) -> list[str]:
    groups: list[list[str]] = []
    index = 0
    total = len(sentences)
    while index < total:
        remaining = total - index
        if remaining <= max_sentences:
            if groups and remaining < min_sentences:
                groups[-1].extend(sentences[index:])
            else:
                groups.append(sentences[index:])
            break
        group_size = max_sentences
        remainder_after_group = remaining - group_size
        if 0 < remainder_after_group < min_sentences:
            group_size -= min_sentences - remainder_after_group
        group_size = max(min_sentences, min(max_sentences, group_size))
        groups.append(sentences[index : index + group_size])
        index += group_size
    return [" ".join(group).strip() for group in groups if group]


def _blocks_to_paragraph_texts(blocks: list[str]) -> list[str]:
    """Reflows an over-long single block via sentence grouping; leaves normal
    paragraph-sized blocks untouched."""
    texts: list[str] = []
    for block in blocks:
        sentence_count = len(_split_sentences(block))
        if len(blocks) == 1 and sentence_count > _MAX_SENTENCES_PER_PARAGRAPH:
            texts.extend(_group_sentences(_split_sentences(block), _MIN_SENTENCES_PER_PARAGRAPH, _MAX_SENTENCES_PER_PARAGRAPH))
        else:
            texts.append(block)
    return texts


def _is_heading(block: str) -> bool:
    return len(block) <= _MAX_HEADING_LENGTH and bool(_CHAPTER_HEADER_RE.match(block))


def parse(path: Path) -> Book:
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    raw_blocks = [b for b in re.split(r"\n\s*\n", _read_text(path)) if b.strip()]
    blocks = [_clean_block(b) for b in raw_blocks]
    blocks = _blocks_to_paragraph_texts(blocks)

    if not blocks:
        raise ValueError(f"No extractable text found in '{path}'.")

    book_title = path.stem
    heading_indices = [i for i, b in enumerate(blocks) if _is_heading(b)]

    if not heading_indices:
        paragraphs = [Paragraph(text=b, index=i) for i, b in enumerate(blocks)]
        return Book(title=book_title, chapters=[Chapter(title=book_title, index=0, paragraphs=paragraphs)])

    chapters: list[Chapter] = []
    for position, start in enumerate(heading_indices):
        stop = heading_indices[position + 1] if position + 1 < len(heading_indices) else len(blocks)
        match = _CHAPTER_HEADER_RE.match(blocks[start])
        title = blocks[start][match.end() :].strip(" :.-") or blocks[start]
        paragraphs = [Paragraph(text=b, index=i) for i, b in enumerate(blocks[start:stop])]
        chapters.append(Chapter(title=title, index=position, paragraphs=paragraphs))

    return Book(title=book_title, chapters=chapters)
