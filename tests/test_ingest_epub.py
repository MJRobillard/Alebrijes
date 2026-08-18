from __future__ import annotations

from pathlib import Path

import pytest

from alebrijes.ingest.epub import parse
from epub_fixtures import build_epub as _build_epub


def test_parse_extracts_chapter_titles_and_paragraphs(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    _build_epub(
        epub_path,
        "Fixture Book",
        {
            "ch1.xhtml": "<p>Chapter 1: The Beginning</p><p>Colin said, \"Let's go.\"</p>",
            "ch2.xhtml": "<p>Chapter 2: The Middle</p><p>Audrey nodded.</p>",
        },
    )

    book = parse(epub_path)

    assert book.title == "Fixture Book"
    assert len(book.chapters) == 2
    assert book.chapters[0].title == "The Beginning"
    assert book.chapters[1].title == "The Middle"
    assert [p.text for p in book.chapters[0].paragraphs] == [
        "Chapter 1: The Beginning",
        'Colin said, "Let\'s go."',
    ]


def test_parse_falls_back_to_single_chapter_when_no_headers_found(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    _build_epub(
        epub_path,
        "Untitled Fixture",
        {"page1.xhtml": "<p>Just some narration with no chapter headings anywhere.</p>"},
    )

    book = parse(epub_path)

    assert len(book.chapters) == 1
    assert book.chapters[0].title == "Untitled Fixture"
    assert book.chapters[0].paragraphs[0].text == "Just some narration with no chapter headings anywhere."


def test_parse_strips_html_tags_and_collapses_whitespace(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    _build_epub(
        epub_path,
        "Fixture Book",
        {"ch1.xhtml": "<p>Chapter 1</p><p>Some   <b>bold</b>   text   with   spaces.</p>"},
    )

    book = parse(epub_path)

    assert book.chapters[0].paragraphs[1].text == "Some bold text with spaces."


def test_parse_raises_clear_error_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        parse(tmp_path / "does_not_exist.epub")


def test_parse_uses_image_alt_text_chapter_heading_fallback(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    _build_epub(
        epub_path,
        "Fixture Book",
        {
            "ch1.xhtml": '<img alt="1 Only the Dark"/><p>The story begins.</p>',
            "ch2.xhtml": '<img alt="2 A New Dawn"/><p>The story continues.</p>',
        },
    )

    book = parse(epub_path)

    assert len(book.chapters) == 2
    assert book.chapters[0].title == "Only the Dark"
    assert book.chapters[1].title == "A New Dawn"
