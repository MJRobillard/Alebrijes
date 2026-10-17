from __future__ import annotations

from pathlib import Path

import pytest

from alebrijes.ingest.txt import parse


def test_parse_splits_on_blank_lines_when_present(tmp_path: Path) -> None:
    txt_path = tmp_path / "book.txt"
    txt_path.write_text("First paragraph here.\n\nSecond paragraph here.\n", encoding="utf-8")

    book = parse(txt_path)

    assert len(book.chapters) == 1
    assert [p.text for p in book.chapters[0].paragraphs] == [
        "First paragraph here.",
        "Second paragraph here.",
    ]


def test_parse_detects_chapter_headings(tmp_path: Path) -> None:
    txt_path = tmp_path / "book.txt"
    txt_path.write_text(
        "Chapter 1: The Beginning\n\nSome narration.\n\nChapter 2: The Middle\n\nMore narration.\n",
        encoding="utf-8",
    )

    book = parse(txt_path)

    assert len(book.chapters) == 2
    assert book.chapters[0].title == "The Beginning"
    assert book.chapters[1].title == "The Middle"


def test_parse_reflows_single_wall_of_text_by_sentence_count(tmp_path: Path) -> None:
    sentences = " ".join(f"This is sentence number {i}." for i in range(1, 16))
    txt_path = tmp_path / "book.txt"
    txt_path.write_text(sentences, encoding="utf-8")

    book = parse(txt_path)

    paragraphs = book.chapters[0].paragraphs
    assert len(paragraphs) > 1
    # Every paragraph has at least the minimum; the last one may absorb a
    # too-small remainder and run over the max rather than leaving a tiny
    # trailing paragraph — the ported grouping algorithm's actual behavior.
    for paragraph in paragraphs:
        sentence_count = paragraph.text.count(". ") + 1
        assert sentence_count >= 5
    assert all(paragraph.text.count(". ") + 1 <= 7 for paragraph in paragraphs[:-1])


def test_parse_does_not_reflow_normal_multi_paragraph_text(tmp_path: Path) -> None:
    txt_path = tmp_path / "book.txt"
    txt_path.write_text("One sentence paragraph.\n\nAnother short one.\n\nAnd a third.\n", encoding="utf-8")

    book = parse(txt_path)

    assert len(book.chapters[0].paragraphs) == 3


def test_parse_falls_back_to_latin1_on_invalid_utf8(tmp_path: Path) -> None:
    txt_path = tmp_path / "book.txt"
    txt_path.write_bytes("Caf\xe9 with invalid utf-8 byte.".encode("latin-1"))

    book = parse(txt_path)

    assert "Café" in book.chapters[0].paragraphs[0].text


def test_parse_raises_clear_error_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        parse(tmp_path / "does_not_exist.txt")


def test_parse_raises_clear_error_for_empty_file(tmp_path: Path) -> None:
    txt_path = tmp_path / "book.txt"
    txt_path.write_text("   \n\n   ", encoding="utf-8")

    with pytest.raises(ValueError, match="No extractable text"):
        parse(txt_path)


def test_parse_detects_roman_numeral_chapter_headings(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text("CHAPTER I.\n\nSome narration.\n\nCHAPTER II.\n\nMore narration.\n", encoding="utf-8")

    book = parse(book_path)

    assert len(book.chapters) == 2
    assert [p.text for p in book.chapters[1].paragraphs] == ["CHAPTER II.", "More narration."]
