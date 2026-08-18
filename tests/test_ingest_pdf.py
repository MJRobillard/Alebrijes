from __future__ import annotations

from pathlib import Path

import pytest

from alebrijes.ingest.pdf import (
    _dehyphenate_lines,
    _group_into_paragraphs,
    _strip_headers_and_footers,
    parse,
)
from pdf_fixtures import build_pdf


def test_strip_headers_and_footers_drops_repeated_lines_and_page_markers() -> None:
    pages = [
        ["Running Title", "First page content.", "1"],
        ["Running Title", "Second page content.", "2"],
    ]

    cleaned = _strip_headers_and_footers(pages)

    assert cleaned == [["First page content."], ["Second page content."]]


def test_dehyphenate_lines_joins_wrapped_word() -> None:
    lines = ["This is an exam-", "ple sentence."]

    joined = _dehyphenate_lines(lines)

    assert joined == ["This is an example sentence."]


def test_dehyphenate_lines_does_not_touch_non_hyphenated_lines() -> None:
    lines = ["A plain line.", "Another plain line."]

    assert _dehyphenate_lines(lines) == lines


def test_group_into_paragraphs_splits_on_terminal_punctuation() -> None:
    lines = ["This sentence wraps", "across two lines.", "A second paragraph."]

    paragraphs = _group_into_paragraphs(lines)

    assert paragraphs == ["This sentence wraps across two lines.", "A second paragraph."]


def test_parse_extracts_chapter_titles_and_paragraphs(tmp_path: Path) -> None:
    pdf_path = tmp_path / "book.pdf"
    build_pdf(
        pdf_path,
        [
            ["Chapter 1: The Beginning", "The story starts here."],
            ["Chapter 2: The Middle", "The story continues."],
        ],
    )

    book = parse(pdf_path)

    assert len(book.chapters) == 2
    assert book.chapters[0].title == "The Beginning"
    assert book.chapters[1].title == "The Middle"
    assert "starts here" in book.chapters[0].paragraphs[-1].text


def test_parse_falls_back_to_single_chapter_when_no_headers_found(tmp_path: Path) -> None:
    pdf_path = tmp_path / "book.pdf"
    build_pdf(pdf_path, [["Just some narration with no chapter headings anywhere."]])

    book = parse(pdf_path)

    assert len(book.chapters) == 1
    assert "narration" in book.chapters[0].paragraphs[0].text


def test_parse_strips_repeated_running_header_across_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "book.pdf"
    build_pdf(
        pdf_path,
        [
            ["My Book Title", "Chapter 1: Start", "First page body text."],
            ["My Book Title", "Second page body text."],
        ],
    )

    book = parse(pdf_path)

    all_text = " ".join(p.text for chapter in book.chapters for p in chapter.paragraphs)
    assert "My Book Title" not in all_text


def test_parse_raises_clear_error_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        parse(tmp_path / "does_not_exist.pdf")


def test_parse_decrypts_pdf_with_blank_password(tmp_path: Path) -> None:
    pdf_path = tmp_path / "book.pdf"
    build_pdf(pdf_path, [["Some protected but blank-password content."]], encrypt_user_password="")

    book = parse(pdf_path)

    assert "protected" in book.chapters[0].paragraphs[0].text


def test_parse_raises_clear_error_for_real_password_protected_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "book.pdf"
    build_pdf(pdf_path, [["Secret content."]], encrypt_user_password="hunter2")

    with pytest.raises(ValueError, match="password-protected"):
        parse(pdf_path)
