from __future__ import annotations

from pathlib import Path

import pytest

from alebrijes.ingest.script import parse


def test_parse_sets_script_source_format(tmp_path: Path) -> None:
    path = tmp_path / "book.script"
    path.write_text("NARRATOR: The room was quiet.\n", encoding="utf-8")

    book = parse(path)

    assert book.source_format == "script"


def test_parse_without_chapter_heading_makes_one_implicit_chapter(tmp_path: Path) -> None:
    path = tmp_path / "book.script"
    path.write_text(
        "NARRATOR: The room was quiet.\nCOLIN: Let's take another path.\n", encoding="utf-8"
    )

    book = parse(path)

    assert len(book.chapters) == 1
    assert book.chapters[0].title == "Chapter 1"
    assert [p.text for p in book.chapters[0].paragraphs] == [
        "NARRATOR: The room was quiet.",
        "COLIN: Let's take another path.",
    ]


def test_parse_splits_chapters_on_markdown_heading(tmp_path: Path) -> None:
    path = tmp_path / "book.script"
    path.write_text(
        "# Chapter One\n"
        "NARRATOR: It began.\n"
        "# Chapter Two\n"
        "NARRATOR: It continued.\n"
        "SABLE: You shouldn't be here.\n",
        encoding="utf-8",
    )

    book = parse(path)

    assert [c.title for c in book.chapters] == ["Chapter One", "Chapter Two"]
    assert [p.text for p in book.chapters[0].paragraphs] == ["NARRATOR: It began."]
    assert [p.text for p in book.chapters[1].paragraphs] == [
        "NARRATOR: It continued.",
        "SABLE: You shouldn't be here.",
    ]
    assert book.chapters[0].index == 0
    assert book.chapters[1].index == 1


def test_parse_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "book.script"
    path.write_text("NARRATOR: One.\n\n\nNARRATOR: Two.\n", encoding="utf-8")

    book = parse(path)

    assert [p.text for p in book.chapters[0].paragraphs] == ["NARRATOR: One.", "NARRATOR: Two."]


def test_parse_rejects_a_line_with_no_speaker_prefix(tmp_path: Path) -> None:
    path = tmp_path / "book.script"
    path.write_text("NARRATOR: Fine.\nThis line has no speaker.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="NAME: text"):
        parse(path)


def test_parse_rejects_lowercase_speaker_name(tmp_path: Path) -> None:
    path = tmp_path / "book.script"
    path.write_text("Colin: lowercase name should not match.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="NAME: text"):
        parse(path)


def test_parse_does_not_mistake_a_colon_inside_dialogue_for_a_speaker_prefix(tmp_path: Path) -> None:
    # "Wait:" isn't ALL CAPS, so this must fail clearly rather than silently
    # treating "Wait" as a speaker name.
    path = tmp_path / "book.script"
    path.write_text('COLIN: "Wait: is that true?" he asked.\n', encoding="utf-8")

    book = parse(path)

    assert book.chapters[0].paragraphs[0].text == 'COLIN: "Wait: is that true?" he asked.'


def test_parse_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse(tmp_path / "missing.script")


def test_parse_raises_on_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "book.script"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No script lines"):
        parse(path)


def test_parse_uses_file_stem_as_book_title(tmp_path: Path) -> None:
    path = tmp_path / "My Great Book.script"
    path.write_text("NARRATOR: Hello.\n", encoding="utf-8")

    book = parse(path)

    assert book.title == "My Great Book"
