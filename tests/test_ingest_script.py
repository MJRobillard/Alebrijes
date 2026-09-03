from __future__ import annotations

from pathlib import Path

import pytest

from alebrijes.ingest.script import looks_like_script, parse


def test_looks_like_script_true_for_all_name_colon_lines() -> None:
    assert looks_like_script("FELICITY: Hello.\nSARA: Hi yourself.\n")


def test_looks_like_script_true_with_chapter_and_quiz_markers() -> None:
    assert looks_like_script("# Chapter One\nNARRATOR: It began.\nQUIZ: q1\nCOLIN: Let's go.\n")


def test_looks_like_script_false_for_ordinary_prose() -> None:
    assert not looks_like_script('The room was quiet. "Hello," Colin said.\n')


def test_looks_like_script_false_for_mixed_content() -> None:
    assert not looks_like_script("FELICITY: Hello.\nThen everyone went home.\n")


def test_looks_like_script_false_when_empty() -> None:
    assert not looks_like_script("\n\n")


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


def test_parse_attaches_quiz_marker_to_preceding_line(tmp_path: Path) -> None:
    path = tmp_path / "book.script"
    path.write_text(
        "NARRATOR: Roles delegate access.\nQUIZ: q1\nNOVA: Let's continue.\n", encoding="utf-8"
    )

    book = parse(path)

    assert book.quiz_checkpoints == {"0.0.0": "q1"}
    # The QUIZ line itself produces no spoken paragraph/span.
    assert [p.text for p in book.chapters[0].paragraphs] == [
        "NARRATOR: Roles delegate access.",
        "NOVA: Let's continue.",
    ]


def test_parse_quiz_marker_uses_correct_chapter_index(tmp_path: Path) -> None:
    path = tmp_path / "book.script"
    path.write_text(
        "# Chapter One\nNARRATOR: First.\n# Chapter Two\nNARRATOR: Second.\nQUIZ: q2\n",
        encoding="utf-8",
    )

    book = parse(path)

    assert book.quiz_checkpoints == {"1.0.0": "q2"}


def test_parse_multiple_quiz_markers(tmp_path: Path) -> None:
    path = tmp_path / "book.script"
    path.write_text(
        "NARRATOR: First.\nQUIZ: q1\nNARRATOR: Second.\nQUIZ: q2\n", encoding="utf-8"
    )

    book = parse(path)

    assert book.quiz_checkpoints == {"0.0.0": "q1", "0.1.0": "q2"}


def test_parse_quiz_marker_with_no_preceding_line_in_chapter_raises(tmp_path: Path) -> None:
    path = tmp_path / "book.script"
    path.write_text("QUIZ: q1\nNARRATOR: Hello.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="QUIZ marker has no preceding line"):
        parse(path)


def test_parse_does_not_treat_quiz_as_a_valid_character_name(tmp_path: Path) -> None:
    # A real character literally named QUIZFOO isn't shadowed by the
    # reserved QUIZ: directive (different token, "QUIZ:" != "QUIZFOO:").
    path = tmp_path / "book.script"
    path.write_text("QUIZFOO: This is fine.\n", encoding="utf-8")

    book = parse(path)

    assert book.chapters[0].paragraphs[0].text == "QUIZFOO: This is fine."
    assert book.quiz_checkpoints == {}
