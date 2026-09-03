from __future__ import annotations

from pathlib import Path

import pytest

from alebrijes.ingest import parse_book


def test_txt_written_in_script_format_is_routed_to_script_parser(tmp_path: Path) -> None:
    path = tmp_path / "book.txt"
    path.write_text(
        "FELICITY: Except Bayesian Optimization actually converges.\n"
        "SARA: So if I were a hyperparameter, which one would I be?\n",
        encoding="utf-8",
    )

    book = parse_book(path)

    assert book.source_format == "script"
    assert [p.text for p in book.chapters[0].paragraphs] == [
        "FELICITY: Except Bayesian Optimization actually converges.",
        "SARA: So if I were a hyperparameter, which one would I be?",
    ]


def test_ordinary_prose_txt_is_routed_to_txt_parser(tmp_path: Path) -> None:
    path = tmp_path / "book.txt"
    path.write_text(
        'The room was quiet. "Hello," Colin said. She walked away slowly, thinking about tomorrow.\n',
        encoding="utf-8",
    )

    book = parse_book(path)

    assert book.source_format == "prose"


def test_script_extension_still_dispatches_to_script_parser(tmp_path: Path) -> None:
    path = tmp_path / "book.script"
    path.write_text("NARRATOR: The room was quiet.\n", encoding="utf-8")

    book = parse_book(path)

    assert book.source_format == "script"


def test_unsupported_extension_raises() -> None:
    with pytest.raises(NotImplementedError):
        parse_book(Path("book.docx"))
