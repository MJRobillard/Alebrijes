from __future__ import annotations

from pathlib import Path

import pytest

from alebrijes.cast import load
from alebrijes.cli import main
from epub_fixtures import build_epub


def test_cast_command_writes_cast_file_with_detected_characters(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    epub_path = tmp_path / "book.epub"
    build_epub(
        epub_path,
        "Fixture Book",
        {"ch1.xhtml": '<p>Chapter 1</p><p>Colin said, "Let\'s take another path."</p>'},
    )
    out_path = tmp_path / "book.cast.yaml"

    exit_code = main(["cast", str(epub_path), "--out", str(out_path)])

    assert exit_code == 0
    assert out_path.exists()
    cast_file = load(out_path)
    assert cast_file.voice_for("Narrator")
    assert cast_file.voice_for("Colin")
    assert "Wrote cast file" in capsys.readouterr().out


def test_cast_command_defaults_output_path_next_to_book(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    build_epub(epub_path, "Fixture Book", {"ch1.xhtml": "<p>Just narration.</p>"})

    main(["cast", str(epub_path)])

    assert (tmp_path / "book.cast.yaml").exists()


def test_cast_command_rejects_unsupported_format(tmp_path: Path) -> None:
    unsupported = tmp_path / "book.docx"
    unsupported.write_text("Some text.")

    with pytest.raises(NotImplementedError, match="docx"):
        main(["cast", str(unsupported)])


def test_cast_command_supports_txt_books(tmp_path: Path) -> None:
    txt_path = tmp_path / "book.txt"
    txt_path.write_text('Colin said, "Let\'s take another path through the woods."\n', encoding="utf-8")

    exit_code = main(["cast", str(txt_path)])

    assert exit_code == 0
    cast_file = load(tmp_path / "book.cast.yaml")
    assert cast_file.voice_for("Colin")
