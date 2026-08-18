"""Structural tests for the pipeline skeleton itself.

These exist so Phase 1's "every stage importable, no circular deps" exit
criteria (see PHASES.md) has a concrete check, and so stubbed stages fail
loudly (NotImplementedError) rather than silently doing nothing.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_every_pipeline_stage_imports() -> None:
    import alebrijes.ingest  # noqa: F401
    import alebrijes.ingest.epub  # noqa: F401
    import alebrijes.segment  # noqa: F401
    import alebrijes.attribute  # noqa: F401
    import alebrijes.cast  # noqa: F401
    import alebrijes.synth  # noqa: F401
    import alebrijes.assemble  # noqa: F401
    import alebrijes.cli  # noqa: F401


def test_cli_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "alebrijes.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "render" in result.stdout


def test_unsupported_ingest_format_fails_loudly_not_silently(tmp_path) -> None:
    from alebrijes.cli import _ingest_book

    unsupported = tmp_path / "book.docx"
    unsupported.write_text("not a supported format")

    with pytest.raises(NotImplementedError, match="docx"):
        _ingest_book(unsupported)


def test_cast_file_round_trips(tmp_path) -> None:
    from alebrijes.cast import CastEntry, CastFile, load, save

    cast_file = CastFile(
        book_title="Fixture Book",
        entries=[CastEntry(character="Narrator", voice_state_path="voices/narrator.pt")],
    )
    path = tmp_path / "cast.yaml"

    save(cast_file, path)
    loaded = load(path)

    assert loaded.book_title == "Fixture Book"
    assert loaded.voice_for("Narrator") == "voices/narrator.pt"
