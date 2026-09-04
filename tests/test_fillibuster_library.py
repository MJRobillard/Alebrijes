from __future__ import annotations

from pathlib import Path

from alebrijes.session.fillibuster_library import list_fillibusters
from alebrijes.transcript import Transcript
from alebrijes.transcript import save as save_transcript


def _write(directory: Path, fillibuster_id: str, *, title: str = "") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{fillibuster_id}.m4b").write_bytes(b"not real audio, just a fixture placeholder")
    save_transcript(
        Transcript(book_title=title or fillibuster_id),
        directory / f"{fillibuster_id}.transcript.json",
    )


def test_list_fillibusters_returns_empty_for_missing_directory(tmp_path: Path) -> None:
    assert list_fillibusters(tmp_path / "does-not-exist") == []


def test_list_fillibusters_returns_empty_for_empty_directory(tmp_path: Path) -> None:
    assert list_fillibusters(tmp_path) == []


def test_list_fillibusters_reads_title_from_transcript_sorted_by_id(tmp_path: Path) -> None:
    _write(tmp_path, "b", title="B Title")
    _write(tmp_path, "a", title="A Title")

    metas = list_fillibusters(tmp_path)

    assert [m.id for m in metas] == ["a", "b"]
    assert metas[0].title == "A Title"
    assert metas[0].path == str(tmp_path / "a.m4b")


def test_list_fillibusters_skips_an_m4b_with_no_transcript_sidecar(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "orphan.m4b").write_bytes(b"no transcript sidecar for this one")
    _write(tmp_path, "a", title="A Title")

    metas = list_fillibusters(tmp_path)

    assert [m.id for m in metas] == ["a"]


def test_list_fillibusters_ignores_non_m4b_files(tmp_path: Path) -> None:
    _write(tmp_path, "a", title="A Title")
    (tmp_path / "notes.txt").write_text("not a fillibuster entry", encoding="utf-8")

    metas = list_fillibusters(tmp_path)

    assert [m.id for m in metas] == ["a"]
