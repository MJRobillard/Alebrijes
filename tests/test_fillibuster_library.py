from __future__ import annotations

from pathlib import Path

from alebrijes.session.fillibuster_library import (
    list_fillibusters,
    load_fillibuster_sentences,
    load_fillibuster_text,
)


def _write(directory: Path, fillibuster_id: str, text: str, *, title: str = "", description: str = "") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{fillibuster_id}.txt").write_text(text, encoding="utf-8")
    (directory / f"{fillibuster_id}.yaml").write_text(
        f"title: {title or fillibuster_id}\ndescription: {description}\n", encoding="utf-8"
    )


def test_list_fillibusters_returns_empty_for_missing_directory(tmp_path: Path) -> None:
    assert list_fillibusters(tmp_path / "does-not-exist") == []


def test_list_fillibusters_returns_empty_for_empty_directory(tmp_path: Path) -> None:
    assert list_fillibusters(tmp_path) == []


def test_list_fillibusters_reads_metadata_sorted_by_id(tmp_path: Path) -> None:
    _write(tmp_path, "b", "B text.", title="B Title", description="About B")
    _write(tmp_path, "a", "A text.", title="A Title", description="About A")

    metas = list_fillibusters(tmp_path)

    assert [m.id for m in metas] == ["a", "b"]
    assert metas[0].title == "A Title"
    assert metas[0].description == "About A"


def test_list_fillibusters_skips_a_txt_with_no_yaml_sibling(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "orphan.txt").write_text("No metadata for this one.", encoding="utf-8")
    _write(tmp_path, "a", "A text.")

    metas = list_fillibusters(tmp_path)

    assert [m.id for m in metas] == ["a"]


def test_list_fillibusters_defaults_title_to_id_when_yaml_omits_it(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "a.txt").write_text("A text.", encoding="utf-8")
    (tmp_path / "a.yaml").write_text("description: just a description\n", encoding="utf-8")

    metas = list_fillibusters(tmp_path)

    assert metas[0].title == "a"
    assert metas[0].description == "just a description"


def test_load_fillibuster_text_strips_surrounding_whitespace(tmp_path: Path) -> None:
    _write(tmp_path, "a", "  Some text.  \n\n")

    assert load_fillibuster_text(tmp_path, "a") == "Some text."


def test_load_fillibuster_text_raises_for_unknown_id(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        load_fillibuster_text(tmp_path, "missing")


def test_load_fillibuster_sentences_splits_each_script_and_concatenates_in_order(tmp_path: Path) -> None:
    _write(tmp_path, "a", "First A sentence. Second A sentence.")
    _write(tmp_path, "b", "Only B sentence.")

    sentences = load_fillibuster_sentences(tmp_path, ["a", "b"])

    assert sentences == ["First A sentence.", "Second A sentence.", "Only B sentence."]


def test_load_fillibuster_sentences_respects_requested_order_not_directory_order(tmp_path: Path) -> None:
    _write(tmp_path, "a", "A sentence.")
    _write(tmp_path, "b", "B sentence.")

    sentences = load_fillibuster_sentences(tmp_path, ["b", "a"])

    assert sentences == ["B sentence.", "A sentence."]


def test_load_fillibuster_sentences_empty_ids_returns_empty_list(tmp_path: Path) -> None:
    assert load_fillibuster_sentences(tmp_path, []) == []


def test_load_fillibuster_sentences_includes_trailing_text_with_no_terminal_punctuation(tmp_path: Path) -> None:
    _write(tmp_path, "a", "A full sentence. And a trailing fragment with no period")

    sentences = load_fillibuster_sentences(tmp_path, ["a"])

    assert sentences == ["A full sentence.", "And a trailing fragment with no period"]
