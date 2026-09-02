from __future__ import annotations

from pathlib import Path

import numpy as np

from alebrijes.rag import RagEntry, RagIndex, load, save, search
from alebrijes.rag.build import build_rag_index
from alebrijes.segment import SpanKind
from alebrijes.transcript import SpanTiming, Transcript


def _entry(span_id: str, seq: int, text: str) -> RagEntry:
    return RagEntry(span_id=span_id, chapter_index=0, paragraph_index=seq, seq=seq, text=text)


def _index_with_orthogonal_vectors() -> RagIndex:
    # Three 2D unit vectors at 0/90/180 degrees so similarity ranking is
    # unambiguous: querying along the first axis should rank entry "a"
    # highest, "c" (opposite direction) lowest.
    entries = [_entry("a", 0, "alpha"), _entry("b", 1, "beta"), _entry("c", 2, "gamma")]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    return RagIndex(entries=entries, vectors=vectors)


def test_search_ranks_by_cosine_similarity() -> None:
    index = _index_with_orthogonal_vectors()

    hits = search(index, np.array([1.0, 0.0]), max_seq=2, top_k=3)

    assert [hit.entry.span_id for hit in hits] == ["a", "b", "c"]
    assert hits[0].score == pytest_approx(1.0)
    assert hits[-1].score == pytest_approx(-1.0)


def test_search_never_returns_entries_past_max_seq() -> None:
    index = _index_with_orthogonal_vectors()

    hits = search(index, np.array([1.0, 0.0]), max_seq=0, top_k=3)

    assert [hit.entry.span_id for hit in hits] == ["a"]


def test_search_respects_top_k() -> None:
    index = _index_with_orthogonal_vectors()

    hits = search(index, np.array([1.0, 0.0]), max_seq=2, top_k=1)

    assert len(hits) == 1
    assert hits[0].entry.span_id == "a"


def test_search_on_empty_index_returns_no_hits() -> None:
    index = RagIndex(entries=[], vectors=np.zeros((0, 2), dtype=np.float32))

    hits = search(index, np.array([1.0, 0.0]), max_seq=100)

    assert hits == []


def test_save_then_load_round_trips_entries_and_vectors(tmp_path: Path) -> None:
    index = _index_with_orthogonal_vectors()
    npz_path = tmp_path / "book.rag.npz"
    json_path = tmp_path / "book.rag.json"

    save(index, npz_path, json_path)
    loaded = load(npz_path, json_path)

    assert loaded.entries == index.entries
    assert np.allclose(loaded.vectors, index.vectors)


def _transcript_with_spans(texts: list[str]) -> Transcript:
    spans = [
        SpanTiming(
            span_id=f"0.{i}.0",
            chapter_index=0,
            paragraph_index=i,
            character="Narrator",
            kind=SpanKind.NARRATION,
            text=text,
            start_ms=i * 1000,
            end_ms=(i + 1) * 1000,
        )
        for i, text in enumerate(texts)
    ]
    return Transcript(book_title="Fixture", chapters=[], spans=spans)


def test_build_rag_index_embeds_each_span_and_assigns_sequential_seq() -> None:
    transcript = _transcript_with_spans(["first span", "second span", "third span"])
    calls = []

    def fake_embed(text: str) -> np.ndarray:
        calls.append(text)
        return np.array([len(text), 1.0], dtype=np.float32)

    index = build_rag_index(transcript, fake_embed)

    assert calls == ["first span", "second span", "third span"]
    assert [e.seq for e in index.entries] == [0, 1, 2]
    assert [e.span_id for e in index.entries] == ["0.0.0", "0.1.0", "0.2.0"]
    assert index.vectors.shape == (3, 2)


def test_build_rag_index_reports_progress() -> None:
    transcript = _transcript_with_spans(["a", "b"])
    progress_calls = []

    build_rag_index(
        transcript,
        lambda text: np.array([1.0, 0.0], dtype=np.float32),
        on_progress=lambda done, total: progress_calls.append((done, total)),
    )

    assert progress_calls == [(1, 2), (2, 2)]


def test_build_rag_index_on_empty_transcript_returns_empty_index() -> None:
    transcript = _transcript_with_spans([])

    index = build_rag_index(transcript, lambda text: np.array([1.0]))

    assert index.entries == []
    assert index.vectors.shape == (0, 0)


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, abs=1e-5)
