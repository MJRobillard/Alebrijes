from __future__ import annotations

from pathlib import Path

from alebrijes.segment import SpanKind
from alebrijes.transcript import ChapterMark, SpanTiming, Transcript, load, save


def _transcript() -> Transcript:
    return Transcript(
        book_title="Fixture Book",
        chapters=[ChapterMark(index=0, title="Chapter One", start_ms=0, end_ms=5000)],
        spans=[
            SpanTiming(
                span_id="0.0.0",
                chapter_index=0,
                paragraph_index=0,
                character="Narrator",
                kind=SpanKind.NARRATION,
                text="It was a dark and stormy night.",
                start_ms=0,
                end_ms=2500,
            ),
            SpanTiming(
                span_id="0.0.1",
                chapter_index=0,
                paragraph_index=0,
                character="Colin",
                kind=SpanKind.DIALOGUE,
                text="Let's take another path.",
                start_ms=2500,
                end_ms=5000,
            ),
        ],
    )


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    transcript = _transcript()
    path = tmp_path / "book.transcript.json"

    save(transcript, path)
    loaded = load(path)

    assert loaded == transcript


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    transcript = _transcript()
    path = tmp_path / "nested" / "book.transcript.json"

    save(transcript, path)

    assert path.exists()


def test_load_restores_span_kind_enum(tmp_path: Path) -> None:
    path = tmp_path / "book.transcript.json"
    save(_transcript(), path)

    loaded = load(path)

    assert loaded.spans[0].kind is SpanKind.NARRATION
    assert loaded.spans[1].kind is SpanKind.DIALOGUE


def test_quiz_checkpoints_round_trip(tmp_path: Path) -> None:
    transcript = _transcript()
    transcript.quiz_checkpoints = {"0.0.1": "q1"}
    path = tmp_path / "book.transcript.json"

    save(transcript, path)
    loaded = load(path)

    assert loaded.quiz_checkpoints == {"0.0.1": "q1"}


def test_quiz_checkpoints_defaults_to_empty_dict_when_absent_from_json(tmp_path: Path) -> None:
    path = tmp_path / "book.transcript.json"
    path.write_text('{"book_title": "Old Sidecar", "chapters": [], "spans": []}')

    loaded = load(path)

    assert loaded.quiz_checkpoints == {}
