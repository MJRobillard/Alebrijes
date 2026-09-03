from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from alebrijes.cast import CastFile
from alebrijes.ingest import Book, Chapter
from alebrijes.session.buffer import ChapterAudio, LookaheadRenderer


def _book(num_chapters: int) -> Book:
    return Book(
        title="Fixture",
        chapters=[Chapter(title=f"Ch{i}", index=i, paragraphs=[]) for i in range(num_chapters)],
    )


class _FakeAgent:
    sample_rate = 1000


def _fake_resolve(chapter, cast_file, *, script_format=False):
    return [f"span-for-{chapter.index}"]


def _fake_synthesize(voiced_spans, voice_states, agent, pause_ms=250):
    return np.array([1, 2, 3], dtype=np.int16), [f"timing-{voiced_spans[0]}"]


def _wait_until(predicate, timeout=3.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_renders_ahead_up_to_lookahead_limit_then_blocks() -> None:
    renderer = LookaheadRenderer(
        _book(5),
        CastFile(book_title="Fixture"),
        voice_states={},
        agent=_FakeAgent(),
        lookahead_chapters=1,
        resolve_chapter_spans_fn=_fake_resolve,
        synthesize_chapter_fn=_fake_synthesize,
    )
    try:
        renderer.start()
        assert _wait_until(lambda: renderer.chapters_done == 1)
        # Give it a beat to (wrongly) render further if the gate isn't working.
        time.sleep(0.2)
        assert renderer.chapters_done == 1
    finally:
        renderer.stop()


def test_mark_consumed_unblocks_further_rendering() -> None:
    renderer = LookaheadRenderer(
        _book(5),
        CastFile(book_title="Fixture"),
        voice_states={},
        agent=_FakeAgent(),
        lookahead_chapters=1,
        resolve_chapter_spans_fn=_fake_resolve,
        synthesize_chapter_fn=_fake_synthesize,
    )
    try:
        renderer.start()
        assert _wait_until(lambda: renderer.chapters_done == 1)

        renderer.mark_consumed(0)
        assert _wait_until(lambda: renderer.chapters_done == 2)

        renderer.mark_consumed(1)
        assert _wait_until(lambda: renderer.chapters_done == 3)
    finally:
        renderer.stop()


def test_on_chapter_ready_receives_chapters_in_order() -> None:
    received: list[ChapterAudio] = []
    renderer = LookaheadRenderer(
        _book(3),
        CastFile(book_title="Fixture"),
        voice_states={},
        agent=_FakeAgent(),
        lookahead_chapters=5,  # no gating -- render the whole (small) book
        on_chapter_ready=received.append,
        resolve_chapter_spans_fn=_fake_resolve,
        synthesize_chapter_fn=_fake_synthesize,
    )
    try:
        renderer.start()
        assert _wait_until(lambda: len(received) == 3)
        assert [c.chapter_index for c in received] == [0, 1, 2]
        assert received[0].title == "Ch0"
        assert received[0].sample_rate == 1000
        assert received[0].audio.tolist() == [1, 2, 3]
        assert received[0].timings == ["timing-span-for-0"]
    finally:
        renderer.stop()


def test_stop_halts_a_blocked_renderer_promptly() -> None:
    renderer = LookaheadRenderer(
        _book(5),
        CastFile(book_title="Fixture"),
        voice_states={},
        agent=_FakeAgent(),
        lookahead_chapters=1,
        resolve_chapter_spans_fn=_fake_resolve,
        synthesize_chapter_fn=_fake_synthesize,
    )
    renderer.start()
    assert _wait_until(lambda: renderer.chapters_done == 1)

    start = time.time()
    renderer.stop()
    assert time.time() - start < 2.0


def test_synthesis_error_is_captured_not_raised_in_background_thread() -> None:
    def _raising_synthesize(voiced_spans, voice_states, agent, pause_ms=250):
        raise RuntimeError("synthesis exploded")

    renderer = LookaheadRenderer(
        _book(2),
        CastFile(book_title="Fixture"),
        voice_states={},
        agent=_FakeAgent(),
        lookahead_chapters=5,
        resolve_chapter_spans_fn=_fake_resolve,
        synthesize_chapter_fn=_raising_synthesize,
    )
    try:
        renderer.start()
        assert _wait_until(lambda: renderer.error is not None)
        assert "synthesis exploded" in str(renderer.error)
    finally:
        renderer.stop()


def test_synthesis_error_invokes_on_error_with_failing_chapter_index() -> None:
    def _raising_synthesize(voiced_spans, voice_states, agent, pause_ms=250):
        raise RuntimeError("synthesis exploded")

    errors: list[tuple[int, Exception]] = []
    renderer = LookaheadRenderer(
        _book(2),
        CastFile(book_title="Fixture"),
        voice_states={},
        agent=_FakeAgent(),
        lookahead_chapters=5,
        resolve_chapter_spans_fn=_fake_resolve,
        synthesize_chapter_fn=_raising_synthesize,
        on_error=lambda chapter_index, exc: errors.append((chapter_index, exc)),
    )
    try:
        renderer.start()
        assert _wait_until(lambda: len(errors) == 1)
        chapter_index, exc = errors[0]
        assert chapter_index == 0
        assert "synthesis exploded" in str(exc)
    finally:
        renderer.stop()
