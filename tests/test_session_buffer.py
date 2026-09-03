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


def _fake_synthesize(voiced_spans, voice_states, agent, pause_ms=250, on_line_done=None):
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


def _fake_resolve_multi_span(num_spans_per_chapter: int):
    def _resolve(chapter, cast_file, *, script_format=False):
        return [f"span-{chapter.index}-{i}" for i in range(num_spans_per_chapter)]

    return _resolve


def test_first_chapter_with_many_spans_is_split_into_a_fast_start_head_and_tail() -> None:
    from alebrijes.session.buffer import FAST_START_HEAD_SPANS

    received: list[ChapterAudio] = []
    renderer = LookaheadRenderer(
        _book(2),
        CastFile(book_title="Fixture"),
        voice_states={},
        agent=_FakeAgent(),
        lookahead_chapters=5,
        on_chapter_ready=received.append,
        resolve_chapter_spans_fn=_fake_resolve_multi_span(FAST_START_HEAD_SPANS + 3),
        synthesize_chapter_fn=_fake_synthesize,
    )
    try:
        renderer.start()
        assert _wait_until(lambda: len(received) >= 3)  # head + tail of ch0, then all of ch1
        # Two on_chapter_ready calls for chapter 0 (head, then tail) before chapter 1's single call.
        assert [c.chapter_index for c in received[:3]] == [0, 0, 1]
        assert _wait_until(lambda: renderer.chapters_done == 2)  # still counted as 2 whole chapters
    finally:
        renderer.stop()


def test_chapter_with_few_spans_is_not_split() -> None:
    from alebrijes.session.buffer import FAST_START_HEAD_SPANS

    received: list[ChapterAudio] = []
    renderer = LookaheadRenderer(
        _book(1),
        CastFile(book_title="Fixture"),
        voice_states={},
        agent=_FakeAgent(),
        lookahead_chapters=5,
        on_chapter_ready=received.append,
        resolve_chapter_spans_fn=_fake_resolve_multi_span(FAST_START_HEAD_SPANS),  # exactly the threshold, not over
        synthesize_chapter_fn=_fake_synthesize,
    )
    try:
        renderer.start()
        assert _wait_until(lambda: renderer.chapters_done == 1)
        time.sleep(0.2)
        assert len(received) == 1
    finally:
        renderer.stop()


def test_single_chapter_book_is_never_split_even_with_many_spans() -> None:
    # Splitting a book's only (first-and-last) chapter would make
    # server.py's "no more chapters" sentinel fire after just the head.
    from alebrijes.session.buffer import FAST_START_HEAD_SPANS

    received: list[ChapterAudio] = []
    renderer = LookaheadRenderer(
        _book(1),
        CastFile(book_title="Fixture"),
        voice_states={},
        agent=_FakeAgent(),
        lookahead_chapters=5,
        on_chapter_ready=received.append,
        resolve_chapter_spans_fn=_fake_resolve_multi_span(FAST_START_HEAD_SPANS + 3),
        synthesize_chapter_fn=_fake_synthesize,
    )
    try:
        renderer.start()
        assert _wait_until(lambda: renderer.chapters_done == 1)
        time.sleep(0.2)
        assert len(received) == 1
    finally:
        renderer.stop()


def test_on_line_progress_receives_per_line_updates_from_synthesize_chapter() -> None:
    def _synthesize_with_progress(voiced_spans, voice_states, agent, pause_ms=250, on_line_done=None):
        if on_line_done is not None:
            on_line_done(1, 2)
            on_line_done(2, 2)
        return np.array([1, 2, 3], dtype=np.int16), [f"timing-{voiced_spans[0]}"]

    progress: list[tuple[int, int, int]] = []
    renderer = LookaheadRenderer(
        _book(1),
        CastFile(book_title="Fixture"),
        voice_states={},
        agent=_FakeAgent(),
        lookahead_chapters=5,
        resolve_chapter_spans_fn=_fake_resolve,
        synthesize_chapter_fn=_synthesize_with_progress,
        on_line_progress=lambda chapter_index, done, total: progress.append((chapter_index, done, total)),
    )
    try:
        renderer.start()
        assert _wait_until(lambda: len(progress) == 2)
        assert progress == [(0, 1, 2), (0, 2, 2)]
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
    def _raising_synthesize(voiced_spans, voice_states, agent, pause_ms=250, on_line_done=None):
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
    def _raising_synthesize(voiced_spans, voice_states, agent, pause_ms=250, on_line_done=None):
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
