"""Rolling lookahead renderer for interactive listening sessions.

Renders a book chapter-by-chapter, staying at most `lookahead_chapters`
ahead of the playback position the consumer reports via `mark_consumed`,
instead of `render.render_book`'s whole-book-upfront approach — so playback
can start as soon as chapter 0 is ready, not after the whole book renders.

Reuses `render.resolve_chapter_spans` / `render.synthesize_chapter`
unchanged (injectable here only for testing) — this is a different
consumer of those building blocks, not a fork of them. Background
rendering keeps running through an interrupt: it's *consumption*
(`mark_consumed`) that pauses when a session is mid-conversation, not
generation — see the session state machine in `session/__init__.py`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from alebrijes.cast import CastFile
from alebrijes.ingest import Book
from alebrijes.render import resolve_chapter_spans, synthesize_chapter
from alebrijes.transcript import SpanTiming

DEFAULT_LOOKAHEAD_CHAPTERS = 2

# The very first chapter's audio otherwise has to fully finish -- every
# line, synthesized sequentially -- before the listener hears anything at
# all, which for a long chapter can be minutes of dead air. Splitting just
# chapter 0 into a small "head" (synthesized and sent first) plus the rest
# gets real audio playing almost immediately, at the cost of one extra
# on_chapter_ready call for that one chapter. A whole-span count, not a
# duration or sentence count: splitting *inside* a span would let its
# SpanTiming straddle two separate audio blobs, which the transcript
# highlighting (one span's timing is local to the one blob it's in) and
# quiz-checkpoint triggering (keyed to a whole span_id) both assume never
# happens.
FAST_START_HEAD_SPANS = 2


@dataclass
class ChapterAudio:
    chapter_index: int
    title: str
    audio: np.ndarray
    sample_rate: int
    timings: list[SpanTiming] = field(default_factory=list)


class LookaheadRenderer:
    def __init__(
        self,
        book: Book,
        cast_file: CastFile,
        voice_states: dict,
        agent: object,
        *,
        lookahead_chapters: int = DEFAULT_LOOKAHEAD_CHAPTERS,
        pause_ms: int = 250,
        on_chapter_ready: Optional[Callable[[ChapterAudio], None]] = None,
        on_error: Optional[Callable[[int, Exception], None]] = None,
        on_line_progress: Optional[Callable[[int, int, int], None]] = None,
        resolve_chapter_spans_fn: Callable = resolve_chapter_spans,
        synthesize_chapter_fn: Callable = synthesize_chapter,
    ) -> None:
        self._book = book
        self._cast_file = cast_file
        self._voice_states = voice_states
        self._agent = agent
        self._lookahead_chapters = lookahead_chapters
        self._pause_ms = pause_ms
        self._on_chapter_ready = on_chapter_ready
        self._on_error = on_error
        # (chapter_index, lines_done, lines_total) -- fires per span within
        # the chapter currently being synthesized, since on_chapter_ready
        # only fires once the *whole* chapter is done (see synthesize_chapter's
        # on_line_done docstring).
        self._on_line_progress = on_line_progress
        self._resolve_chapter_spans = resolve_chapter_spans_fn
        self._synthesize_chapter = synthesize_chapter_fn

        self._consumed_chapter = -1
        self._chapters_done = 0
        self._error: Optional[Exception] = None
        self._cond = threading.Condition()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def chapters_done(self) -> int:
        with self._cond:
            return self._chapters_done

    @property
    def error(self) -> Optional[Exception]:
        with self._cond:
            return self._error

    def start(self) -> "LookaheadRenderer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=5)

    def mark_consumed(self, chapter_index: int) -> None:
        """Reports that playback has finished with `chapter_index`, letting
        the renderer advance its lookahead window past it."""
        with self._cond:
            if chapter_index > self._consumed_chapter:
                self._consumed_chapter = chapter_index
                self._cond.notify_all()

    def _synthesize_and_emit(self, chapter, voiced_spans: list) -> bool:
        """Synthesizes one piece of `chapter`'s spans and emits it via
        on_chapter_ready. Returns False (after recording/reporting the
        error) on failure, so the caller can stop -- same contract the old
        single-call-per-chapter `_run` body had."""
        try:

            def _on_line_done(lines_done: int, lines_total: int) -> None:
                if self._on_line_progress is not None:
                    self._on_line_progress(chapter.index, lines_done, lines_total)

            audio, timings = self._synthesize_chapter(
                voiced_spans,
                self._voice_states,
                self._agent,
                pause_ms=self._pause_ms,
                on_line_done=_on_line_done,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced via .error, not crashing the thread
            with self._cond:
                self._error = exc
            if self._on_error is not None:
                self._on_error(chapter.index, exc)
            return False

        chapter_audio = ChapterAudio(
            chapter_index=chapter.index,
            title=chapter.title or f"Chapter {chapter.index + 1}",
            audio=audio,
            sample_rate=self._agent.sample_rate,
            timings=timings,
        )
        if self._on_chapter_ready is not None:
            self._on_chapter_ready(chapter_audio)
        return True

    def _run(self) -> None:
        for chapter in self._book.chapters:
            with self._cond:
                while (
                    not self._stop_event.is_set()
                    and chapter.index - self._consumed_chapter > self._lookahead_chapters
                ):
                    self._cond.wait(timeout=0.5)
            if self._stop_event.is_set():
                return

            try:
                voiced_spans = self._resolve_chapter_spans(
                    chapter, self._cast_file, script_format=self._book.source_format == "script"
                )
            except Exception as exc:  # noqa: BLE001 - surfaced via .error, not crashing the thread
                with self._cond:
                    self._error = exc
                if self._on_error is not None:
                    self._on_error(chapter.index, exc)
                return

            # Only the book's very first chapter gets the fast-start split
            # (see FAST_START_HEAD_SPANS) -- and only when there's a later
            # chapter to fall back on, since server.py's "no more chapters"
            # sentinel fires off this chapter being the *last* one too; on
            # a single-chapter book that would fire after just the head.
            is_fast_start_chapter = (
                chapter is self._book.chapters[0]
                and len(self._book.chapters) > 1
                and len(voiced_spans) > FAST_START_HEAD_SPANS
            )
            if is_fast_start_chapter:
                head, tail = voiced_spans[:FAST_START_HEAD_SPANS], voiced_spans[FAST_START_HEAD_SPANS:]
                if not self._synthesize_and_emit(chapter, head):
                    return
                voiced_spans = tail

            if not self._synthesize_and_emit(chapter, voiced_spans):
                return

            with self._cond:
                self._chapters_done += 1
