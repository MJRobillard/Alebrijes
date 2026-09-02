from __future__ import annotations

import numpy as np
import pytest

from alebrijes.rag import RagEntry, RagIndex
from alebrijes.segment import SpanKind
from alebrijes.session import InteractiveSession, SessionMode, SessionState
from alebrijes.transcript import SpanTiming, Transcript


def _transcript(num_spans: int = 5) -> Transcript:
    spans = [
        SpanTiming(
            span_id=f"0.{i}.0",
            chapter_index=0,
            paragraph_index=i,
            character="Narrator",
            kind=SpanKind.NARRATION,
            text=f"Span text number {i}.",
            start_ms=i * 1000,
            end_ms=(i + 1) * 1000,
        )
        for i in range(num_spans)
    ]
    return Transcript(book_title="Fixture Book", chapters=[], spans=spans)


def _rag_index(num_spans: int = 5) -> RagIndex:
    # One-hot vectors per seq -- embed_fn below echoes them back so search
    # results are fully deterministic in tests.
    entries = [RagEntry(span_id=f"0.{i}.0", chapter_index=0, paragraph_index=i, seq=i, text=f"Span text number {i}.") for i in range(num_spans)]
    vectors = np.eye(num_spans, dtype=np.float32)
    return RagIndex(entries=entries, vectors=vectors)


def _echo_embed_fn(num_spans: int):
    # "seq:N" as a query returns the one-hot vector for span N -- lets tests
    # control exactly which span search() will rank first.
    def embed(text: str) -> np.ndarray:
        if text.startswith("seq:"):
            vec = np.zeros(num_spans, dtype=np.float32)
            vec[int(text[len("seq:") :])] = 1.0
            return vec
        return np.ones(num_spans, dtype=np.float32)  # generic query: no strong preference

    return embed


def _session(**overrides) -> InteractiveSession:
    num_spans = overrides.pop("num_spans", 5)
    chat_calls = overrides.pop("chat_calls", None)

    def default_chat_stream_fn(messages, *, model="hermes3:8b"):
        if chat_calls is not None:
            chat_calls.append(messages)
        yield "canned response"

    kwargs = dict(
        book_title="Fixture Book",
        character_voice="Narrator",
        transcript=_transcript(num_spans),
        rag_index=_rag_index(num_spans),
        embed_fn=_echo_embed_fn(num_spans),
        chat_stream_fn=default_chat_stream_fn,
    )
    kwargs.update(overrides)
    return InteractiveSession(**kwargs)


def test_initial_state_is_playing_at_first_span() -> None:
    session = _session()
    assert session.state is SessionState.PLAYING
    assert session.current_span_id == "0.0.0"
    assert session.current_seq == 0


def test_advance_position_updates_current_seq() -> None:
    session = _session()
    session.advance_position("0.2.0")
    assert session.current_span_id == "0.2.0"
    assert session.current_seq == 2


def test_qa_mode_ignores_checkpoints_and_never_pauses() -> None:
    session = _session(mode=SessionMode.QA, checkpoints=["0.2.0"])
    event = session.advance_position("0.2.0")
    assert event is None
    assert session.state is SessionState.PLAYING


def test_checkpoint_mode_pauses_only_at_a_registered_checkpoint() -> None:
    session = _session(mode=SessionMode.CHECKPOINT, checkpoints=["0.2.0"])

    non_checkpoint_event = session.advance_position("0.1.0")
    assert non_checkpoint_event is None
    assert session.state is SessionState.PLAYING

    checkpoint_event = session.advance_position("0.2.0")
    assert checkpoint_event is not None
    assert checkpoint_event.kind == "awaiting_input"
    assert checkpoint_event.text == "canned response"
    assert session.state is SessionState.PAUSED_FOR_USER


def test_handle_user_message_resumes_playing_after_a_checkpoint_pause() -> None:
    session = _session(mode=SessionMode.CHECKPOINT, checkpoints=["0.2.0"])
    session.advance_position("0.2.0")
    assert session.state is SessionState.PAUSED_FOR_USER

    event = session.handle_user_message("let's continue")

    assert event.kind == "answer"
    assert session.state is SessionState.PLAYING


def test_handle_user_message_sends_mode_specific_messages_to_chat_fn() -> None:
    calls = []
    session = _session(mode=SessionMode.QA, chat_calls=calls)

    session.handle_user_message("who is that?")

    assert len(calls) == 1
    messages = calls[0]
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "who is that?"}


def test_handle_user_message_bounds_sources_to_current_position() -> None:
    session = _session()
    session.advance_position("0.1.0")  # seq 1

    event = session.handle_user_message("seq:4")  # would rank span 4 highest if unbounded

    assert all(int(sid.split(".")[1]) <= 1 for sid in event.sources)


def test_cyoa_mode_injects_steer_back_beat_after_max_improv_exchanges() -> None:
    session = _session(mode=SessionMode.CYOA, max_improv_exchanges=2)
    session.advance_position("0.1.0")  # seq 1, so seq 4 is "upcoming"

    captured = []

    def chat_stream_fn(messages, *, model="hermes3:8b"):
        captured.append(messages[0]["content"])  # system prompt, carries rag_excerpts
        yield "ok"

    session._chat_stream_fn = chat_stream_fn

    session.handle_user_message("seq:0")  # exchange 1: no steer-back yet
    assert session.improv_exchange_count == 1
    assert "Span text number 4" not in captured[-1]

    session.handle_user_message("seq:4")  # exchange 2: hits the cap, steer-back injected
    assert session.improv_exchange_count == 0
    assert "Span text number 4" in captured[-1]


def test_handle_user_message_calls_on_delta_progressively() -> None:
    def chat_stream_fn(messages, *, model="hermes3:8b"):
        yield "Part one. "
        yield "Part two."

    session = _session(chat_stream_fn=chat_stream_fn)
    deltas = []

    event = session.handle_user_message("hi", on_delta=deltas.append)

    assert deltas == ["Part one. ", "Part two."]
    assert event.text == "Part one. Part two."


def test_advance_position_checkpoint_prompt_also_supports_on_delta() -> None:
    def chat_stream_fn(messages, *, model="hermes3:8b"):
        yield "Pause "
        yield "point."

    session = _session(mode=SessionMode.CHECKPOINT, checkpoints=["0.2.0"], chat_stream_fn=chat_stream_fn)
    deltas = []

    event = session.advance_position("0.2.0", on_delta=deltas.append)

    assert deltas == ["Pause ", "point."]
    assert event.text == "Pause point."


def test_extend_transcript_appends_spans_and_updates_seq_lookup() -> None:
    session = _session(num_spans=2)
    new_span = SpanTiming(
        span_id="0.2.0",
        chapter_index=1,
        paragraph_index=0,
        character="Narrator",
        kind=SpanKind.NARRATION,
        text="A new chapter begins.",
        start_ms=2000,
        end_ms=3000,
    )

    session.extend_transcript([new_span])

    assert len(session.transcript.spans) == 3
    session.advance_position("0.2.0")
    assert session.current_seq == 2


def test_history_is_capped_to_history_limit() -> None:
    session = _session(history_limit=2)
    for i in range(5):
        session.handle_user_message(f"question {i}")

    assert len(session.history) == 2
    assert session.history[-1].content == "canned response"
