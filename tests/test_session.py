from __future__ import annotations

import numpy as np
import pytest

from alebrijes.lesson import QuizProgress, QuizQuestion
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


# --- lesson mode: quiz checkpoints, deterministic grading, coffee premise ---


def _quiz_question() -> QuizQuestion:
    return QuizQuestion(
        id="q1",
        prompt="What is an IAM role?",
        options={"A": "A user", "B": "A delegated identity"},
        correct="B",
    )


def test_learning_checkpoint_with_attached_question_sets_awaiting_quiz_question() -> None:
    session = _session(
        mode=SessionMode.LEARNING,
        checkpoints=["0.2.0"],
        quiz_checkpoints={"0.2.0": "q1"},
        quiz_questions={"q1": _quiz_question()},
    )

    event = session.advance_position("0.2.0")

    assert session.awaiting_quiz_question is not None
    assert session.awaiting_quiz_question.id == "q1"
    assert event.kind == "awaiting_input"


def test_learning_checkpoint_with_question_asks_it_verbatim_without_calling_the_llm() -> None:
    calls = []
    session = _session(
        mode=SessionMode.LEARNING,
        checkpoints=["0.2.0"],
        quiz_checkpoints={"0.2.0": "q1"},
        quiz_questions={"q1": _quiz_question()},
        chat_calls=calls,
    )

    event = session.advance_position("0.2.0")

    assert calls == []  # nothing to generate -- the question is already authored
    assert event.kind == "awaiting_input"
    assert "What is an IAM role?" in event.text
    assert "A) A user" in event.text
    assert "B) A delegated identity" in event.text
    assert session.awaiting_quiz_question is not None
    assert session.awaiting_quiz_question.id == "q1"


def test_prefetch_checkpoint_is_a_noop_for_a_checkpoint_with_a_question() -> None:
    # Nothing to prefetch -- asking it doesn't touch the LLM at all.
    calls = []
    session = _session(
        mode=SessionMode.LEARNING,
        checkpoints=["0.2.0"],
        quiz_checkpoints={"0.2.0": "q1"},
        quiz_questions={"q1": _quiz_question()},
        chat_calls=calls,
    )

    session.prefetch_checkpoint("0.2.0")

    assert calls == []


def test_prefetch_checkpoint_caches_response_without_mutating_session_state() -> None:
    # No quiz_checkpoints entry -- a plain CHECKPOINT-mode pause, which
    # still needs an LLM-authored reaction (nothing pre-written for it).
    calls = []
    session = _session(
        mode=SessionMode.CHECKPOINT,
        checkpoints=["0.2.0"],
        chat_calls=calls,
    )

    session.prefetch_checkpoint("0.2.0")

    assert len(calls) == 1  # the LLM call happened...
    assert session.awaiting_quiz_question is None  # ...but nothing else changed yet
    assert session.state == SessionState.PLAYING
    assert session.current_span_id == "0.0.0"
    assert session.history == []


def test_advance_position_uses_prefetched_response_without_calling_the_llm_again() -> None:
    calls = []
    session = _session(
        mode=SessionMode.CHECKPOINT,
        checkpoints=["0.2.0"],
        chat_calls=calls,
    )
    session.prefetch_checkpoint("0.2.0")
    assert len(calls) == 1

    event = session.advance_position("0.2.0")

    assert len(calls) == 1  # advance_position did not call the LLM again
    assert event.kind == "awaiting_input"
    assert event.text == "canned response"
    assert session.awaiting_quiz_question is None


def test_advance_position_falls_back_to_a_live_call_when_not_prefetched() -> None:
    calls = []
    session = _session(
        mode=SessionMode.CHECKPOINT,
        checkpoints=["0.2.0"],
        chat_calls=calls,
    )

    event = session.advance_position("0.2.0")

    assert len(calls) == 1
    assert event.kind == "awaiting_input"


def test_prefetch_checkpoint_is_a_noop_for_a_non_checkpoint_span() -> None:
    calls = []
    session = _session(
        mode=SessionMode.CHECKPOINT,
        checkpoints=["0.2.0"],
        chat_calls=calls,
    )

    session.prefetch_checkpoint("0.3.0")  # not in checkpoints

    assert calls == []


def test_prefetch_checkpoint_is_a_noop_when_already_cached() -> None:
    calls = []
    session = _session(
        mode=SessionMode.CHECKPOINT,
        checkpoints=["0.2.0"],
        chat_calls=calls,
    )

    session.prefetch_checkpoint("0.2.0")
    session.prefetch_checkpoint("0.2.0")

    assert len(calls) == 1


def test_checkpoint_without_attached_question_behaves_as_before() -> None:
    session = _session(
        mode=SessionMode.LEARNING,
        checkpoints=["0.2.0"],
        quiz_checkpoints={},  # no question attached
        quiz_questions={"q1": _quiz_question()},
    )

    session.advance_position("0.2.0")

    assert session.awaiting_quiz_question is None


def test_correct_choice_grades_correct_and_clears_pending_question() -> None:
    session = _session(
        mode=SessionMode.LEARNING,
        checkpoints=["0.2.0"],
        quiz_checkpoints={"0.2.0": "q1"},
        quiz_questions={"q1": _quiz_question()},
    )
    session.advance_position("0.2.0")

    session.handle_user_message("B")

    assert session.awaiting_quiz_question is None
    assert session.quiz_progress.answered["q1"].correct is True
    assert session.quiz_progress.answered["q1"].last_answer == "B"
    assert session.quiz_progress.score == (1, 1)


def test_incorrect_choice_grades_incorrect() -> None:
    session = _session(
        mode=SessionMode.LEARNING,
        checkpoints=["0.2.0"],
        quiz_checkpoints={"0.2.0": "q1"},
        quiz_questions={"q1": _quiz_question()},
    )
    session.advance_position("0.2.0")

    session.handle_user_message("A")

    assert session.quiz_progress.answered["q1"].correct is False
    assert session.quiz_progress.score == (0, 1)


def test_incorrect_answer_grading_note_not_raw_user_text_is_sent_to_the_llm() -> None:
    calls = []
    session = _session(
        mode=SessionMode.LEARNING,
        checkpoints=["0.2.0"],
        quiz_checkpoints={"0.2.0": "q1"},
        quiz_questions={"q1": _quiz_question()},
        chat_calls=calls,
    )
    session.advance_position("0.2.0")
    calls.clear()

    session.handle_user_message("I think A")

    prompt_sent = calls[0][-1]["content"]
    assert prompt_sent != "I think A"
    assert "incorrect" in prompt_sent.lower()
    # The raw text the listener actually said is still what's recorded in history.
    assert session.history[-2].content == "I think A"


def test_wrong_answer_falls_back_to_correct_answer_text_when_llm_call_fails() -> None:
    """If the LLM can't be reached (Ollama down/restarting/timed out), the
    listener still needs to hear that they were wrong and what the right
    answer was -- the grade already happened deterministically, so this
    shouldn't just raise and strand them with a bare error."""

    def failing_chat_stream_fn(messages, *, model="hermes3:8b"):
        raise ConnectionError("could not reach Ollama")
        yield  # pragma: no cover -- makes this a generator, never reached

    session = _session(
        mode=SessionMode.LEARNING,
        checkpoints=["0.2.0"],
        quiz_checkpoints={"0.2.0": "q1"},
        quiz_questions={"q1": _quiz_question()},
        chat_stream_fn=failing_chat_stream_fn,
    )
    session.advance_position("0.2.0")

    deltas = []
    event = session.handle_user_message("A", on_delta=deltas.append)

    assert session.quiz_progress.answered["q1"].correct is False  # grading still happened
    assert "B" in event.text  # the correct option, spoken as a fallback
    assert "delegated identity" in event.text
    assert deltas == [event.text]  # still gets synthesized as audio, same as any other reply
    assert session.state is SessionState.PLAYING  # not stuck in GENERATING_RESPONSE


def test_free_text_chat_still_raises_when_llm_call_fails() -> None:
    """No deterministic fallback exists for a free-text/CYOA turn -- unlike
    a graded wrong answer, there's nothing to fall back to, so this should
    still raise (server.py surfaces it as an "error" message)."""

    def failing_chat_stream_fn(messages, *, model="hermes3:8b"):
        raise ConnectionError("could not reach Ollama")
        yield  # pragma: no cover

    session = _session(chat_stream_fn=failing_chat_stream_fn)

    with pytest.raises(ConnectionError):
        session.handle_user_message("what's happening?")


def test_correct_answer_skips_the_llm_and_speaks_praise_via_on_delta() -> None:
    calls = []
    deltas = []
    session = _session(
        mode=SessionMode.LEARNING,
        checkpoints=["0.2.0"],
        quiz_checkpoints={"0.2.0": "q1"},
        quiz_questions={"q1": _quiz_question()},
        chat_calls=calls,
    )
    session.advance_position("0.2.0")
    calls.clear()

    event = session.handle_user_message("I think B", on_delta=deltas.append)

    assert calls == []  # no LLM round-trip for a deterministically-correct answer
    assert deltas == [event.text]
    assert event.kind == "answer"
    assert event.text  # the (rotated, or per-question) praise line
    assert session.history[-2].content == "I think B"
    assert session.history[-1].content == event.text
    assert session.state == SessionState.PLAYING


def test_correct_answer_uses_per_question_praise_when_set() -> None:
    question = QuizQuestion(
        id="q1", prompt="What is an IAM role?", options={"A": "A user", "B": "A delegated identity"},
        correct="B", praise="You nailed it, as usual.",
    )
    session = _session(
        mode=SessionMode.LEARNING,
        checkpoints=["0.2.0"],
        quiz_checkpoints={"0.2.0": "q1"},
        quiz_questions={"q1": question},
    )
    session.advance_position("0.2.0")

    event = session.handle_user_message("B")

    assert event.text == "You nailed it, as usual."


def test_correct_answer_praise_rotates_through_generic_list_without_repeating_first_two() -> None:
    session = _session(
        mode=SessionMode.LEARNING,
        checkpoints=["0.2.0", "0.3.0"],
        quiz_checkpoints={"0.2.0": "q1", "0.3.0": "q1"},
        quiz_questions={"q1": _quiz_question()},
    )
    session.advance_position("0.2.0")
    first = session.handle_user_message("B").text
    session.advance_position("0.3.0")
    second = session.handle_user_message("B").text

    assert first != second


def test_unparseable_answer_leaves_question_pending() -> None:
    session = _session(
        mode=SessionMode.LEARNING,
        checkpoints=["0.2.0"],
        quiz_checkpoints={"0.2.0": "q1"},
        quiz_questions={"q1": _quiz_question()},
    )
    session.advance_position("0.2.0")

    session.handle_user_message("I'm not sure, can you repeat that?")

    assert session.awaiting_quiz_question is not None
    assert session.quiz_progress.answered == {}


def test_quiz_progress_persists_to_disk_when_path_given(tmp_path) -> None:
    progress_path = tmp_path / "book.progress.json"
    session = _session(
        mode=SessionMode.LEARNING,
        checkpoints=["0.2.0"],
        quiz_checkpoints={"0.2.0": "q1"},
        quiz_questions={"q1": _quiz_question()},
        quiz_progress_path=progress_path,
    )
    session.advance_position("0.2.0")

    session.handle_user_message("B")

    assert progress_path.exists()
    from alebrijes.lesson import load_progress

    reloaded = load_progress(progress_path)
    assert reloaded.answered["q1"].correct is True


def test_quiz_progress_loads_existing_score_at_construction(tmp_path) -> None:
    from alebrijes.lesson import QuestionResult, save_progress

    progress_path = tmp_path / "book.progress.json"
    save_progress(
        QuizProgress(book_title="Fixture Book", answered={"q0": QuestionResult(True, 1, "A")}),
        progress_path,
    )

    session = _session(quiz_progress_path=progress_path)

    assert session.quiz_progress.score == (1, 1)


def test_repeated_attempts_increment_attempts_not_double_count_naively() -> None:
    session = _session(
        mode=SessionMode.LEARNING,
        checkpoints=["0.2.0", "0.3.0"],
        quiz_checkpoints={"0.2.0": "q1", "0.3.0": "q1"},
        quiz_questions={"q1": _quiz_question()},
    )
    session.advance_position("0.2.0")
    session.handle_user_message("A")  # wrong, attempt 1

    session.advance_position("0.3.0")  # same question triggered again
    session.handle_user_message("B")  # right, attempt 2

    assert session.quiz_progress.answered["q1"].attempts == 2
    assert session.quiz_progress.answered["q1"].correct is True
    assert session.quiz_progress.score == (1, 1)  # one question, currently correct


def test_cyoa_includes_coffee_premise_in_prompt() -> None:
    calls = []
    session = _session(
        mode=SessionMode.CYOA,
        coffee_premise="A mysterious fog rolls into town.",
        chat_calls=calls,
    )

    session.handle_user_message("I investigate the fog.")

    assert "A mysterious fog rolls into town." in calls[0][0]["content"]  # system prompt carries rag_excerpts


def test_qa_mode_does_not_include_coffee_premise() -> None:
    calls = []
    session = _session(mode=SessionMode.QA, coffee_premise="A mysterious fog.", chat_calls=calls)

    session.handle_user_message("who is Colin?")

    assert "A mysterious fog." not in calls[0][0]["content"]
