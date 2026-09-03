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


def test_learning_checkpoint_prompt_includes_exact_question_and_options() -> None:
    calls = []
    session = _session(
        mode=SessionMode.LEARNING,
        checkpoints=["0.2.0"],
        quiz_checkpoints={"0.2.0": "q1"},
        quiz_questions={"q1": _quiz_question()},
        chat_calls=calls,
    )

    session.advance_position("0.2.0")

    prompt_sent = calls[0][-1]["content"]
    assert "What is an IAM role?" in prompt_sent
    assert "A) A user" in prompt_sent
    assert "B) A delegated identity" in prompt_sent


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


def test_grading_note_not_raw_user_text_is_sent_to_the_llm() -> None:
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

    session.handle_user_message("I think B")

    prompt_sent = calls[0][-1]["content"]
    assert prompt_sent != "I think B"
    assert "correct" in prompt_sent.lower()
    # The raw text the listener actually said is still what's recorded in history.
    assert session.history[-2].content == "I think B"


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
