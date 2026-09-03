"""Interactive listening sessions: one state machine, four modes.

QA (free questions), CYOA (improvised branch-and-return), CHECKPOINT
(pause for a choice), and LEARNING (comprehension question with a reactive
follow-up) all go through the same `handle_user_message` entry point and
the same `llm.prompts.build_messages` call — they differ only in trigger
condition, the mode-specific prompt fragment (`llm/prompts.py`), and
whether responding blocks playback. See PHASES plan §4 for the rationale.

`buffer.py` (not imported here) is the separate rolling-lookahead renderer
that feeds this session its audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from alebrijes.lesson import QuestionResult, QuizProgress, QuizQuestion, parse_choice, save_progress
from alebrijes.lesson import load_progress as load_quiz_progress
from alebrijes.llm.prompts import ChatTurn, Mode as SessionMode, build_messages
from alebrijes.rag import RagIndex, search as search_rag
from alebrijes.transcript import Transcript

DEFAULT_HISTORY_LIMIT = 8
DEFAULT_MAX_IMPROV_EXCHANGES = 3
DEFAULT_TOP_K = 5

# Fallback for a correct quiz answer with no per-question QuizQuestion.praise
# (see InteractiveSession._pick_praise) -- rotated through so a run of
# correct answers doesn't repeat the same line every time.
_GENERIC_PRAISE = ["Correct.", "That's it.", "Exactly right.", "Well done."]

__all__ = [
    "SessionMode",
    "SessionState",
    "SessionEvent",
    "InteractiveSession",
]


class SessionState(str, Enum):
    PLAYING = "playing"
    PAUSED_FOR_USER = "paused_for_user"
    GENERATING_RESPONSE = "generating_response"


@dataclass
class SessionEvent:
    kind: str  # "answer" | "awaiting_input"
    text: str = ""
    sources: list[str] = field(default_factory=list)


class InteractiveSession:
    def __init__(
        self,
        *,
        book_title: str,
        character_voice: str,
        transcript: Transcript,
        rag_index: RagIndex,
        embed_fn: Callable[[str], np.ndarray],
        chat_stream_fn: Callable[..., Iterator[str]],
        mode: SessionMode = SessionMode.QA,
        checkpoints: Optional[list[str]] = None,
        max_improv_exchanges: int = DEFAULT_MAX_IMPROV_EXCHANGES,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        top_k: int = DEFAULT_TOP_K,
        quiz_questions: Optional[dict[str, QuizQuestion]] = None,
        quiz_checkpoints: Optional[dict[str, str]] = None,
        quiz_progress_path: Optional[Path] = None,
        coffee_premise: str = "",
    ) -> None:
        self.book_title = book_title
        self.character_voice = character_voice
        self.transcript = transcript
        self.rag_index = rag_index
        self.mode = mode
        self.checkpoints = set(checkpoints or [])
        self.max_improv_exchanges = max_improv_exchanges
        self.history_limit = history_limit
        self.top_k = top_k

        # Lesson mode: authored practice questions (quiz_checkpoints maps a
        # checkpoint's span_id -> question id in quiz_questions -- see
        # ingest/script.py's `QUIZ:` marker and lesson.py), a persisted
        # running score, and a free-form premise document that grounds
        # CYOA ("coffee mode") improvisation. All optional -- a session with
        # none of this behaves exactly as before lesson mode existed.
        self.quiz_questions = quiz_questions or {}
        self.quiz_checkpoints = quiz_checkpoints or {}
        self.quiz_progress_path = quiz_progress_path
        self.quiz_progress: QuizProgress = (
            load_quiz_progress(quiz_progress_path, book_title=book_title)
            if quiz_progress_path is not None
            else QuizProgress(book_title=book_title)
        )
        self.coffee_premise = coffee_premise
        self.awaiting_quiz_question: Optional[QuizQuestion] = None

        self._embed_fn = embed_fn
        self._chat_stream_fn = chat_stream_fn
        self._seq_by_span_id = {span.span_id: i for i, span in enumerate(transcript.spans)}

        self.current_span_id: Optional[str] = transcript.spans[0].span_id if transcript.spans else None
        self.state = SessionState.PLAYING
        self.history: list[ChatTurn] = []
        self.improv_exchange_count = 0
        self._praise_rotation_index = 0
        # span_id -> prompt_text, populated by prefetch_checkpoint ahead of
        # playback actually reaching a (non-quiz-question) checkpoint, so
        # advance_position can skip the LLM round-trip. A checkpoint with
        # an attached QuizQuestion is never cached here -- it's asked
        # verbatim, no LLM involved at all.
        self._checkpoint_prompt_cache: dict[str, str] = {}

    @property
    def current_seq(self) -> int:
        if self.current_span_id is None:
            return 0
        return self._seq_by_span_id.get(self.current_span_id, 0)

    def _position_summary(self, seq: Optional[int] = None) -> str:
        if self.current_span_id is None or not self.transcript.spans:
            return "the start of the book"
        span = self.transcript.spans[self.current_seq if seq is None else seq]
        return f"chapter {span.chapter_index + 1}, near: “{span.text[:80]}”"

    def _current_paragraph_text(self, seq: Optional[int] = None) -> str:
        """Full (untruncated) text of the span at the given position (the
        listener's current one by default) -- unlike `_position_summary`'s
        80-char pointer, this is meant for a chat backend with its own RAG
        over the same book (e.g. `llm.scripter.ScripterClient`) to ground
        retrieval/replies in exactly where the listener is, not just the
        raw question.

        `seq` is overridden by `prefetch_checkpoint` to build a checkpoint's
        prompt against *its own* position, since prefetching only makes
        sense ahead of the listener actually reaching it -- current_seq
        would be stale at that point."""
        if self.current_span_id is None or not self.transcript.spans:
            return ""
        return self.transcript.spans[self.current_seq if seq is None else seq].text

    def extend_transcript(self, new_spans: list) -> None:
        """Appends newly-available spans (e.g. a chapter a live
        `session.buffer.LookaheadRenderer` just finished) to the transcript
        this session was constructed with, keeping `current_seq`/RAG
        position-bounding correct for a book that's still being rendered."""
        start_seq = len(self.transcript.spans)
        self.transcript.spans.extend(new_spans)
        for offset, span in enumerate(new_spans):
            self._seq_by_span_id[span.span_id] = start_seq + offset

    def _retrieve(self, query_text: str, *, max_seq: Optional[int] = None) -> list:
        if not self.rag_index.entries:
            return []
        query_vector = self._embed_fn(query_text)
        bound = self.current_seq if max_seq is None else max_seq
        return search_rag(self.rag_index, query_vector, max_seq=bound, top_k=self.top_k)

    def _checkpoint_question(self, span_id: str) -> Optional[QuizQuestion]:
        question_id = self.quiz_checkpoints.get(span_id) if self.mode is SessionMode.LEARNING else None
        return self.quiz_questions.get(question_id) if question_id else None

    def _checkpoint_prompt_messages(self, span_id: str) -> list[dict]:
        """Builds the "react to reaching this checkpoint in character"
        messages for `span_id` -- shared by `advance_position` (live) and
        `prefetch_checkpoint` (ahead of time). Grounded in `span_id`'s own
        position (not necessarily `self.current_span_id`, which is stale
        during a prefetch), so RAG retrieval and position context are
        correct either way.

        Only ever called for a checkpoint with no attached QuizQuestion --
        one *with* a question is asked verbatim, no LLM involved at all
        (see `_quiz_question_text`); this is for a plain CHECKPOINT-mode
        pause, or a LEARNING-mode checkpoint that isn't actually a quiz
        question, where there's nothing pre-authored to say and the LLM
        has to generate a reaction."""
        seq = self._seq_by_span_id.get(span_id, self.current_seq)
        hits = self._retrieve("what has just happened in the story", max_seq=seq)
        return build_messages(
            mode=self.mode,
            book_title=self.book_title,
            character_voice=self.character_voice,
            position_summary=self._position_summary(seq),
            rag_excerpts=[hit.entry.text for hit in hits],
            history=self.history[-self.history_limit :],
            user_message="(narration has reached a pause point)",
            current_paragraph=self._current_paragraph_text(seq),
        )

    def prefetch_checkpoint(self, span_id: str) -> None:
        """Precomputes and caches the checkpoint response for `span_id` --
        the (sometimes multi-second) LLM round-trip that `advance_position`
        would otherwise have to do right as playback reaches it. Meant to
        be called well ahead of that, e.g. as soon as the chapter
        containing this checkpoint is rendered (see server.py's
        `_drain_chapters`), so by the time the listener actually gets
        there the response is just sitting in the cache.

        Safe to call any time before the checkpoint is reached, and safe
        to call concurrently for several checkpoints at once (each writes
        a distinct cache key) -- reads session state but mutates only the
        cache, never `awaiting_quiz_question`/history/state, which must
        only change once the checkpoint is actually reached. A no-op if
        `span_id` isn't a checkpoint in the current mode, is already
        cached, or (see `_checkpoint_prompt_messages`) has an attached
        QuizQuestion -- nothing to prefetch there, it's asked verbatim."""
        if span_id in self._checkpoint_prompt_cache or span_id not in self.checkpoints:
            return
        if self.mode not in (SessionMode.CHECKPOINT, SessionMode.LEARNING):
            return
        if self._checkpoint_question(span_id) is not None:
            return
        messages = self._checkpoint_prompt_messages(span_id)
        prompt_text = "".join(self._chat_stream_fn(messages))
        self._checkpoint_prompt_cache[span_id] = prompt_text

    def advance_position(
        self, span_id: str, *, on_delta: Optional[Callable[[str], None]] = None
    ) -> Optional[SessionEvent]:
        """Called as playback reaches `span_id`. Returns an `awaiting_input`
        event (and pauses) if this position is a checkpoint in CHECKPOINT or
        LEARNING mode; otherwise just updates position and returns None.
        See `handle_user_message` for what `on_delta` is for."""
        self.current_span_id = span_id
        if self.mode not in (SessionMode.CHECKPOINT, SessionMode.LEARNING):
            return None
        if span_id not in self.checkpoints:
            return None

        self.state = SessionState.PAUSED_FOR_USER

        question = self._checkpoint_question(span_id)
        if question is not None:
            # Asked verbatim -- the question is already exactly authored
            # (quiz.yaml), so there's nothing for the LLM to generate here.
            # It only gets involved afterward: _incorrect_note (a wrong
            # answer) or the normal chat flow (free text instead of a
            # letter) -- see handle_user_message.
            self.awaiting_quiz_question = question
            prompt_text = self._quiz_question_text(question)
            if on_delta is not None:
                on_delta(prompt_text)
        else:
            cached = self._checkpoint_prompt_cache.pop(span_id, None)
            if cached is not None:
                prompt_text = cached
                if on_delta is not None:
                    # The whole (already-generated) text in one call --
                    # still gets synthesized sentence-by-sentence
                    # downstream (see server.py's SentenceChunker), just
                    # without the LLM token-streaming step this skips.
                    on_delta(prompt_text)
            else:
                messages = self._checkpoint_prompt_messages(span_id)
                chunks: list[str] = []
                for delta in self._chat_stream_fn(messages):
                    if on_delta is not None:
                        on_delta(delta)
                    chunks.append(delta)
                prompt_text = "".join(chunks)

        self.history.append(ChatTurn(role="assistant", content=prompt_text))
        self.history = self.history[-self.history_limit :]
        return SessionEvent(kind="awaiting_input", text=prompt_text)

    @staticmethod
    def _quiz_question_text(question: QuizQuestion) -> str:
        """The question exactly as authored in quiz.yaml -- spoken/shown
        verbatim, never paraphrased by an LLM."""
        options_block = "\n".join(f"{key}) {text}" for key, text in question.options.items())
        return f"{question.prompt}\n{options_block}"

    def _grade_quiz_answer(self, choice: str) -> tuple[bool, QuizQuestion]:
        """Grades `choice` against the pending question, deterministically
        (the LLM never judges correctness), updates and persists
        `quiz_progress`, and clears `awaiting_quiz_question`.

        See `handle_user_message` for what happens with the result: a
        correct answer skips the LLM entirely (nothing here needs
        authoring); an incorrect one gets `_incorrect_note`'s instructive
        note for the LLM to react to."""
        question = self.awaiting_quiz_question
        assert question is not None
        is_correct = choice == question.correct

        prior = self.quiz_progress.answered.get(question.id)
        attempts = (prior.attempts if prior else 0) + 1
        self.quiz_progress.answered[question.id] = QuestionResult(
            correct=is_correct, attempts=attempts, last_answer=choice
        )
        if self.quiz_progress_path is not None:
            save_progress(self.quiz_progress, self.quiz_progress_path)
        self.awaiting_quiz_question = None
        return is_correct, question

    @staticmethod
    def _incorrect_note(choice: str, question: QuizQuestion) -> str:
        correct_text = question.options.get(question.correct, "")
        return (
            f"(the listener answered {choice}, which is incorrect -- the correct "
            f"answer was {question.correct}) {correct_text}. React in character to "
            "their specific answer, gently correcting the misconception.)"
        )

    def _pick_praise(self, question: QuizQuestion) -> str:
        """An author-written `praise` on the question wins; otherwise
        rotate through a small generic list so consecutive correct answers
        don't all get the identical line."""
        if question.praise:
            return question.praise
        praise = _GENERIC_PRAISE[self._praise_rotation_index % len(_GENERIC_PRAISE)]
        self._praise_rotation_index += 1
        return praise

    def handle_user_message(
        self, user_message: str, *, on_delta: Optional[Callable[[str], None]] = None
    ) -> SessionEvent:
        """`on_delta`, if given, is called with each raw content delta from
        the LLM as it streams in — before this method has assembled the
        full answer or updated history. Callers that need to react
        progressively (e.g. server.py's chat handler, synthesizing audio
        sentence-by-sentence via `synth.streaming.SentenceChunker` as the
        response is generated) hook in here instead of waiting for the
        `SessionEvent` this returns.
        """
        self.state = SessionState.GENERATING_RESPONSE

        # A pending quiz question takes priority over a normal LLM turn --
        # but only if the message actually parses as a choice; free text
        # while a question is pending falls through to the normal flow
        # below rather than forcing a grade on an unrelated message.
        effective_message = user_message
        if self.awaiting_quiz_question is not None:
            choice = parse_choice(user_message)
            if choice is not None:
                is_correct, question = self._grade_quiz_answer(choice)
                if is_correct:
                    # Deterministic grade -- nothing here needs the LLM to
                    # author a reaction. Feed the praise through on_delta
                    # directly so it's still synthesized/streamed as audio
                    # via the exact same pipeline a real LLM response uses
                    # (server.py's _speak_llm_call), just without the
                    # network round-trip.
                    praise = self._pick_praise(question)
                    if on_delta is not None:
                        on_delta(praise)
                    self.history.append(ChatTurn(role="user", content=user_message))
                    self.history.append(ChatTurn(role="assistant", content=praise))
                    self.history = self.history[-self.history_limit :]
                    self.state = SessionState.PLAYING
                    return SessionEvent(kind="answer", text=praise, sources=[])
                effective_message = self._incorrect_note(choice, question)

        hits = self._retrieve(user_message)
        rag_excerpts = [hit.entry.text for hit in hits]
        sources = [hit.entry.span_id for hit in hits]

        if self.mode is SessionMode.CYOA:
            if self.coffee_premise:
                rag_excerpts = [self.coffee_premise] + rag_excerpts

            self.improv_exchange_count += 1
            if self.improv_exchange_count >= self.max_improv_exchanges:
                upcoming = self._retrieve(user_message, max_seq=len(self.transcript.spans) - 1)
                steer_back = [hit.entry.text for hit in upcoming if hit.entry.seq > self.current_seq][:1]
                rag_excerpts = rag_excerpts + steer_back
                self.improv_exchange_count = 0

        messages = build_messages(
            mode=self.mode,
            book_title=self.book_title,
            character_voice=self.character_voice,
            position_summary=self._position_summary(),
            rag_excerpts=rag_excerpts,
            history=self.history[-self.history_limit :],
            user_message=effective_message,
            current_paragraph=self._current_paragraph_text(),
        )
        chunks: list[str] = []
        for delta in self._chat_stream_fn(messages):
            if on_delta is not None:
                on_delta(delta)
            chunks.append(delta)
        answer = "".join(chunks)

        self.history.append(ChatTurn(role="user", content=user_message))
        self.history.append(ChatTurn(role="assistant", content=answer))
        self.history = self.history[-self.history_limit :]
        self.state = SessionState.PLAYING

        return SessionEvent(kind="answer", text=answer, sources=sources)
