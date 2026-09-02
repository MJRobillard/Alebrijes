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
from typing import Callable, Iterator, Optional

import numpy as np

from alebrijes.llm.prompts import ChatTurn, Mode as SessionMode, build_messages
from alebrijes.rag import RagIndex, search as search_rag
from alebrijes.transcript import Transcript

DEFAULT_HISTORY_LIMIT = 8
DEFAULT_MAX_IMPROV_EXCHANGES = 3
DEFAULT_TOP_K = 5

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

        self._embed_fn = embed_fn
        self._chat_stream_fn = chat_stream_fn
        self._seq_by_span_id = {span.span_id: i for i, span in enumerate(transcript.spans)}

        self.current_span_id: Optional[str] = transcript.spans[0].span_id if transcript.spans else None
        self.state = SessionState.PLAYING
        self.history: list[ChatTurn] = []
        self.improv_exchange_count = 0

    @property
    def current_seq(self) -> int:
        if self.current_span_id is None:
            return 0
        return self._seq_by_span_id.get(self.current_span_id, 0)

    def _position_summary(self) -> str:
        if self.current_span_id is None or not self.transcript.spans:
            return "the start of the book"
        span = self.transcript.spans[self.current_seq]
        return f"chapter {span.chapter_index + 1}, near: “{span.text[:80]}”"

    def _retrieve(self, query_text: str, *, max_seq: Optional[int] = None) -> list:
        if not self.rag_index.entries:
            return []
        query_vector = self._embed_fn(query_text)
        bound = self.current_seq if max_seq is None else max_seq
        return search_rag(self.rag_index, query_vector, max_seq=bound, top_k=self.top_k)

    def advance_position(self, span_id: str) -> Optional[SessionEvent]:
        """Called as playback reaches `span_id`. Returns an `awaiting_input`
        event (and pauses) if this position is a checkpoint in CHECKPOINT or
        LEARNING mode; otherwise just updates position and returns None."""
        self.current_span_id = span_id
        if self.mode not in (SessionMode.CHECKPOINT, SessionMode.LEARNING):
            return None
        if span_id not in self.checkpoints:
            return None

        self.state = SessionState.PAUSED_FOR_USER
        hits = self._retrieve("what has just happened in the story")
        messages = build_messages(
            mode=self.mode,
            book_title=self.book_title,
            character_voice=self.character_voice,
            position_summary=self._position_summary(),
            rag_excerpts=[hit.entry.text for hit in hits],
            history=self.history[-self.history_limit :],
            user_message="(narration has reached a pause point)",
        )
        prompt_text = "".join(self._chat_stream_fn(messages))
        self.history.append(ChatTurn(role="assistant", content=prompt_text))
        self.history = self.history[-self.history_limit :]
        return SessionEvent(kind="awaiting_input", text=prompt_text)

    def handle_user_message(self, user_message: str) -> SessionEvent:
        self.state = SessionState.GENERATING_RESPONSE
        hits = self._retrieve(user_message)
        rag_excerpts = [hit.entry.text for hit in hits]
        sources = [hit.entry.span_id for hit in hits]

        if self.mode is SessionMode.CYOA:
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
            user_message=user_message,
        )
        answer = "".join(self._chat_stream_fn(messages))

        self.history.append(ChatTurn(role="user", content=user_message))
        self.history.append(ChatTurn(role="assistant", content=answer))
        self.history = self.history[-self.history_limit :]
        self.state = SessionState.PLAYING

        return SessionEvent(kind="answer", text=answer, sources=sources)
