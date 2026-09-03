"""HTTP client for a local "scripter" chat engine (github.com/.../scripter,
`steering/server.py`'s `POST /chat {session_id, message} -> {reply}`).

Unlike Ollama, this server is not a generic instruction-following chat
model: it holds its own per-session history/summary server-side and grounds
replies through its own RAG + control-vector method (see `steering.chat`).
So unlike `OllamaClient.chat_stream`, this adapter does not forward the full
`messages` list (system prompt, narrator_guidelines, book context) each
call -- the server wouldn't use it anyway -- it only forwards the latest
user turn, keyed to one `session_id` for the lifetime of this client so the
server's own history threads correctly across a conversation.

Also unlike Ollama, `/chat` is not a streaming endpoint (`stream: False` is
hardcoded on the server side) -- `chat_stream` still returns an `Iterator[str]`
to match `alebrijes.server.ChatStreamFn`'s contract, it just yields the
whole reply as a single chunk instead of incremental deltas.
"""

from __future__ import annotations

import uuid
from typing import Iterator, Optional

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
MODEL_NAME = "local"

# steering/chat.py's take_turn can be slow (RAG + control-vector generation
# on CPU/GPU) -- generous read timeout, same rationale as ollama.py's.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)


class ScripterUnavailable(Exception):
    """Raised when the local scripter chat server can't be reached or errors out."""


class ScripterClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, client: Optional[httpx.Client] = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=_DEFAULT_TIMEOUT)
        # One session_id per ScripterClient instance -- callers that want
        # independent conversations (e.g. one per Alebrijes interactive
        # session) should construct a fresh client each time, mirroring how
        # server.py's _pin_chat_model makes a new one per session.
        self._session_id = uuid.uuid4().hex

    def chat_stream(self, messages: list[dict], *, model: str = MODEL_NAME) -> Iterator[str]:
        latest_user_message = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        try:
            response = self._client.post(
                f"{self._base_url}/chat",
                json={"session_id": self._session_id, "message": latest_user_message},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ScripterUnavailable(f"Could not reach the local scripter server at {self._base_url}: {exc}") from exc

        yield response.json()["reply"]
