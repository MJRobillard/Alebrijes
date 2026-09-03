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

One exception: the current paragraph. `llm/prompts.py:build_system_prompt`
embeds the listener's full current-position paragraph in the system
message under a "Current paragraph (full text):" marker (separate from
`position_summary`, which is just an 80-char truncated pointer meant for a
generic instruction-following model). This adapter pulls that block back
out and sends it as `context` in the POST body -- the scripter server has
its own RAG over the same underlying book, so grounding it in exactly
where the listener is (not just their raw question) is worth the one
exception to "only forward the latest user turn".

Also unlike Ollama, `/chat` is not a streaming endpoint (`stream: False` is
hardcoded on the server side) -- `chat_stream` still returns an `Iterator[str]`
to match `alebrijes.server.ChatStreamFn`'s contract, it just yields the
whole reply as a single chunk instead of incremental deltas.
"""

from __future__ import annotations

import re
import uuid
from typing import Iterator, Optional

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
MODEL_NAME = "local"

_CURRENT_PARAGRAPH_RE = re.compile(r"Current paragraph \(full text\):\n(.*?)\n\n", re.DOTALL)

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
        system_message = next((m["content"] for m in messages if m.get("role") == "system"), "")
        paragraph_match = _CURRENT_PARAGRAPH_RE.search(system_message)
        context = paragraph_match.group(1) if paragraph_match else None

        try:
            response = self._client.post(
                f"{self._base_url}/chat",
                json={"session_id": self._session_id, "message": latest_user_message, "context": context},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ScripterUnavailable(f"Could not reach the local scripter server at {self._base_url}: {exc}") from exc

        yield response.json()["reply"]
