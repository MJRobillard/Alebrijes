"""Thin HTTP client for a local Ollama server (chat streaming + embeddings).

No LLM SDK dependency — Ollama's API is plain JSON/NDJSON over HTTP, and
`httpx` is already used elsewhere in this repo (server.py's TestClient).
Interactive narration is meant to degrade gracefully when Ollama isn't
running: every call here raises `OllamaUnavailable` on connection/HTTP
failure rather than letting a raw httpx exception surface, so callers (the
session layer, the server) can catch one clear error type and tell the user
"interactive mode isn't available" instead of crashing.
"""

from __future__ import annotations

import json
import os
from typing import Iterator, Optional

import httpx
import numpy as np

DEFAULT_BASE_URL = "http://localhost:11434"
# Env-var overridable, not hardcoded-only -- so switching to a faster/slower
# local model (e.g. a small model for a CPU-only machine) doesn't need a
# source edit: `set ALEBRIJES_CHAT_MODEL=deepseek-r1:1.5b` (Windows) or
# `export ALEBRIJES_CHAT_MODEL=deepseek-r1:1.5b` (macOS/Linux) before
# launching the app.
DEFAULT_CHAT_MODEL = os.environ.get("ALEBRIJES_CHAT_MODEL", "hermes3:8b")
DEFAULT_EMBED_MODEL = os.environ.get("ALEBRIJES_EMBED_MODEL", "nomic-embed-text")

# Connect fast-fails (Ollama not running should surface as a clear error in
# a few seconds, not hang); reads get up to 60s per chunk -- generous enough
# for CPU-only local inference without leaving a hung request unbounded.
# A timeout here is now caught and surfaced as a clear WebSocket error
# message (server.py's session_ws) rather than crashing the connection.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)


class OllamaUnavailable(Exception):
    """Raised when the local Ollama server can't be reached or errors out."""


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, client: Optional[httpx.Client] = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=_DEFAULT_TIMEOUT)

    def chat_stream(self, messages: list[dict], *, model: str = DEFAULT_CHAT_MODEL) -> Iterator[str]:
        """Yields content deltas as Ollama streams a chat response.

        Ollama's `/api/chat` with `stream: true` returns one JSON object per
        line (NDJSON), each carrying the next content delta; the final line
        has `done: true` and no further content should be emitted after it.
        """
        try:
            with self._client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json={"model": model, "messages": messages, "stream": True},
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done"):
                        return
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(f"Could not reach Ollama at {self._base_url}: {exc}") from exc

    def embed(self, text: str, *, model: str = DEFAULT_EMBED_MODEL) -> np.ndarray:
        try:
            response = self._client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(f"Could not reach Ollama at {self._base_url}: {exc}") from exc

        data = response.json()
        return np.asarray(data["embedding"], dtype=np.float32)

    def list_models(self) -> list[str]:
        """Names of models already pulled locally (`ollama list`'s data),
        for populating a model picker -- not every name is a chat model
        (embedding models show up too), so callers filter as needed."""
        try:
            response = self._client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(f"Could not reach Ollama at {self._base_url}: {exc}") from exc

        data = response.json()
        return [entry["name"] for entry in data.get("models", [])]
