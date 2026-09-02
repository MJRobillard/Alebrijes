from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from alebrijes.llm.ollama import OllamaClient, OllamaUnavailable


def _client_with_transport(handler) -> OllamaClient:
    transport = httpx.MockTransport(handler)
    return OllamaClient(client=httpx.Client(transport=transport))


def test_chat_stream_yields_content_deltas_from_ndjson() -> None:
    lines = [
        json.dumps({"message": {"content": "Once "}, "done": False}),
        json.dumps({"message": {"content": "upon a time."}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["model"] == "hermes3:8b"
        return httpx.Response(200, content="\n".join(lines) + "\n")

    client = _client_with_transport(handler)
    chunks = list(client.chat_stream([{"role": "user", "content": "hi"}]))

    assert chunks == ["Once ", "upon a time."]


def test_chat_stream_stops_at_done_even_with_trailing_content() -> None:
    lines = [
        json.dumps({"message": {"content": "Hello."}, "done": True}),
        json.dumps({"message": {"content": "Should not appear."}, "done": False}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content="\n".join(lines) + "\n")

    client = _client_with_transport(handler)
    chunks = list(client.chat_stream([{"role": "user", "content": "hi"}]))

    assert chunks == ["Hello."]


def test_chat_stream_raises_ollama_unavailable_on_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client_with_transport(handler)

    with pytest.raises(OllamaUnavailable):
        list(client.chat_stream([{"role": "user", "content": "hi"}]))


def test_embed_returns_float32_array() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embeddings"
        body = json.loads(request.content)
        assert body["model"] == "nomic-embed-text"
        assert body["prompt"] == "some span text"
        return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})

    client = _client_with_transport(handler)
    vector = client.embed("some span text")

    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float32
    assert vector.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_embed_raises_ollama_unavailable_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "model not found"})

    client = _client_with_transport(handler)

    with pytest.raises(OllamaUnavailable):
        client.embed("text")
