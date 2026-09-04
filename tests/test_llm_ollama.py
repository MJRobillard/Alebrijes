from __future__ import annotations

import json
import time

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


def test_list_models_returns_model_names() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={"models": [{"name": "hermes3:8b", "details": {"family": "llama"}}]},
        )

    client = _client_with_transport(handler)
    names = client.list_models()

    assert names == ["hermes3:8b"]


def test_list_models_filters_out_embedding_only_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "hermes3:8b", "details": {"family": "llama"}},
                    {"name": "nomic-embed-text:latest", "details": {"family": "nomic-bert"}},
                    {"name": "some-bert-embedder:latest", "details": {"family": "bert"}},
                ]
            },
        )

    client = _client_with_transport(handler)
    names = client.list_models()

    assert names == ["hermes3:8b"]


def test_list_models_keeps_an_unrecognized_family_rather_than_hiding_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": [{"name": "some-new-model:latest", "details": {"family": "some-new-family"}}]},
        )

    client = _client_with_transport(handler)
    names = client.list_models()

    assert names == ["some-new-model:latest"]


def test_list_models_raises_ollama_unavailable_on_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client_with_transport(handler)

    with pytest.raises(OllamaUnavailable):
        client.list_models()


def test_embed_raises_ollama_unavailable_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "model not found"})

    client = _client_with_transport(handler)

    with pytest.raises(OllamaUnavailable):
        client.embed("text")


def test_keep_warm_pings_generate_with_keep_alive() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"done": True})

    client = _client_with_transport(handler)
    handle = client.keep_warm(model="hermes3:8b", interval_s=0.05, keep_alive="5m")
    try:
        deadline = time.monotonic() + 2.0
        while len(calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        handle.stop()

    assert len(calls) >= 2
    assert calls[0] == {"model": "hermes3:8b", "keep_alive": "5m"}


def test_keep_warm_skips_tick_while_a_real_call_is_in_flight() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"done": True})

    client = _client_with_transport(handler)
    client._busy_lock.acquire()  # simulate a chat_stream call already in flight
    try:
        handle = client.keep_warm(interval_s=0.05)
        time.sleep(0.2)
        handle.stop()
    finally:
        client._busy_lock.release()

    assert calls == []  # every heartbeat tick was skipped while the lock was held


def test_keep_warm_stop_ends_the_heartbeat_thread() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"done": True})

    client = _client_with_transport(handler)
    handle = client.keep_warm(interval_s=0.05)
    time.sleep(0.05)
    handle.stop()

    assert not handle._thread.is_alive()


def test_keep_warm_keeps_retrying_when_ollama_is_unreachable() -> None:
    """Ollama not being reachable *yet* (still starting up) is a timing
    issue, not permanent -- the heartbeat must keep trying every interval,
    not die on the first failed ping (see keep_warm's docstring)."""
    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        raise httpx.ConnectError("connection refused", request=request)

    client = _client_with_transport(handler)
    handle = client.keep_warm(interval_s=0.05)
    try:
        deadline = time.monotonic() + 2.0
        while call_count[0] < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        handle.stop()

    assert call_count[0] >= 3  # kept pinging across multiple failures
    assert not handle._thread.is_alive()  # .stop() still ends it cleanly


def test_keep_warm_recovers_once_ollama_becomes_reachable() -> None:
    call_count = [0]
    succeeded = [False]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        if call_count[0] < 3:
            raise httpx.ConnectError("connection refused", request=request)
        succeeded[0] = True
        return httpx.Response(200, json={"done": True})

    client = _client_with_transport(handler)
    handle = client.keep_warm(interval_s=0.05)
    try:
        deadline = time.monotonic() + 2.0
        while not succeeded[0] and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        handle.stop()

    assert succeeded[0]  # a later ping succeeded after earlier failures


def test_chat_stream_holds_busy_lock_for_the_call_and_releases_after() -> None:
    lock_states: list[bool] = []
    client: OllamaClient

    def handler(request: httpx.Request) -> httpx.Response:
        lock_states.append(client._busy_lock.locked())
        content = json.dumps({"message": {"content": "hi"}, "done": True}) + "\n"
        return httpx.Response(200, content=content)

    client = _client_with_transport(handler)
    list(client.chat_stream([{"role": "user", "content": "hi"}]))

    assert lock_states == [True]
    assert not client._busy_lock.locked()


def test_chat_stream_releases_busy_lock_even_on_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client_with_transport(handler)

    with pytest.raises(OllamaUnavailable):
        list(client.chat_stream([{"role": "user", "content": "hi"}]))

    assert not client._busy_lock.locked()
