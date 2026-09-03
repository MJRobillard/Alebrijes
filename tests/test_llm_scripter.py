from __future__ import annotations

import json

import httpx
import pytest

from alebrijes.llm.scripter import ScripterClient, ScripterUnavailable


def _client_with_transport(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_chat_stream_sends_latest_user_message_and_yields_reply() -> None:
    seen_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(json.loads(request.content))
        return httpx.Response(200, json={"reply": "hello there"})

    client = ScripterClient(client=_client_with_transport(handler))
    messages = [
        {"role": "system", "content": "You are a narrator."},
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "an earlier reply"},
        {"role": "user", "content": "second turn"},
    ]

    result = list(client.chat_stream(messages))

    assert result == ["hello there"]
    assert seen_payload["message"] == "second turn"
    assert seen_payload["session_id"]  # non-empty, stable per client


def test_chat_stream_extracts_current_paragraph_from_system_message_as_context() -> None:
    seen_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(json.loads(request.content))
        return httpx.Response(200, json={"reply": "ok"})

    client = ScripterClient(client=_client_with_transport(handler))
    system_content = (
        "some guidelines\n\n---\n\nBook: Fixture\nYou are speaking as: Narrator\n"
        "Listener's current position: chapter 1, near: “...”\n\n"
        "Current paragraph (full text):\nColin walked into the garden and paused.\n\n"
        "Retrieved excerpts (bounded to the listener's current position):\n(no relevant excerpts retrieved)\n\n"
        "mode fragment"
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "who is in the garden?"},
    ]

    list(client.chat_stream(messages))

    assert seen_payload["context"] == "Colin walked into the garden and paused."
    assert seen_payload["message"] == "who is in the garden?"


def test_chat_stream_sends_null_context_when_system_message_has_no_paragraph_marker() -> None:
    seen_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(json.loads(request.content))
        return httpx.Response(200, json={"reply": "ok"})

    client = ScripterClient(client=_client_with_transport(handler))
    messages = [
        {"role": "system", "content": "no marker here"},
        {"role": "user", "content": "hi"},
    ]

    list(client.chat_stream(messages))

    assert seen_payload["context"] is None


def test_chat_stream_reuses_the_same_session_id_across_calls() -> None:
    seen_session_ids = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_session_ids.append(json.loads(request.content)["session_id"])
        return httpx.Response(200, json={"reply": "ok"})

    client = ScripterClient(client=_client_with_transport(handler))
    list(client.chat_stream([{"role": "user", "content": "one"}]))
    list(client.chat_stream([{"role": "user", "content": "two"}]))

    assert seen_session_ids[0] == seen_session_ids[1]


def test_chat_stream_raises_scripter_unavailable_on_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = ScripterClient(client=_client_with_transport(handler))

    with pytest.raises(ScripterUnavailable):
        list(client.chat_stream([{"role": "user", "content": "hi"}]))
