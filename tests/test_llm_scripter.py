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
