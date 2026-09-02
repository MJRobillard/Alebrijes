from __future__ import annotations

import numpy as np

from alebrijes.synth.streaming import stream_response


class _RecordingAgent:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return np.array([len(text)], dtype=np.int16)


def test_stream_response_synthesizes_as_soon_as_a_sentence_completes() -> None:
    agent = _RecordingAgent()
    # Delivered as separate deltas, the way an LLM streams tokens -- a
    # sentence boundary lands mid-way through the second delta.
    deltas = iter(["Let me think. ", "Colin met Sable. ", "She warned him"])

    chunks = list(stream_response(deltas, agent))

    assert agent.calls == ["Let me think.", "Colin met Sable.", "She warned him"]
    assert len(chunks) == 3


def test_stream_response_yields_first_sentence_before_consuming_later_deltas() -> None:
    # Prove synthesis actually happens incrementally, not only after the
    # whole iterator is exhausted -- the generator must produce its first
    # audio chunk without pulling the never-provided remaining deltas.
    agent = _RecordingAgent()

    def deltas():
        yield "Short opener. "
        raise AssertionError("Should not need to pull past the first sentence to yield it")

    gen = stream_response(deltas(), agent)
    first = next(gen)

    assert agent.calls == ["Short opener."]
    assert first.tolist() == [len("Short opener.")]


def test_stream_response_flushes_trailing_text_with_no_terminal_punctuation() -> None:
    agent = _RecordingAgent()
    deltas = iter(["No terminal punctuation here"])

    list(stream_response(deltas, agent))

    assert agent.calls == ["No terminal punctuation here"]


def test_stream_response_on_empty_input_yields_nothing() -> None:
    agent = _RecordingAgent()

    chunks = list(stream_response(iter([]), agent))

    assert chunks == []
    assert agent.calls == []
