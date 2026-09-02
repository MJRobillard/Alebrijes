"""Sentence-chunked TTS streaming for interactive-narration responses.

Not a new TTS engine — reuses the session's own `TTSAgent` (same cloned
Narrator voice as the book) so an interrupt answer sounds like the rest of
the audiobook. Responsiveness instead comes from synthesizing each sentence
as soon as the LLM has streamed it, rather than waiting for the whole
response: paired with `llm/narrator_guidelines.md`'s "lead with a short
sentence" instruction, the first audio is ready well before the model has
finished generating.
"""

from __future__ import annotations

import re
from typing import Iterator, Protocol

import numpy as np

_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]+\s*")


class SynthesizesText(Protocol):
    def synthesize(self, text: str) -> np.ndarray: ...


def stream_response(text_chunks: Iterator[str], agent: SynthesizesText) -> Iterator[np.ndarray]:
    """Yields one audio array per sentence as `text_chunks` (LLM token/content
    deltas) are consumed, synthesizing each sentence the moment its boundary
    appears rather than after the whole iterator is exhausted.
    """
    buffer = ""
    for delta in text_chunks:
        buffer += delta
        while True:
            match = _SENTENCE_BOUNDARY_RE.search(buffer)
            if not match:
                break
            sentence = buffer[: match.end()].strip()
            buffer = buffer[match.end() :]
            if sentence:
                yield agent.synthesize(sentence)

    tail = buffer.strip()
    if tail:
        yield agent.synthesize(tail)
