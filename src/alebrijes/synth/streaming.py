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
from typing import Iterator, Optional, Protocol

import numpy as np

_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]+\s*")


class SynthesizesText(Protocol):
    def synthesize(self, text: str) -> np.ndarray: ...


class SentenceChunker:
    """Incremental sentence splitter: feed it text deltas as they arrive,
    get back each newly-completed sentence.

    The pull-based `stream_response` below is built on this; a caller that
    needs to react to each delta as it's produced (e.g. server.py's chat
    handler, which synthesizes progressively while still consuming a
    single LLM stream) can drive the same buffering logic directly instead.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        sentences: list[str] = []
        while True:
            match = _SENTENCE_BOUNDARY_RE.search(self._buffer)
            if not match:
                break
            sentence = self._buffer[: match.end()].strip()
            self._buffer = self._buffer[match.end() :]
            if sentence:
                sentences.append(sentence)
        return sentences

    def flush(self) -> Optional[str]:
        """Call once the source is exhausted to get any trailing text that
        never reached a sentence boundary (e.g. a response with no closing
        punctuation)."""
        tail = self._buffer.strip()
        self._buffer = ""
        return tail or None


def stream_response(text_chunks: Iterator[str], agent: SynthesizesText) -> Iterator[np.ndarray]:
    """Yields one audio array per sentence as `text_chunks` (LLM token/content
    deltas) are consumed, synthesizing each sentence the moment its boundary
    appears rather than after the whole iterator is exhausted.
    """
    chunker = SentenceChunker()
    for delta in text_chunks:
        for sentence in chunker.feed(delta):
            yield agent.synthesize(sentence)

    tail = chunker.flush()
    if tail:
        yield agent.synthesize(tail)
