"""Builds a `RagIndex` from a rendered book's `Transcript`.

Chunk granularity is per-span (see rag/__init__.py's module docstring) — the
transcript's span order defines `seq`, so this must run after `render_book`
has produced `<book>.transcript.json`, but it's independently rerunnable
(e.g. to re-embed after switching embedding models without re-synthesizing
audio).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from alebrijes.rag import RagEntry, RagIndex, _normalize
from alebrijes.transcript import Transcript

EmbedFn = Callable[[str], np.ndarray]


def build_rag_index(
    transcript: Transcript,
    embed_fn: EmbedFn,
    *,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> RagIndex:
    total = len(transcript.spans)
    entries: list[RagEntry] = []
    vectors: list[np.ndarray] = []

    for seq, span in enumerate(transcript.spans):
        vector = _normalize(np.asarray(embed_fn(span.text), dtype=np.float32))
        entries.append(
            RagEntry(
                span_id=span.span_id,
                chapter_index=span.chapter_index,
                paragraph_index=span.paragraph_index,
                seq=seq,
                text=span.text,
            )
        )
        vectors.append(vector)
        if on_progress is not None:
            on_progress(seq + 1, total)

    if not vectors:
        return RagIndex(entries=[], vectors=np.zeros((0, 0), dtype=np.float32))

    return RagIndex(entries=entries, vectors=np.stack(vectors).astype(np.float32))
