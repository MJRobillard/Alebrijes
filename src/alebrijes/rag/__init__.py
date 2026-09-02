"""RAG index: per-span embeddings + position-bounded cosine search.

One book's worth of spans (thousands, not millions) doesn't need an ANN
library — a flat, L2-normalized numpy array and a brute-force dot product
is fast enough and keeps this dependency-cautious repo's zero-vector-DB
footprint (see pyproject.toml). `seq` (a span's 0-based position in reading
order, aligned with `transcript.py`'s span order) is what position-bounded
retrieval filters on: this is the no-spoilers guarantee for interactive
narration — see rag/build.py for how the index gets built and
llm/prompts.py for how search results reach the model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RagEntry:
    span_id: str
    chapter_index: int
    paragraph_index: int
    seq: int
    text: str


@dataclass
class RagIndex:
    entries: list[RagEntry] = field(default_factory=list)
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))


@dataclass
class RagHit:
    entry: RagEntry
    score: float


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def search(index: RagIndex, query_vector: np.ndarray, *, max_seq: int, top_k: int = 5) -> list[RagHit]:
    """Cosine-similarity search over entries with `seq <= max_seq` only.

    Filtering happens before ranking, not after — this is what guarantees a
    retrieval hit can never come from later in the book than the listener's
    current position.
    """
    if not index.entries:
        return []

    mask = np.array([entry.seq <= max_seq for entry in index.entries])
    if not mask.any():
        return []

    query = _normalize(np.asarray(query_vector, dtype=np.float32))
    candidate_vectors = index.vectors[mask]
    candidate_entries = [entry for entry, keep in zip(index.entries, mask) if keep]

    scores = candidate_vectors @ query
    top_k = min(top_k, len(candidate_entries))
    order = np.argsort(scores)[::-1][:top_k]

    return [RagHit(entry=candidate_entries[i], score=float(scores[i])) for i in order]


def save(index: RagIndex, npz_path: Path, json_path: Path) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, vectors=index.vectors)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps([asdict(entry) for entry in index.entries]))


def load(npz_path: Path, json_path: Path) -> RagIndex:
    with np.load(npz_path) as data:
        vectors = data["vectors"]
    entries = [RagEntry(**entry) for entry in json.loads(json_path.read_text())]
    return RagIndex(entries=entries, vectors=vectors)
