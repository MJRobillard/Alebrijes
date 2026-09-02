from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alebrijes.cli import main
from alebrijes.rag import load as load_rag
from alebrijes.segment import SpanKind
from alebrijes.transcript import SpanTiming, Transcript, save as save_transcript


def _fixture_transcript() -> Transcript:
    return Transcript(
        book_title="Fixture Book",
        chapters=[],
        spans=[
            SpanTiming(
                span_id=f"0.{i}.0",
                chapter_index=0,
                paragraph_index=i,
                character="Narrator",
                kind=SpanKind.NARRATION,
                text=f"Span number {i}.",
                start_ms=i * 1000,
                end_ms=(i + 1) * 1000,
            )
            for i in range(3)
        ],
    )


def _fake_embed(self, text: str, *, model: str = "nomic-embed-text") -> np.ndarray:
    return np.array([float(len(text)), 1.0], dtype=np.float32)


def test_rag_build_command_writes_index_derived_from_transcript_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr("alebrijes.llm.ollama.OllamaClient.embed", _fake_embed)

    transcript_path = tmp_path / "book.transcript.json"
    save_transcript(_fixture_transcript(), transcript_path)

    exit_code = main(["rag", "build", "--transcript", str(transcript_path)])

    assert exit_code == 0
    npz_path = tmp_path / "book.rag.npz"
    json_path = tmp_path / "book.rag.json"
    assert npz_path.exists()
    assert json_path.exists()

    index = load_rag(npz_path, json_path)
    assert len(index.entries) == 3
    assert [e.seq for e in index.entries] == [0, 1, 2]
    assert "Wrote RAG index" in capsys.readouterr().out


def test_rag_build_command_respects_explicit_out_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("alebrijes.llm.ollama.OllamaClient.embed", _fake_embed)

    transcript_path = tmp_path / "book.transcript.json"
    save_transcript(_fixture_transcript(), transcript_path)
    out_prefix = tmp_path / "custom"

    main(["rag", "build", "--transcript", str(transcript_path), "--out-prefix", str(out_prefix)])

    assert (tmp_path / "custom.rag.npz").exists()
    assert (tmp_path / "custom.rag.json").exists()
