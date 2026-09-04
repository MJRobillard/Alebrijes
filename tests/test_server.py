from __future__ import annotations

import base64
import io
import json
import time
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from alebrijes.cast import CastEntry, CastFile, save as save_cast
from alebrijes.ingest import Book, Chapter, Paragraph
from alebrijes.segment import SpanKind
from alebrijes.server import create_app, find_free_port, generate_token
from alebrijes.transcript import ChapterMark, SpanTiming, Transcript, save as save_transcript
from epub_fixtures import build_epub

TOKEN = "test-token"
ORIGIN = "http://127.0.0.1:9999"


def _client(**kwargs) -> TestClient:
    app = create_app(TOKEN, allowed_origin=ORIGIN, **kwargs)
    return TestClient(app)


def _headers(token: str = TOKEN, origin: str | None = ORIGIN) -> dict:
    headers = {"x-alebrijes-token": token}
    if origin is not None:
        headers["origin"] = origin
    return headers


# --- hardening ---------------------------------------------------------


def test_missing_token_is_rejected() -> None:
    client = _client()
    response = client.get("/api/health", headers={"origin": ORIGIN})
    assert response.status_code == 403


# --- /api/llm/models ---------------------------------------------------


def test_llm_models_returns_names_and_default() -> None:
    client = _client(list_models_fn=lambda: ["hermes3:8b", "deepseek-r1:1.5b"])
    response = client.get("/api/llm/models", headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["models"] == ["local", "hermes3:8b", "deepseek-r1:1.5b"]
    assert body["default"]  # hermes3:8b unless ALEBRIJES_CHAT_MODEL is set


def test_llm_models_surfaces_ollama_unavailable_as_502() -> None:
    from alebrijes.llm.ollama import OllamaUnavailable

    def _raise():
        raise OllamaUnavailable("no connection")

    client = _client(list_models_fn=_raise)
    response = client.get("/api/llm/models", headers=_headers())

    assert response.status_code == 502


def test_wrong_token_is_rejected() -> None:
    client = _client()
    response = client.get("/api/health", headers=_headers(token="wrong"))
    assert response.status_code == 403


def test_correct_token_and_origin_is_allowed() -> None:
    client = _client()
    response = client.get("/api/health", headers=_headers())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "warnings" in body


def test_wrong_origin_is_rejected_even_with_correct_token() -> None:
    client = _client()
    response = client.get("/api/health", headers=_headers(origin="http://evil.example"))
    assert response.status_code == 403


def test_missing_origin_is_allowed() -> None:
    # Browsers always set Origin on cross-origin requests -- the exact case
    # this defends against -- so a request with no Origin at all can't be an
    # attacker's cross-origin fetch; requiring it unconditionally would only
    # break legitimate same-origin requests that omit it.
    client = _client()
    response = client.get("/api/health", headers=_headers(origin=None))
    assert response.status_code == 200


def test_find_free_port_returns_a_usable_port() -> None:
    port = find_free_port()
    assert 1024 < port < 65536


def test_generate_token_is_random_each_call() -> None:
    assert generate_token() != generate_token()


# --- frontend index route ----------------------------------------------
#
# Deliberately unauthenticated (the page has to load before it can present a
# token), so the important thing to verify is that it never leaks the
# server's own secret token into that unauthenticated response.


def test_index_route_requires_no_token() -> None:
    client = _client()
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower()


def test_index_route_does_not_leak_the_server_token() -> None:
    client = _client()
    response = client.get("/")
    assert TOKEN not in response.text


# --- /api/voices ---------------------------------------------------------


def test_voices_endpoint_returns_builtin_voices_when_no_clones_exist(tmp_path: Path) -> None:
    client = _client(voices_dir=tmp_path / "empty")
    response = client.get("/api/voices", headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert "alba" in body["builtin"]
    assert body["cloned"] == []


def test_voices_endpoint_lists_cloned_voices_from_directory(tmp_path: Path) -> None:
    (tmp_path / "narrator_clone.pt").write_bytes(b"fake-state")
    (tmp_path / "villain_clone.pt").write_bytes(b"fake-state")

    client = _client(voices_dir=tmp_path)
    response = client.get("/api/voices", headers=_headers())

    body = response.json()
    names = {entry["name"] for entry in body["cloned"]}
    assert names == {"narrator_clone", "villain_clone"}
    assert all(Path(entry["voice_state_path"]).exists() for entry in body["cloned"])


def test_voices_endpoint_requires_token(tmp_path: Path) -> None:
    client = _client(voices_dir=tmp_path)
    response = client.get("/api/voices")
    assert response.status_code == 403


def test_add_voice_imports_and_returns_updated_voice_list(tmp_path: Path) -> None:
    source = tmp_path / "narrator.pt"
    source.write_bytes(b"fake voice state")
    voices_dir = tmp_path / "voices"

    calls = []

    def fake_import(source_path: Path, name: str, voices_dir_arg: Path) -> Path:
        calls.append((source_path, name, voices_dir_arg))
        voices_dir_arg.mkdir(parents=True, exist_ok=True)
        out = voices_dir_arg / f"{name}.pt"
        out.write_bytes(b"fake-state")
        return out

    client = _client(voices_dir=voices_dir, import_voice_fn=fake_import)
    response = client.post(
        "/api/voices",
        headers=_headers(),
        json={"source": str(source), "name": "narrator_import"},
    )

    assert response.status_code == 200
    body = response.json()
    names = {c["name"] for c in body["cloned"]}
    assert "narrator_import" in names
    assert calls == [(source, "narrator_import", voices_dir)]


def test_add_voice_rejects_missing_source_file(tmp_path: Path) -> None:
    client = _client(voices_dir=tmp_path)
    response = client.post(
        "/api/voices", headers=_headers(), json={"source": str(tmp_path / "missing.pt"), "name": "test"}
    )
    assert response.status_code == 400


def test_add_voice_rejects_non_pt_extension(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"not a voice state")

    client = _client(voices_dir=tmp_path)
    response = client.post("/api/voices", headers=_headers(), json={"source": str(source), "name": "test"})

    assert response.status_code == 400
    assert "voice-state file" in response.json()["detail"]


def test_add_voice_rejects_blank_name(tmp_path: Path) -> None:
    source = tmp_path / "narrator.pt"
    source.write_bytes(b"fake voice state")

    client = _client(voices_dir=tmp_path)
    response = client.post("/api/voices", headers=_headers(), json={"source": str(source), "name": "   "})

    assert response.status_code == 400
    assert "name is required" in response.json()["detail"]


def test_add_voice_reports_import_errors_clearly(tmp_path: Path) -> None:
    source = tmp_path / "narrator.pt"
    source.write_bytes(b"fake voice state")

    def failing_import(*args, **kwargs) -> Path:
        raise RuntimeError("not a valid torch file")

    client = _client(voices_dir=tmp_path, import_voice_fn=failing_import)
    response = client.post("/api/voices", headers=_headers(), json={"source": str(source), "name": "test"})

    assert response.status_code == 400
    assert "not a valid torch file" in response.json()["detail"]


def test_default_import_voice_rejects_a_file_that_isnt_a_real_torch_state(tmp_path: Path) -> None:
    # Uses the real (non-injected) default import function, not a fake --
    # proves the validation actually works, not just that the route wires
    # errors through correctly.
    source = tmp_path / "fake.pt"
    source.write_bytes(b"this is not a torch state file")

    client = _client(voices_dir=tmp_path / "voices")  # default import_voice_fn
    response = client.post("/api/voices", headers=_headers(), json={"source": str(source), "name": "test"})

    assert response.status_code == 400
    assert "doesn't look like a valid voice-state file" in response.json()["detail"]


def test_default_import_voice_accepts_a_real_torch_state(tmp_path: Path) -> None:
    import torch

    source = tmp_path / "real.pt"
    torch.save({"conditioning": "alba"}, source)
    voices_dir = tmp_path / "voices"

    client = _client(voices_dir=voices_dir)  # default import_voice_fn
    response = client.post("/api/voices", headers=_headers(), json={"source": str(source), "name": "narrator"})

    assert response.status_code == 200
    imported = voices_dir / "narrator.pt"
    assert imported.exists()
    assert torch.load(imported, weights_only=False) == {"conditioning": "alba"}


def test_default_import_voice_accepts_a_safetensors_file(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    source = tmp_path / "real.safetensors"
    save_file({"conditioning": torch.tensor([1.0, 2.0, 3.0])}, str(source))
    voices_dir = tmp_path / "voices"

    client = _client(voices_dir=voices_dir)  # default import_voice_fn
    response = client.post("/api/voices", headers=_headers(), json={"source": str(source), "name": "narrator"})

    assert response.status_code == 200
    imported = voices_dir / "narrator.pt"
    assert imported.exists()
    assert torch.equal(torch.load(imported, weights_only=False)["conditioning"], torch.tensor([1.0, 2.0, 3.0]))


# --- /api/preview ----------------------------------------------------------


def test_preview_endpoint_returns_audio_bytes_from_injected_synth_fn() -> None:
    calls = []

    def fake_synthesize(text: str, voice_state_path: str) -> bytes:
        calls.append((text, voice_state_path))
        return b"RIFF-fake-wav-bytes"

    client = _client(synthesize_preview_fn=fake_synthesize)
    response = client.post(
        "/api/preview", headers=_headers(), json={"text": "Hello there.", "voice_state_path": "alba"}
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFF-fake-wav-bytes"
    assert calls == [("Hello there.", "alba")]


def test_preview_endpoint_rejects_overly_long_text() -> None:
    client = _client(synthesize_preview_fn=lambda text, voice: b"")
    response = client.post(
        "/api/preview", headers=_headers(), json={"text": "x" * 501, "voice_state_path": "alba"}
    )
    assert response.status_code == 400


def test_preview_endpoint_reports_synthesis_errors_clearly() -> None:
    def failing_synthesize(text: str, voice_state_path: str) -> bytes:
        raise RuntimeError("no such voice")

    client = _client(synthesize_preview_fn=failing_synthesize)
    response = client.post(
        "/api/preview", headers=_headers(), json={"text": "Hi.", "voice_state_path": "nonexistent"}
    )

    assert response.status_code == 400
    assert "no such voice" in response.json()["detail"]


def test_preview_endpoint_requires_token() -> None:
    client = _client(synthesize_preview_fn=lambda text, voice: b"")
    response = client.post("/api/preview", json={"text": "Hi.", "voice_state_path": "alba"})
    assert response.status_code == 403


# --- /api/cast -----------------------------------------------------------


def test_cast_endpoint_creates_cast_file_and_returns_characters(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    build_epub(
        epub_path,
        "Fixture Book",
        {"ch1.xhtml": '<p>Chapter 1</p><p>Colin said, "Let\'s take another path."</p>'},
    )

    client = _client()
    response = client.post("/api/cast", headers=_headers(), json={"book": str(epub_path)})

    assert response.status_code == 200
    body = response.json()
    assert Path(body["cast_path"]).exists()
    characters = [entry["character"] for entry in body["entries"]]
    assert "Colin" in characters
    assert "Narrator" in characters
    assert all(entry["voice_state_path"] for entry in body["entries"])


def test_cast_endpoint_rejects_unsupported_format(tmp_path: Path) -> None:
    unsupported = tmp_path / "book.docx"
    unsupported.write_text("not supported")

    client = _client()
    response = client.post("/api/cast", headers=_headers(), json={"book": str(unsupported)})

    assert response.status_code == 422


def test_update_cast_endpoint_persists_edited_voices(tmp_path: Path) -> None:
    cast_path = tmp_path / "book.cast.yaml"
    save_cast(
        CastFile(book_title="Fixture", entries=[CastEntry(character="Narrator", voice_state_path="alba")]),
        cast_path,
    )

    client = _client()
    response = client.put(
        "/api/cast",
        headers=_headers(),
        json={"cast_path": str(cast_path), "entries": [{"character": "Narrator", "voice_state_path": "custom.pt"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["entries"] == [{"character": "Narrator", "voice_state_path": "custom.pt"}]
    from alebrijes.cast import load as load_cast

    assert load_cast(cast_path).voice_for("Narrator") == "custom.pt"


def test_update_cast_endpoint_rejects_missing_cast_file(tmp_path: Path) -> None:
    client = _client()
    response = client.put(
        "/api/cast",
        headers=_headers(),
        json={"cast_path": str(tmp_path / "missing.yaml"), "entries": []},
    )
    assert response.status_code == 400


def test_cast_endpoint_rejects_missing_book(tmp_path: Path) -> None:
    client = _client()
    response = client.post(
        "/api/cast", headers=_headers(), json={"book": str(tmp_path / "missing.epub")}
    )

    assert response.status_code == 400


# --- /api/render -----------------------------------------------------------


def _fake_render_book(book: Book, cast_file: CastFile, output_path: Path, *, pause_ms, on_chapter_done=None, model=None) -> Path:
    total = len(book.chapters)
    for i in range(total):
        time.sleep(0.01)
        if on_chapter_done is not None:
            on_chapter_done(i + 1, total)
    output_path.write_bytes(b"fake-m4b")
    return output_path


def _fixture_book() -> Book:
    return Book(
        title="Fixture",
        chapters=[
            Chapter(title="One", index=0, paragraphs=[Paragraph(text="Some text.", index=0)]),
            Chapter(title="Two", index=1, paragraphs=[Paragraph(text="More text.", index=0)]),
        ],
    )


def test_render_endpoint_starts_job_and_reports_progress_to_completion(tmp_path: Path) -> None:
    cast_file = CastFile(book_title="Fixture", entries=[CastEntry(character="Narrator", voice_state_path="alba")])
    cast_path = tmp_path / "cast.yaml"
    save_cast(cast_file, cast_path)
    book_path = tmp_path / "book.epub"
    book_path.write_bytes(b"unused, ingest_book is overridden")
    out_path = tmp_path / "book.m4b"

    client = _client(ingest_book=lambda path: _fixture_book(), render_fn=_fake_render_book)

    start = client.post(
        "/api/render",
        headers=_headers(),
        json={"book": str(book_path), "cast": str(cast_path), "out": str(out_path)},
    )
    assert start.status_code == 200
    job_id = start.json()["job_id"]

    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/render/{job_id}", headers=_headers()).json()
        if status["status"] != "running":
            break
        time.sleep(0.02)

    assert status["status"] == "done"
    assert status["chapters_done"] == 2
    assert status["chapters_total"] == 2
    assert status["chapter_titles"] == ["One", "Two"]
    assert status["elapsed_seconds"] > 0
    assert status["output_path"] == str(out_path)
    assert out_path.read_bytes() == b"fake-m4b"


def test_render_endpoint_chapter_range_only_renders_selected_chapters(tmp_path: Path) -> None:
    cast_file = CastFile(book_title="Fixture", entries=[CastEntry(character="Narrator", voice_state_path="alba")])
    cast_path = tmp_path / "cast.yaml"
    save_cast(cast_file, cast_path)
    book_path = tmp_path / "book.epub"
    book_path.write_bytes(b"unused, ingest_book is overridden")
    out_path = tmp_path / "book.m4b"

    seen_books: list[Book] = []

    def _recording_render(book, cast_file, output_path, *, pause_ms, on_chapter_done=None, model=None):
        seen_books.append(book)
        output_path.write_bytes(b"fake-m4b")
        return output_path

    client = _client(ingest_book=lambda path: _fixture_book(), render_fn=_recording_render)

    start = client.post(
        "/api/render",
        headers=_headers(),
        json={"book": str(book_path), "cast": str(cast_path), "out": str(out_path), "chapter_start": 2, "chapter_end": 2},
    )
    assert start.status_code == 200
    job_id = start.json()["job_id"]

    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/render/{job_id}", headers=_headers()).json()
        if status["status"] != "running":
            break
        time.sleep(0.02)

    assert status["status"] == "done"
    assert len(seen_books) == 1
    assert [c.title for c in seen_books[0].chapters] == ["Two"]


def test_render_endpoint_elapsed_time_freezes_after_completion(tmp_path: Path) -> None:
    cast_file = CastFile(book_title="Fixture", entries=[CastEntry(character="Narrator", voice_state_path="alba")])
    cast_path = tmp_path / "cast.yaml"
    save_cast(cast_file, cast_path)
    book_path = tmp_path / "book.epub"
    book_path.write_bytes(b"unused")

    client = _client(ingest_book=lambda path: _fixture_book(), render_fn=_fake_render_book)
    start = client.post(
        "/api/render", headers=_headers(), json={"book": str(book_path), "cast": str(cast_path)}
    )
    job_id = start.json()["job_id"]

    deadline = time.time() + 5
    while time.time() < deadline:
        status = client.get(f"/api/render/{job_id}", headers=_headers()).json()
        if status["status"] != "running":
            break
        time.sleep(0.02)

    first_elapsed = client.get(f"/api/render/{job_id}", headers=_headers()).json()["elapsed_seconds"]
    time.sleep(0.05)
    second_elapsed = client.get(f"/api/render/{job_id}", headers=_headers()).json()["elapsed_seconds"]

    assert first_elapsed == second_elapsed


def test_render_endpoint_reports_error_status_on_failure(tmp_path: Path) -> None:
    def _failing_render(*args, **kwargs) -> Path:
        raise RuntimeError("synthesis exploded")

    cast_file = CastFile(book_title="Fixture", entries=[CastEntry(character="Narrator", voice_state_path="alba")])
    cast_path = tmp_path / "cast.yaml"
    save_cast(cast_file, cast_path)
    book_path = tmp_path / "book.epub"
    book_path.write_bytes(b"unused")

    client = _client(ingest_book=lambda path: _fixture_book(), render_fn=_failing_render)

    start = client.post(
        "/api/render", headers=_headers(), json={"book": str(book_path), "cast": str(cast_path)}
    )
    job_id = start.json()["job_id"]

    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/render/{job_id}", headers=_headers()).json()
        if status["status"] != "running":
            break
        time.sleep(0.02)

    assert status["status"] == "error"
    assert "synthesis exploded" in status["error"]


def test_render_status_for_unknown_job_id_is_404() -> None:
    client = _client()
    response = client.get("/api/render/does-not-exist", headers=_headers())
    assert response.status_code == 404


# --- /api/book/transcript, /api/book/audio, /listen (Phase A player) ------


def _fixture_transcript() -> Transcript:
    return Transcript(
        book_title="Fixture",
        chapters=[ChapterMark(index=0, title="One", start_ms=0, end_ms=1000)],
        spans=[
            SpanTiming(
                span_id="0.0.0",
                chapter_index=0,
                paragraph_index=0,
                character="Narrator",
                kind=SpanKind.NARRATION,
                text="Some text.",
                start_ms=0,
                end_ms=1000,
            )
        ],
    )


def test_book_transcript_endpoint_derives_path_from_m4b_and_returns_json(tmp_path: Path) -> None:
    m4b_path = tmp_path / "book.m4b"
    save_transcript(_fixture_transcript(), m4b_path.with_suffix(".transcript.json"))

    client = _client()
    response = client.get(
        "/api/book/transcript", headers=_headers(), params={"path": str(m4b_path)}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["book_title"] == "Fixture"
    assert body["spans"][0]["span_id"] == "0.0.0"


def test_book_transcript_endpoint_404s_when_sidecar_missing(tmp_path: Path) -> None:
    client = _client()
    response = client.get(
        "/api/book/transcript", headers=_headers(), params={"path": str(tmp_path / "missing.m4b")}
    )
    assert response.status_code == 404


def test_book_audio_endpoint_returns_file_bytes(tmp_path: Path) -> None:
    m4b_path = tmp_path / "book.m4b"
    m4b_path.write_bytes(b"fake-m4b-bytes")

    client = _client()
    response = client.get("/api/book/audio", headers=_headers(), params={"path": str(m4b_path)})

    assert response.status_code == 200
    assert response.content == b"fake-m4b-bytes"


def test_book_audio_endpoint_404s_when_file_missing(tmp_path: Path) -> None:
    client = _client()
    response = client.get(
        "/api/book/audio", headers=_headers(), params={"path": str(tmp_path / "missing.m4b")}
    )
    assert response.status_code == 404


def test_listen_route_is_unauthenticated_static_shell() -> None:
    client = _client()
    response = client.get("/listen")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_quiz_route_is_unauthenticated_static_shell() -> None:
    client = _client()
    response = client.get("/quiz")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def _write_fillibuster_fixture(
    directory: Path, fillibuster_id: str, text: str, *, title: str | None = None, description: str = ""
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{fillibuster_id}.txt").write_text(text, encoding="utf-8")
    (directory / f"{fillibuster_id}.yaml").write_text(
        f"title: {title or fillibuster_id}\ndescription: {description}\n", encoding="utf-8"
    )


def test_fillibusters_endpoint_lists_library_metadata(tmp_path: Path) -> None:
    fillibusters_dir = tmp_path / "fillibusters"
    _write_fillibuster_fixture(fillibusters_dir, "a", "Filler A.", title="A Title", description="About A")
    _write_fillibuster_fixture(fillibusters_dir, "b", "Filler B.", title="B Title", description="About B")

    client = _client(fillibusters_dir=fillibusters_dir)
    response = client.get("/api/fillibusters", headers=_headers())

    assert response.status_code == 200
    assert response.json() == {
        "fillibusters": [
            {"id": "a", "title": "A Title", "description": "About A"},
            {"id": "b", "title": "B Title", "description": "About B"},
        ]
    }


def test_fillibusters_endpoint_empty_when_no_scripts_configured(tmp_path: Path) -> None:
    client = _client(fillibusters_dir=tmp_path / "does-not-exist")
    response = client.get("/api/fillibusters", headers=_headers())

    assert response.status_code == 200
    assert response.json() == {"fillibusters": []}


# --- /api/rag/build ---------------------------------------------------------


def _fixture_transcript_for_rag() -> Transcript:
    return Transcript(
        book_title="Fixture",
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


def _fake_embed_fn(text: str):
    import numpy as np

    return np.array([float(len(text)), 1.0], dtype=np.float32)


def test_rag_build_endpoint_starts_job_and_writes_index(tmp_path: Path) -> None:
    transcript_path = tmp_path / "book.transcript.json"
    save_transcript(_fixture_transcript_for_rag(), transcript_path)

    client = _client(embed_fn=_fake_embed_fn)
    start = client.post(
        "/api/rag/build", headers=_headers(), json={"transcript": str(transcript_path)}
    )
    assert start.status_code == 200
    job_id = start.json()["job_id"]

    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/rag/build/{job_id}", headers=_headers()).json()
        if status["status"] != "running":
            break
        time.sleep(0.02)

    assert status["status"] == "done"
    assert status["spans_done"] == 3
    assert status["spans_total"] == 3
    assert Path(status["output_prefix"] + ".rag.npz").exists()
    assert Path(status["output_prefix"] + ".rag.json").exists()


def test_rag_build_status_for_unknown_job_id_is_404() -> None:
    client = _client()
    response = client.get("/api/rag/build/does-not-exist", headers=_headers())
    assert response.status_code == 404


# --- /api/session/ask --------------------------------------------------------


def _fake_chat_stream(messages, *, model="hermes3:8b"):
    # Echo the last user message content back, split into a couple of
    # "streamed" chunks, so tests can assert on assembled output without
    # a real Ollama server.
    last_user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    yield "Answer: "
    yield last_user


def test_session_ask_endpoint_returns_grounded_answer(tmp_path: Path) -> None:
    from alebrijes.rag import RagIndex, save as save_rag
    from alebrijes.rag.build import build_rag_index

    transcript = _fixture_transcript_for_rag()
    index = build_rag_index(transcript, _fake_embed_fn)
    npz_path = tmp_path / "book.rag.npz"
    json_path = tmp_path / "book.rag.json"
    save_rag(index, npz_path, json_path)

    client = _client(embed_fn=_fake_embed_fn, chat_stream_fn=_fake_chat_stream)
    response = client.post(
        "/api/session/ask",
        headers=_headers(),
        json={
            "book_title": "Fixture",
            "position_summary": "partway through",
            "rag_npz": str(npz_path),
            "rag_json": str(json_path),
            "current_seq": 1,
            "question": "what's happening?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Answer: what's happening?"
    # Position-bounded: only spans with seq <= 1 may be cited as sources.
    assert all(sid.split(".")[1] in ("0", "1") for sid in body["sources"])


# --- /api/session/start, /api/session/{id}/ws --------------------------------


class _FakeSynthAgent:
    sample_rate = 8000

    def __init__(self):
        self.voice_calls: list[tuple[str, object]] = []

    def use_voice_state(self, voice, state) -> None:
        self.voice_calls.append((voice, state))

    def synthesize(self, text: str):
        import numpy as np

        return np.ones(100, dtype=np.int16)

    def stream_chunks(self, text: str):
        # One chunk per fake call -- real TTSAgent.stream_chunks may yield
        # several per sentence, but the tests here only assert on aggregate
        # audio_chunk behavior (character/text), not chunk count.
        yield self.synthesize(text)


def _setup_session_fixture(tmp_path: Path):
    transcript = _fixture_transcript_for_rag()
    m4b_path = tmp_path / "book.m4b"
    save_transcript(transcript, m4b_path.with_suffix(".transcript.json"))

    from alebrijes.rag.build import build_rag_index
    from alebrijes.rag import save as save_rag

    index = build_rag_index(transcript, _fake_embed_fn)
    save_rag(index, m4b_path.with_suffix(".rag.npz"), m4b_path.with_suffix(".rag.json"))

    cast_path = tmp_path / "book.cast.yaml"
    save_cast(CastFile(book_title="Fixture", entries=[CastEntry(character="Narrator", voice_state_path="alba")]), cast_path)

    return m4b_path, cast_path


def test_session_start_requires_existing_transcript_and_rag_index(tmp_path: Path) -> None:
    cast_path = tmp_path / "book.cast.yaml"
    save_cast(CastFile(book_title="Fixture", entries=[]), cast_path)

    client = _client()
    response = client.post(
        "/api/session/start",
        headers=_headers(),
        json={"m4b": str(tmp_path / "missing.m4b"), "cast": str(cast_path)},
    )
    assert response.status_code == 400


def test_session_start_reports_spoken_true_when_narrator_agent_builds(tmp_path: Path) -> None:
    m4b_path, cast_path = _setup_session_fixture(tmp_path)

    client = _client(
        embed_fn=_fake_embed_fn,
        chat_stream_fn=_fake_chat_stream,
        build_narrator_agent_fn=lambda cast_file: _FakeSynthAgent(),
    )
    response = client.post(
        "/api/session/start", headers=_headers(), json={"m4b": str(m4b_path), "cast": str(cast_path)}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["spoken"] is True
    assert body["session_id"]


def test_session_start_with_chat_model_pins_it_for_that_sessions_chat_calls(tmp_path: Path) -> None:
    m4b_path, cast_path = _setup_session_fixture(tmp_path)
    seen_models = []

    def chat_stream_fn(messages, *, model="hermes3:8b"):
        seen_models.append(model)
        yield "ok"

    client = _client(embed_fn=_fake_embed_fn, chat_stream_fn=chat_stream_fn)
    start = client.post(
        "/api/session/start",
        headers=_headers(),
        json={"m4b": str(m4b_path), "cast": str(cast_path), "chat_model": "deepseek-r1:1.5b"},
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        ws.send_json({"type": "chat", "text": "hi"})
        ws.receive_json()

    assert seen_models == ["deepseek-r1:1.5b"]



def test_ws_chat_streams_filler_audio_while_waiting_on_a_slow_reply(tmp_path: Path) -> None:
    m4b_path, cast_path = _setup_session_fixture(tmp_path)
    fillibusters_dir = tmp_path / "fillibusters"
    _write_fillibuster_fixture(fillibusters_dir, "a", "Filler sentence one.")

    def slow_chat_stream(messages, *, model="hermes3:8b"):
        time.sleep(0.7)  # comfortably past _FILLER_START_DELAY_S (0.4s)
        yield "Real reply."

    client = _client(
        embed_fn=_fake_embed_fn,
        chat_stream_fn=slow_chat_stream,
        build_narrator_agent_fn=lambda cast_file: _FakeSynthAgent(),
        fillibusters_dir=fillibusters_dir,
    )
    start = client.post(
        "/api/session/start",
        headers=_headers(),
        json={"m4b": str(m4b_path), "cast": str(cast_path), "fillibuster_ids": ["a"]},
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        ws.send_json({"type": "chat", "text": "hi"})
        received = []
        while True:
            msg = ws.receive_json()
            received.append(msg)
            if msg["type"] == "answer":
                break

    types = [m["type"] for m in received]
    assert "filler_audio_chunk" in types
    assert types[-1] == "answer"
    # Once real audio starts, filler must not resume -- it's a stand-in for
    # the wait, not something that plays alongside the real reply.
    first_real_audio = types.index("audio_chunk")
    assert "filler_audio_chunk" not in types[first_real_audio:]


class _TextVaryingSynthAgent:
    """Like _FakeSynthAgent, but its output actually depends on the input
    text (length-keyed) -- proves distinct fillibuster sentences really get
    synthesized and sent, not the same clip repeated."""

    sample_rate = 8000

    def use_voice_state(self, voice, state) -> None:
        pass

    def synthesize(self, text: str):
        import numpy as np

        return np.full(50, len(text) % 256, dtype=np.int16)

    def stream_chunks(self, text: str):
        yield self.synthesize(text)


def test_ws_chat_streams_multiple_distinct_fillibuster_sentences_back_to_back(tmp_path: Path) -> None:
    """Regression test for the one-ahead prefetch restructure in
    _filler_loop (mirrors LookaheadRenderer/_drain_chapters's "keep the
    next one ready" pattern) -- with a fillibuster script long enough for
    several sentences and a long enough wait, playback should read through
    them in order, not stall or repeat the same sentence's synthesis
    forever."""
    m4b_path, cast_path = _setup_session_fixture(tmp_path)
    fillibusters_dir = tmp_path / "fillibusters"
    _write_fillibuster_fixture(
        fillibusters_dir,
        "a",
        "First filler sentence here. Second filler sentence, noticeably longer than the first one.",
    )

    def slow_chat_stream(messages, *, model="hermes3:8b"):
        time.sleep(1.0)  # comfortably past a few filler rotations
        yield "Real reply."

    client = _client(
        embed_fn=_fake_embed_fn,
        chat_stream_fn=slow_chat_stream,
        build_narrator_agent_fn=lambda cast_file: _TextVaryingSynthAgent(),
        fillibusters_dir=fillibusters_dir,
    )
    start = client.post(
        "/api/session/start",
        headers=_headers(),
        json={"m4b": str(m4b_path), "cast": str(cast_path), "fillibuster_ids": ["a"]},
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        ws.send_json({"type": "chat", "text": "hi"})
        received = []
        while True:
            msg = ws.receive_json()
            received.append(msg)
            if msg["type"] == "answer":
                break

    filler_payloads = [m["data"] for m in received if m["type"] == "filler_audio_chunk"]
    assert len(filler_payloads) >= 2
    assert len(set(filler_payloads)) >= 2  # both distinct sentences actually streamed, not one repeated


class _FakeKeepWarmHandle:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _FakeOllamaClient:
    """Stands in for `alebrijes.llm.ollama.OllamaClient` so a test can
    assert on `.keep_warm()`/`.stop()` calls without a real Ollama server.
    `create_app` imports `OllamaClient` from `alebrijes.llm.ollama` inside
    its own body (a fresh lookup on every call), so monkeypatching the name
    in that module is enough to substitute this in."""

    instances: list["_FakeOllamaClient"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.keep_warm_calls: list[_FakeKeepWarmHandle] = []
        _FakeOllamaClient.instances.append(self)

    def chat_stream(self, messages, *, model="hermes3:8b"):
        yield "ok"

    def embed(self, text: str):
        import numpy as np

        return np.zeros(2, dtype=np.float32)

    def list_models(self) -> list[str]:
        return []

    def keep_warm(self, *args, **kwargs) -> _FakeKeepWarmHandle:
        handle = _FakeKeepWarmHandle()
        self.keep_warm_calls.append(handle)
        return handle


def test_ws_keeps_the_real_chat_model_warm_for_the_connections_lifetime(tmp_path, monkeypatch) -> None:
    m4b_path, cast_path = _setup_session_fixture(tmp_path)
    _FakeOllamaClient.instances = []
    monkeypatch.setattr("alebrijes.llm.ollama.OllamaClient", _FakeOllamaClient)

    # Only chat_stream_fn is left to create_app's default (real-client)
    # branch, so exactly one _FakeOllamaClient instance -- the chat one --
    # gets created.
    client = _client(embed_fn=_fake_embed_fn, list_models_fn=lambda: [])
    start = client.post(
        "/api/session/start", headers=_headers(), json={"m4b": str(m4b_path), "cast": str(cast_path)}
    )
    session_id = start.json()["session_id"]

    chat_client = _FakeOllamaClient.instances[0]
    assert chat_client.keep_warm_calls == []

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        assert len(chat_client.keep_warm_calls) == 1
        handle = chat_client.keep_warm_calls[0]
        assert handle.stopped is False

        # Sent as soon as the connection opens, before any client message --
        # this is what lets the frontend's pulse dot initialize correctly.
        state_msg = ws.receive_json()
        assert state_msg == {"type": "keep_warm_state", "available": True, "enabled": True}

        ws.send_json({"type": "chat", "text": "hi"})
        ws.receive_json()

    assert handle.stopped is True


def test_ws_keep_warm_toggle_stops_and_restarts_the_heartbeat(tmp_path, monkeypatch) -> None:
    m4b_path, cast_path = _setup_session_fixture(tmp_path)
    _FakeOllamaClient.instances = []
    monkeypatch.setattr("alebrijes.llm.ollama.OllamaClient", _FakeOllamaClient)

    client = _client(embed_fn=_fake_embed_fn, list_models_fn=lambda: [])
    start = client.post(
        "/api/session/start", headers=_headers(), json={"m4b": str(m4b_path), "cast": str(cast_path)}
    )
    session_id = start.json()["session_id"]
    chat_client = _FakeOllamaClient.instances[0]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        initial = ws.receive_json()
        assert initial == {"type": "keep_warm_state", "available": True, "enabled": True}
        first_handle = chat_client.keep_warm_calls[0]

        ws.send_json({"type": "keep_warm_toggle", "enabled": False})
        off_state = ws.receive_json()
        assert off_state == {"type": "keep_warm_state", "available": True, "enabled": False}
        assert first_handle.stopped is True
        assert len(chat_client.keep_warm_calls) == 1  # no new heartbeat started by turning off

        ws.send_json({"type": "keep_warm_toggle", "enabled": True})
        on_state = ws.receive_json()
        assert on_state == {"type": "keep_warm_state", "available": True, "enabled": True}
        assert len(chat_client.keep_warm_calls) == 2  # a fresh heartbeat started
        second_handle = chat_client.keep_warm_calls[1]
        assert second_handle.stopped is False


def test_ws_does_not_keep_warm_when_session_is_pinned_to_the_local_scripter_model(tmp_path, monkeypatch) -> None:
    m4b_path, cast_path = _setup_session_fixture(tmp_path)
    _FakeOllamaClient.instances = []
    monkeypatch.setattr("alebrijes.llm.ollama.OllamaClient", _FakeOllamaClient)
    monkeypatch.setattr(
        "alebrijes.llm.scripter.ScripterClient.chat_stream",
        lambda self, messages: iter(["ok"]),
    )

    client = _client(embed_fn=_fake_embed_fn, list_models_fn=lambda: [])
    start = client.post(
        "/api/session/start",
        headers=_headers(),
        json={"m4b": str(m4b_path), "cast": str(cast_path), "chat_model": "local"},
    )
    session_id = start.json()["session_id"]

    chat_client = _FakeOllamaClient.instances[0]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        ws.send_json({"type": "chat", "text": "hi"})
        # No initial keep_warm_state frame for an unavailable session -- the
        # first message is real response content (audio_chunk or answer,
        # depending on whether a narrator agent built), never a status frame.
        first = ws.receive_json()
        assert first["type"] != "keep_warm_state"

    assert chat_client.keep_warm_calls == []


def test_pick_voice_for_text_returns_earliest_named_character() -> None:
    from alebrijes.server import _pick_voice_for_text

    cast_file = CastFile(
        book_title="Fixture",
        entries=[
            CastEntry(character="Narrator", voice_state_path="alba"),
            CastEntry(character="Felicity", voice_state_path="fantine"),
            CastEntry(character="Sara", voice_state_path="cosette"),
        ],
    )
    voice_states = {"Narrator": ("alba", "n"), "Felicity": ("fantine", "f"), "Sara": ("cosette", "s")}

    result = _pick_voice_for_text("Sara turns to Felicity and smiles.", cast_file, voice_states)

    assert result == ("Sara", ("cosette", "s"))


def test_pick_voice_for_text_returns_none_when_no_character_named() -> None:
    from alebrijes.server import _pick_voice_for_text

    cast_file = CastFile(book_title="Fixture", entries=[CastEntry(character="Felicity", voice_state_path="fantine")])
    voice_states = {"Felicity": ("fantine", "f")}

    assert _pick_voice_for_text("Nobody mentioned here.", cast_file, voice_states) is None


def test_pick_voice_for_text_returns_none_without_cast_or_voice_states() -> None:
    from alebrijes.server import _pick_voice_for_text

    assert _pick_voice_for_text("Felicity says hi.", None, {"Felicity": ("fantine", "f")}) is None
    cast_file = CastFile(book_title="Fixture", entries=[CastEntry(character="Felicity", voice_state_path="fantine")])
    assert _pick_voice_for_text("Felicity says hi.", cast_file, None) is None


def test_pin_chat_model_local_routes_to_scripter_client_ignoring_chat_stream_fn(monkeypatch) -> None:
    from alebrijes.server import _pin_chat_model

    class _FakeScripterClient:
        def __init__(self):
            self.messages_seen = None

        def chat_stream(self, messages, *, model="local"):
            self.messages_seen = messages
            yield "reply from scripter"

    fake_client = _FakeScripterClient()
    monkeypatch.setattr("alebrijes.llm.scripter.ScripterClient", lambda: fake_client)

    def _unused_chat_stream_fn(messages, *, model=None):
        raise AssertionError("should not be called when model is 'local'")

    pinned = _pin_chat_model(_unused_chat_stream_fn, "local")
    result = list(pinned([{"role": "user", "content": "hi"}]))

    assert result == ["reply from scripter"]
    assert fake_client.messages_seen == [{"role": "user", "content": "hi"}]


def test_session_start_with_max_improv_exchanges_pins_cyoa_steer_back_timing(tmp_path: Path) -> None:
    m4b_path, cast_path = _setup_session_fixture(tmp_path)
    seen_prompts = []

    def chat_stream_fn(messages, *, model="hermes3:8b"):
        seen_prompts.append(messages[0]["content"])
        yield "ok"

    client = _client(embed_fn=_fake_embed_fn, chat_stream_fn=chat_stream_fn)
    start = client.post(
        "/api/session/start",
        headers=_headers(),
        json={"m4b": str(m4b_path), "cast": str(cast_path), "mode": "cyoa", "max_improv_exchanges": 1},
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        ws.send_json({"type": "chat", "text": "I open the door."})
        ws.receive_json()

    # max_improv_exchanges=1 means the very first exchange already hits the
    # cap and should inject an upcoming (higher-seq) span as a steer-back
    # beat -- with the default of 3 this wouldn't happen until turn three.
    assert "Span number 1" in seen_prompts[0] or "Span number 2" in seen_prompts[0]


def test_session_start_reports_spoken_false_when_agent_build_fails(tmp_path: Path) -> None:
    m4b_path, cast_path = _setup_session_fixture(tmp_path)

    def _raise(cast_file):
        raise RuntimeError("no GPU/model available")

    client = _client(embed_fn=_fake_embed_fn, chat_stream_fn=_fake_chat_stream, build_narrator_agent_fn=_raise)
    response = client.post(
        "/api/session/start", headers=_headers(), json={"m4b": str(m4b_path), "cast": str(cast_path)}
    )

    assert response.status_code == 200
    assert response.json()["spoken"] is False


def test_session_ws_unknown_session_id_closes() -> None:
    client = _client()
    with pytest.raises(Exception):
        with client.websocket_connect(f"/api/session/does-not-exist/ws?token={TOKEN}"):
            pass


def test_session_ws_rejects_missing_or_wrong_token(tmp_path: Path) -> None:
    m4b_path, cast_path = _setup_session_fixture(tmp_path)
    client = _client(
        embed_fn=_fake_embed_fn, chat_stream_fn=_fake_chat_stream, build_narrator_agent_fn=lambda c: _FakeSynthAgent()
    )
    start = client.post(
        "/api/session/start", headers=_headers(), json={"m4b": str(m4b_path), "cast": str(cast_path)}
    )
    session_id = start.json()["session_id"]

    with pytest.raises(Exception):
        with client.websocket_connect(f"/api/session/{session_id}/ws?token=wrong-token"):
            pass


def test_session_ws_position_message_triggers_awaiting_input_for_checkpoint(tmp_path: Path) -> None:
    m4b_path, cast_path = _setup_session_fixture(tmp_path)

    def chat_stream_fn(messages, *, model="hermes3:8b"):
        yield "What do you think happens next?"

    def _no_agent(cast_file):
        raise RuntimeError("no model in this test")

    client = _client(embed_fn=_fake_embed_fn, chat_stream_fn=chat_stream_fn, build_narrator_agent_fn=_no_agent)
    start = client.post(
        "/api/session/start",
        headers=_headers(),
        json={"m4b": str(m4b_path), "cast": str(cast_path), "mode": "checkpoint", "checkpoints": ["0.1.0"]},
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        ws.send_json({"type": "position", "span_id": "0.0.0"})
        # Not a checkpoint -- nothing should arrive for this one; prove it by
        # immediately following with the real checkpoint and checking that's
        # the very first message received.
        ws.send_json({"type": "position", "span_id": "0.1.0"})
        message = ws.receive_json()

    assert message["type"] == "awaiting_input"
    assert message["text"] == "What do you think happens next?"


def test_session_ws_chat_returns_answer_and_streams_audio_when_spoken(tmp_path: Path) -> None:
    m4b_path, cast_path = _setup_session_fixture(tmp_path)

    def chat_stream_fn(messages, *, model="hermes3:8b"):
        yield "Hi there. "
        yield "Nice to meet you."

    client = _client(
        embed_fn=_fake_embed_fn,
        chat_stream_fn=chat_stream_fn,
        build_narrator_agent_fn=lambda cast_file: _FakeSynthAgent(),
    )
    start = client.post(
        "/api/session/start", headers=_headers(), json={"m4b": str(m4b_path), "cast": str(cast_path)}
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        ws.send_json({"type": "chat", "text": "hello"})
        messages = [ws.receive_json(), ws.receive_json(), ws.receive_json()]

    audio_messages = [m for m in messages if m["type"] == "audio_chunk"]
    answer_messages = [m for m in messages if m["type"] == "answer"]
    assert len(audio_messages) == 2  # one per sentence
    assert len(answer_messages) == 1
    assert answer_messages[0]["text"] == "Hi there. Nice to meet you."

    # Each audio chunk must be a self-contained, decodable WAV file.
    for chunk in audio_messages:
        wav_bytes = base64.b64decode(chunk["data"])
        with wave.open(io.BytesIO(wav_bytes)) as handle:
            assert handle.getnframes() > 0


def test_session_ws_chat_delivers_first_sentence_audio_before_generation_finishes(tmp_path: Path) -> None:
    # Proves delivery is truly progressive, not just synthesis: the second
    # sentence is deliberately slow to arrive from the LLM, so the first
    # sentence's audio_chunk must reach the client well before that -- if
    # the server were batching all audio until handle_user_message returns
    # (the bug this replaced), the first message wouldn't show up until
    # after the full ~0.5s generation, not within the tight window below.
    import time

    m4b_path, cast_path = _setup_session_fixture(tmp_path)

    def chat_stream_fn(messages, *, model="hermes3:8b"):
        yield "First sentence. "
        time.sleep(0.5)
        yield "Second sentence."

    client = _client(
        embed_fn=_fake_embed_fn,
        chat_stream_fn=chat_stream_fn,
        build_narrator_agent_fn=lambda cast_file: _FakeSynthAgent(),
    )
    start = client.post(
        "/api/session/start", headers=_headers(), json={"m4b": str(m4b_path), "cast": str(cast_path)}
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        ws.send_json({"type": "chat", "text": "hello"})
        started = time.time()
        first = ws.receive_json()
        first_elapsed = time.time() - started
        ws.receive_json()  # second audio_chunk
        ws.receive_json()  # answer

    assert first["type"] == "audio_chunk"
    assert first_elapsed < 0.4  # well before the 0.5s sleep before sentence two


class _ManySmallChunksSynthAgent:
    """A fake agent whose stream_chunks yields many tiny (10ms) chunks per
    sentence, like pocket_tts's real ~80ms internal decode chunks -- proves
    _coalesced_chunks groups them into a couple of reasonably-sized
    websocket messages instead of one-message-per-raw-chunk."""

    sample_rate = 1000

    def use_voice_state(self, voice, state) -> None:
        pass

    def stream_chunks(self, text: str):
        import numpy as np

        for _ in range(20):  # 20 * 10 samples @ 1000Hz = 200ms of audio
            yield np.ones(10, dtype=np.int16)


def test_session_ws_chat_coalesces_many_small_synth_chunks_into_few_messages(tmp_path: Path) -> None:
    m4b_path, cast_path = _setup_session_fixture(tmp_path)

    def chat_stream_fn(messages, *, model="hermes3:8b"):
        yield "A single short sentence."

    client = _client(
        embed_fn=_fake_embed_fn,
        chat_stream_fn=chat_stream_fn,
        build_narrator_agent_fn=lambda cast_file: _ManySmallChunksSynthAgent(),
    )
    start = client.post(
        "/api/session/start", headers=_headers(), json={"m4b": str(m4b_path), "cast": str(cast_path)}
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        ws.send_json({"type": "chat", "text": "hello"})
        messages = [ws.receive_json(), ws.receive_json()]

    audio_messages = [m for m in messages if m["type"] == "audio_chunk"]
    # 20 raw 10ms chunks (200ms total, below the 300ms coalescing floor) are
    # never individually flushed mid-stream -- they all land in the single
    # trailing flush once the generator is exhausted, not 20 messages.
    assert len(audio_messages) == 1
    assert audio_messages[0]["text"] == "A single short sentence."


class _LongManySmallChunksSynthAgent(_ManySmallChunksSynthAgent):
    """Same as _ManySmallChunksSynthAgent but produces enough audio (400ms)
    to cross the coalescing floor once, so streaming actually splits into
    a fast first piece plus one trailing piece -- not just one flush."""

    def stream_chunks(self, text: str):
        import numpy as np

        for _ in range(40):  # 40 * 10 samples @ 1000Hz = 400ms of audio
            yield np.ones(10, dtype=np.int16)


def test_session_ws_chat_flushes_a_long_reply_in_two_coalesced_pieces(tmp_path: Path) -> None:
    m4b_path, cast_path = _setup_session_fixture(tmp_path)

    def chat_stream_fn(messages, *, model="hermes3:8b"):
        yield "A longer sentence with more audio in it."

    client = _client(
        embed_fn=_fake_embed_fn,
        chat_stream_fn=chat_stream_fn,
        build_narrator_agent_fn=lambda cast_file: _LongManySmallChunksSynthAgent(),
    )
    start = client.post(
        "/api/session/start", headers=_headers(), json={"m4b": str(m4b_path), "cast": str(cast_path)}
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        ws.send_json({"type": "chat", "text": "hello"})
        messages = [ws.receive_json(), ws.receive_json(), ws.receive_json()]

    audio_messages = [m for m in messages if m["type"] == "audio_chunk"]
    # 400ms of audio, 300ms coalescing floor -- first flush at 300ms
    # (carries the sentence text), second (trailing 100ms) has none.
    assert len(audio_messages) == 2
    assert audio_messages[0]["text"] == "A longer sentence with more audio in it."
    assert audio_messages[1]["text"] == ""


def test_session_ws_chat_sends_text_only_when_not_spoken(tmp_path: Path) -> None:
    m4b_path, cast_path = _setup_session_fixture(tmp_path)

    def chat_stream_fn(messages, *, model="hermes3:8b"):
        yield "Text only answer."

    def _raise(cast_file):
        raise RuntimeError("no model")

    client = _client(embed_fn=_fake_embed_fn, chat_stream_fn=chat_stream_fn, build_narrator_agent_fn=_raise)
    start = client.post(
        "/api/session/start", headers=_headers(), json={"m4b": str(m4b_path), "cast": str(cast_path)}
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        ws.send_json({"type": "chat", "text": "hello"})
        message = ws.receive_json()

    assert message["type"] == "answer"
    assert message["text"] == "Text only answer."


# --- /api/session/start_live, live chapter streaming over the ws -----------


def _fake_build_render_pipeline(cast_file):
    return _FakeSynthAgent(), {"Narrator": ("alba", object())}


def _receive_skipping_progress(ws):
    """Reads websocket frames until one that isn't a chapter_progress
    update -- those interleave unpredictably with chapter_ready/other
    frames (both stream off the same background renderer thread via
    separate queues), so tests that don't care about progress skip them."""
    while True:
        message = ws.receive_json()
        if message["type"] != "chapter_progress":
            return message


def test_session_start_live_returns_session_id_and_spoken_true(tmp_path: Path) -> None:
    client = _client(
        ingest_book=lambda path: _fixture_book(),
        embed_fn=_fake_embed_fn,
        chat_stream_fn=_fake_chat_stream,
        build_render_pipeline_fn=_fake_build_render_pipeline,
    )
    cast_path = tmp_path / "book.cast.yaml"
    save_cast(CastFile(book_title="Fixture", entries=[CastEntry(character="Narrator", voice_state_path="alba")]), cast_path)

    response = client.post(
        "/api/session/start_live",
        headers=_headers(),
        json={"book": "unused.epub", "cast": str(cast_path)},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["spoken"] is True
    assert body["book_title"] == "Fixture"


def test_session_start_live_rejects_when_narrator_pipeline_fails(tmp_path: Path) -> None:
    cast_path = tmp_path / "book.cast.yaml"
    save_cast(CastFile(book_title="Fixture", entries=[CastEntry(character="Narrator", voice_state_path="alba")]), cast_path)

    def _raise(cast_file):
        raise RuntimeError("no pocket-tts model available")

    client = _client(ingest_book=lambda path: _fixture_book(), build_render_pipeline_fn=_raise)
    response = client.post(
        "/api/session/start_live", headers=_headers(), json={"book": "unused.epub", "cast": str(cast_path)}
    )
    assert response.status_code == 400


def test_session_ws_live_streams_chapters_as_they_render(tmp_path: Path) -> None:
    cast_path = tmp_path / "book.cast.yaml"
    save_cast(CastFile(book_title="Fixture", entries=[CastEntry(character="Narrator", voice_state_path="alba")]), cast_path)

    client = _client(
        ingest_book=lambda path: _fixture_book(),  # 2 chapters, see _fixture_book()
        embed_fn=_fake_embed_fn,
        chat_stream_fn=_fake_chat_stream,
        build_render_pipeline_fn=_fake_build_render_pipeline,
    )
    start = client.post(
        "/api/session/start_live",
        headers=_headers(),
        json={"book": "unused.epub", "cast": str(cast_path), "lookahead_chapters": 5},
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        first = _receive_skipping_progress(ws)
        second = _receive_skipping_progress(ws)

    assert first["type"] == "chapter_ready"
    assert second["type"] == "chapter_ready"
    assert {first["chapter_index"], second["chapter_index"]} == {0, 1}
    assert first["spans"]  # at least one span with real text/timings
    assert "text" in first["spans"][0] and "start_ms" in first["spans"][0]

    wav_bytes = base64.b64decode(first["data"])
    with wave.open(io.BytesIO(wav_bytes)) as handle:
        assert handle.getnframes() > 0


def test_session_ws_live_sends_chapter_progress_while_rendering(tmp_path: Path) -> None:
    cast_path = tmp_path / "book.cast.yaml"
    save_cast(CastFile(book_title="Fixture", entries=[CastEntry(character="Narrator", voice_state_path="alba")]), cast_path)

    client = _client(
        ingest_book=lambda path: _fixture_book(),  # 2 chapters, 1 line each
        embed_fn=_fake_embed_fn,
        chat_stream_fn=_fake_chat_stream,
        build_render_pipeline_fn=_fake_build_render_pipeline,
    )
    start = client.post(
        "/api/session/start_live",
        headers=_headers(),
        json={"book": "unused.epub", "cast": str(cast_path), "lookahead_chapters": 5},
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        # 2 chapter_progress (one line each) + 2 chapter_ready -- order between
        # the two streams isn't guaranteed, so just collect all four.
        messages = [ws.receive_json() for _ in range(4)]

    types = [m["type"] for m in messages]
    assert types.count("chapter_progress") == 2
    assert types.count("chapter_ready") == 2
    progress = next(m for m in messages if m["type"] == "chapter_progress")
    assert progress["lines_done"] == 1
    assert progress["lines_total"] == 1
    assert progress["chapter_index"] in (0, 1)


def test_session_ws_live_chat_response_uses_voice_of_named_character(tmp_path: Path) -> None:
    cast_path = tmp_path / "book.cast.yaml"
    save_cast(
        CastFile(
            book_title="Fixture",
            entries=[
                CastEntry(character="Narrator", voice_state_path="alba"),
                CastEntry(character="Felicity", voice_state_path="fantine"),
                CastEntry(character="Sara", voice_state_path="cosette"),
            ],
        ),
        cast_path,
    )

    fake_agent = _FakeSynthAgent()

    def _build_render_pipeline(cast_file):
        return fake_agent, {
            "Narrator": ("alba", "narrator-state"),
            "Felicity": ("fantine", "felicity-state"),
            "Sara": ("cosette", "sara-state"),
        }

    def chat_stream_fn(messages, *, model="hermes3:8b"):
        yield "Felicity says hello."

    client = _client(
        ingest_book=lambda path: _fixture_book(),
        embed_fn=_fake_embed_fn,
        chat_stream_fn=chat_stream_fn,
        build_render_pipeline_fn=_build_render_pipeline,
    )
    start = client.post(
        "/api/session/start_live",
        headers=_headers(),
        json={"book": "unused.epub", "cast": str(cast_path), "lookahead_chapters": 5},
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        ws.send_json({"type": "chat", "text": "who is felicity"})
        audio_chunk = _receive_until(ws, "audio_chunk")

    assert audio_chunk["character"] == "Felicity"
    assert ("fantine", "felicity-state") in fake_agent.voice_calls


def test_session_ws_live_chat_response_falls_back_to_narrator_voice_when_unnamed(tmp_path: Path) -> None:
    cast_path = tmp_path / "book.cast.yaml"
    save_cast(
        CastFile(
            book_title="Fixture",
            entries=[
                CastEntry(character="Narrator", voice_state_path="alba"),
                CastEntry(character="Felicity", voice_state_path="fantine"),
            ],
        ),
        cast_path,
    )

    fake_agent = _FakeSynthAgent()

    def _build_render_pipeline(cast_file):
        return fake_agent, {
            "Narrator": ("alba", "narrator-state"),
            "Felicity": ("fantine", "felicity-state"),
        }

    def chat_stream_fn(messages, *, model="hermes3:8b"):
        yield "Nobody named here says hello."

    client = _client(
        ingest_book=lambda path: _fixture_book(),
        embed_fn=_fake_embed_fn,
        chat_stream_fn=chat_stream_fn,
        build_render_pipeline_fn=_build_render_pipeline,
    )
    start = client.post(
        "/api/session/start_live",
        headers=_headers(),
        json={"book": "unused.epub", "cast": str(cast_path), "lookahead_chapters": 5},
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        ws.send_json({"type": "chat", "text": "hi"})
        audio_chunk = _receive_until(ws, "audio_chunk")

    assert audio_chunk["character"] == "Narrator"
    assert ("alba", "narrator-state") in fake_agent.voice_calls


def test_session_ws_live_chapter_consumed_advances_lookahead(tmp_path: Path) -> None:
    cast_path = tmp_path / "book.cast.yaml"
    save_cast(CastFile(book_title="Fixture", entries=[CastEntry(character="Narrator", voice_state_path="alba")]), cast_path)

    client = _client(
        ingest_book=lambda path: _fixture_book(),
        embed_fn=_fake_embed_fn,
        chat_stream_fn=_fake_chat_stream,
        build_render_pipeline_fn=_fake_build_render_pipeline,
    )
    start = client.post(
        "/api/session/start_live",
        headers=_headers(),
        json={"book": "unused.epub", "cast": str(cast_path), "lookahead_chapters": 1},
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        first = _receive_skipping_progress(ws)
        assert first["chapter_index"] == 0

        ws.send_json({"type": "chapter_consumed", "chapter_index": 0})
        second = _receive_skipping_progress(ws)
        assert second["chapter_index"] == 1


# --- lesson mode: quiz_path/coffee_path over the live session --------------


def _fixture_script_book_with_quiz() -> Book:
    return Book(
        title="Fixture Lesson",
        source_format="script",
        quiz_checkpoints={"0.0.0": "q1"},
        chapters=[
            Chapter(
                title="One",
                index=0,
                paragraphs=[Paragraph(text="NARRATOR: Roles delegate access.", index=0)],
            )
        ],
    )


def _write_quiz_file(tmp_path: Path) -> Path:
    quiz_path = tmp_path / "book.quiz.yaml"
    quiz_path.write_text(
        "questions:\n"
        "  q1:\n"
        "    prompt: \"What is an IAM role?\"\n"
        "    options:\n"
        "      A: \"A user\"\n"
        "      B: \"A delegated identity\"\n"
        "    correct: B\n",
        encoding="utf-8",
    )
    return quiz_path


def _receive_until(ws, expected_type: str, max_messages: int = 10) -> dict:
    """Reads messages off `ws` until one of `expected_type` arrives,
    discarding audio_chunk messages along the way -- spoken sessions
    (real TTS agent available) interleave those before the final
    answer/awaiting_input message; see _speak_llm_call in server.py."""
    for _ in range(max_messages):
        message = ws.receive_json()
        if message["type"] == expected_type:
            return message
    raise AssertionError(f"Never received a {expected_type!r} message within {max_messages} messages")


def test_session_start_live_with_quiz_path_merges_script_quiz_checkpoints(tmp_path: Path) -> None:
    cast_path = tmp_path / "book.cast.yaml"
    save_cast(CastFile(book_title="Fixture Lesson", entries=[CastEntry(character="Narrator", voice_state_path="alba")]), cast_path)
    quiz_path = _write_quiz_file(tmp_path)

    def chat_stream_fn(messages, *, model="hermes3:8b"):
        yield "What is an IAM role? A) A user B) A delegated identity"

    client = _client(
        ingest_book=lambda path: _fixture_script_book_with_quiz(),
        embed_fn=_fake_embed_fn,
        chat_stream_fn=chat_stream_fn,
        build_render_pipeline_fn=_fake_build_render_pipeline,
    )
    start = client.post(
        "/api/session/start_live",
        headers=_headers(),
        json={
            "book": "unused.script",
            "cast": str(cast_path),
            "mode": "learning",
            "lookahead_chapters": 5,
            "quiz_path": str(quiz_path),
        },
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        chapter_ready = _receive_skipping_progress(ws)
        assert chapter_ready["type"] == "chapter_ready"

        ws.send_json({"type": "position", "span_id": "0.0.0"})
        awaiting = _receive_until(ws, "awaiting_input")
        assert awaiting["choices"] == {"A": "A user", "B": "A delegated identity"}

        ws.send_json({"type": "chat", "text": "B"})
        answer = _receive_until(ws, "answer")
        assert answer["type"] == "answer"

        # Query while the session (and its WS) is still live -- a session
        # is scoped to its connection, same as everywhere else in this file.
        score = client.get(f"/api/session/{session_id}/score", headers=_headers())
        assert score.status_code == 200
        assert score.json() == {"correct": 1, "total": 1}

    # quiz_path is "book.quiz.yaml" -- .with_suffix(".progress.json") only
    # swaps the final ".yaml", giving "book.quiz.progress.json".
    progress = json.loads((tmp_path / "book.quiz.progress.json").read_text())
    assert progress["answered"]["q1"]["correct"] is True


def test_session_ws_live_quiz_checkpoint_asks_the_question_with_no_llm_call(tmp_path: Path) -> None:
    # A checkpoint with an attached quiz question is asked verbatim --
    # no prefetch needed, no LLM round-trip at all, live or otherwise.
    cast_path = tmp_path / "book.cast.yaml"
    save_cast(CastFile(book_title="Fixture Lesson", entries=[CastEntry(character="Narrator", voice_state_path="alba")]), cast_path)
    quiz_path = _write_quiz_file(tmp_path)

    calls = []

    def chat_stream_fn(messages, *, model="hermes3:8b"):
        calls.append(messages)
        yield "unused"

    client = _client(
        ingest_book=lambda path: _fixture_script_book_with_quiz(),
        embed_fn=_fake_embed_fn,
        chat_stream_fn=chat_stream_fn,
        build_render_pipeline_fn=_fake_build_render_pipeline,
    )
    start = client.post(
        "/api/session/start_live",
        headers=_headers(),
        json={
            "book": "unused.script",
            "cast": str(cast_path),
            "mode": "learning",
            "lookahead_chapters": 5,
            "quiz_path": str(quiz_path),
        },
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        _receive_skipping_progress(ws)  # chapter_ready

        ws.send_json({"type": "position", "span_id": "0.0.0"})
        awaiting = _receive_until(ws, "awaiting_input")

    assert awaiting["choices"] == {"A": "A user", "B": "A delegated identity"}
    assert calls == []  # never touched the LLM to ask it


def _fixture_script_book_two_chapters() -> Book:
    return Book(
        title="Fixture Lesson",
        source_format="script",
        chapters=[
            Chapter(title="One", index=0, paragraphs=[Paragraph(text="NARRATOR: Roles delegate access.", index=0)]),
            Chapter(title="Two", index=1, paragraphs=[Paragraph(text="NARRATOR: Policies grant permissions.", index=0)]),
        ],
    )


def test_session_ws_live_looks_ahead_one_chapter_to_prefetch_its_checkpoint_too(tmp_path: Path) -> None:
    # Plain (non-quiz) CHECKPOINT-mode pauses, since a quiz question's
    # response is never prefetched (see prefetch_checkpoint) -- this is
    # about the look-ahead mechanism itself, not quiz-specific behavior.
    cast_path = tmp_path / "book.cast.yaml"
    save_cast(CastFile(book_title="Fixture Lesson", entries=[CastEntry(character="Narrator", voice_state_path="alba")]), cast_path)

    calls = []

    def chat_stream_fn(messages, *, model="hermes3:8b"):
        calls.append(messages)
        yield "a reaction"

    client = _client(
        ingest_book=lambda path: _fixture_script_book_two_chapters(),
        embed_fn=_fake_embed_fn,
        chat_stream_fn=chat_stream_fn,
        build_render_pipeline_fn=_fake_build_render_pipeline,
    )
    start = client.post(
        "/api/session/start_live",
        headers=_headers(),
        json={
            "book": "unused.script",
            "cast": str(cast_path),
            "mode": "checkpoint",
            "checkpoints": ["0.0.0", "1.0.0"],
            "lookahead_chapters": 5,  # both chapters render well ahead of any playback/consumption
        },
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        # Both chapters are tiny (one line each) and the fake render
        # pipeline is instant, so by the time _drain_chapters starts
        # draining, both are typically already queued -- the look-ahead
        # peek should grab chapter 1 in the same pass as chapter 0,
        # without needing a "chapter_consumed" round trip first.
        first = _receive_skipping_progress(ws)
        second = _receive_skipping_progress(ws)
        assert {first["chapter_index"], second["chapter_index"]} == {0, 1}

        # Give both prefetch threads a moment to land.
        deadline = time.time() + 2
        while len(calls) < 2 and time.time() < deadline:
            time.sleep(0.02)
        assert len(calls) == 2  # both chapters' checkpoints prefetched, no "position" sent yet


def test_session_start_live_without_quiz_path_has_no_choices_on_checkpoint(tmp_path: Path) -> None:
    cast_path = tmp_path / "book.cast.yaml"
    save_cast(CastFile(book_title="Fixture", entries=[CastEntry(character="Narrator", voice_state_path="alba")]), cast_path)

    def chat_stream_fn(messages, *, model="hermes3:8b"):
        yield "What do you think happens next?"

    client = _client(
        ingest_book=lambda path: _fixture_script_book_with_quiz(),  # has quiz_checkpoints, but no quiz_path given
        embed_fn=_fake_embed_fn,
        chat_stream_fn=chat_stream_fn,
        build_render_pipeline_fn=_fake_build_render_pipeline,
    )
    start = client.post(
        "/api/session/start_live",
        headers=_headers(),
        json={
            "book": "unused.script",
            "cast": str(cast_path),
            "mode": "learning",
            "lookahead_chapters": 5,
            "checkpoints": ["0.0.0"],  # explicit plain checkpoint, no quiz_path this time
        },
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        ws.receive_json()  # chapter_ready

        ws.send_json({"type": "position", "span_id": "0.0.0"})
        awaiting = _receive_until(ws, "awaiting_input")

    assert "choices" not in awaiting


def test_session_score_returns_404_for_unknown_session() -> None:
    client = _client()
    response = client.get("/api/session/does-not-exist/score", headers=_headers())
    assert response.status_code == 404


def test_session_start_live_with_coffee_path_grounds_cyoa(tmp_path: Path) -> None:
    cast_path = tmp_path / "book.cast.yaml"
    save_cast(CastFile(book_title="Fixture", entries=[CastEntry(character="Narrator", voice_state_path="alba")]), cast_path)
    coffee_path = tmp_path / "book.coffee.md"
    coffee_path.write_text("A mysterious fog rolls into town.\n", encoding="utf-8")

    seen_prompts = []

    def chat_stream_fn(messages, *, model="hermes3:8b"):
        seen_prompts.append(messages[0]["content"])
        yield "ok"

    client = _client(
        ingest_book=lambda path: _fixture_book(),
        embed_fn=_fake_embed_fn,
        chat_stream_fn=chat_stream_fn,
        build_render_pipeline_fn=_fake_build_render_pipeline,
    )
    start = client.post(
        "/api/session/start_live",
        headers=_headers(),
        json={
            "book": "unused.epub",
            "cast": str(cast_path),
            "mode": "cyoa",
            "lookahead_chapters": 5,
            "coffee_path": str(coffee_path),
        },
    )
    session_id = start.json()["session_id"]

    with client.websocket_connect(f"/api/session/{session_id}/ws?token={TOKEN}") as ws:
        ws.receive_json()  # chapter_ready
        ws.send_json({"type": "chat", "text": "I investigate the fog."})
        answer = _receive_until(ws, "answer")

    assert answer["type"] == "answer"
    assert "A mysterious fog rolls into town." in seen_prompts[0]
