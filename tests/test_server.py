from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from alebrijes.cast import CastEntry, CastFile, save as save_cast
from alebrijes.ingest import Book, Chapter, Paragraph
from alebrijes.server import create_app, find_free_port, generate_token
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
