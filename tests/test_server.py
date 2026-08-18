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
    assert response.json() == {"status": "ok"}


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
    assert "Colin" in body["characters"]
    assert "Narrator" in body["characters"]


def test_cast_endpoint_rejects_unsupported_format(tmp_path: Path) -> None:
    unsupported = tmp_path / "book.docx"
    unsupported.write_text("not supported")

    client = _client()
    response = client.post("/api/cast", headers=_headers(), json={"book": str(unsupported)})

    assert response.status_code == 422


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
    assert status["output_path"] == str(out_path)
    assert out_path.read_bytes() == b"fake-m4b"


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
