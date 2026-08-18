"""Local FastAPI backend for the desktop GUI (Phase 3).

Not one of the six pipeline stages — this is the local HTTP API the pywebview
frontend talks to instead of pywebview's built-in JS<->Python bridge, chosen
for real render-progress streaming and reuse against a future remote render
server (see PHASES.md Phase 5). A local HTTP server is a real attack surface
(DNS rebinding: a malicious webpage open in the user's actual browser can get
its JS to hit `127.0.0.1:PORT`), so every route requires all of:

- binding to `127.0.0.1` only, never `0.0.0.0` (enforced by the caller of
  `run_server`, not this module, since binding is a deployment concern)
- an OS-assigned ephemeral port (`find_free_port`), not a fixed one
- a random per-launch secret token (`X-Alebrijes-Token` header) — the
  Jupyter pattern
- `Origin` header validation: rejected if *present and wrong*; allowed if
  *absent*, since browsers always set `Origin` on cross-origin requests (the
  exact requests this defends against), so an attacker's request can't hide
  it — only same-origin requests are ever missing it.

Render is long-running (minutes), so `/api/render` starts a background job
and returns immediately; progress is polled via `/api/render/{job_id}`.
Progress is chapter-level only (`render_book`'s `on_chapter_done` callback),
not per-line — a real streaming (SSE/WebSocket) upgrade is a documented
future refinement, not built in this pass.
"""

from __future__ import annotations

import secrets
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from alebrijes.cast import CastEntry, CastFile, load as load_cast, save as save_cast
from alebrijes.cast.build import LOW_CONFIDENCE_THRESHOLD, build_cast_file
from alebrijes.config import BUILTIN_VOICES, Config
from alebrijes.ingest import Book
from alebrijes.preview import synthesize_preview as default_synthesize_preview
from alebrijes.render import render_book as default_render_book


def find_free_port(host: str = "127.0.0.1") -> int:
    """Asks the OS for an unused port. Small TOCTOU race between this call and
    binding the real server to it is accepted — standard practice for local
    ephemeral port selection, not a security boundary."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def generate_token() -> str:
    return secrets.token_urlsafe(32)


class CastRequest(BaseModel):
    book: str
    out: Optional[str] = None


class CastEntryModel(BaseModel):
    character: str
    voice_state_path: str


class LowConfidenceLine(BaseModel):
    character: str
    confidence: float
    text: str


class CastResponse(BaseModel):
    cast_path: str
    book_title: str
    entries: list[CastEntryModel]
    low_confidence_lines: list[LowConfidenceLine]


class CastUpdateRequest(BaseModel):
    cast_path: str
    entries: list[CastEntryModel]


class ClonedVoice(BaseModel):
    name: str
    voice_state_path: str


class VoicesResponse(BaseModel):
    builtin: list[str]
    cloned: list[ClonedVoice]


class PreviewRequest(BaseModel):
    text: str
    voice_state_path: str


class RenderRequest(BaseModel):
    book: str
    cast: str
    out: Optional[str] = None
    pause_ms: int = 250


class RenderJobStatus(BaseModel):
    job_id: str
    status: str  # "running" | "done" | "error"
    chapters_done: int = 0
    chapters_total: int = 0
    chapter_titles: list[str] = []
    elapsed_seconds: float = 0.0
    output_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class _RenderJob:
    status: str = "running"
    chapters_done: int = 0
    chapters_total: int = 0
    chapter_titles: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


IngestFn = Callable[[Path], Book]
RenderFn = Callable[..., Path]
SynthesizePreviewFn = Callable[[str, str], bytes]


def _default_ingest_book(book_path: Path) -> Book:
    # Local import: mirrors alebrijes.cli._ingest_book's dispatch-by-extension,
    # duplicated rather than imported from cli.py to avoid a server->cli
    # dependency for what's a small, stable piece of logic.
    suffix = book_path.suffix.lower()
    if suffix == ".epub":
        from alebrijes.ingest.epub import parse
    elif suffix == ".pdf":
        from alebrijes.ingest.pdf import parse
    elif suffix == ".txt":
        from alebrijes.ingest.txt import parse
    else:
        raise NotImplementedError(f"Ingest for '{suffix}' files is not supported.")
    return parse(book_path)


def create_app(
    token: str,
    *,
    allowed_origin: str = "http://127.0.0.1",
    ingest_book: IngestFn = _default_ingest_book,
    render_fn: RenderFn = default_render_book,
    synthesize_preview_fn: SynthesizePreviewFn = default_synthesize_preview,
    voices_dir: Optional[Path] = None,
) -> FastAPI:
    if voices_dir is None:
        voices_dir = Config().voices_dir
    app = FastAPI(title="alebrijes")
    jobs: dict[str, _RenderJob] = {}

    def _verify_request(request: Request) -> None:
        supplied = request.headers.get("x-alebrijes-token")
        if not supplied or not secrets.compare_digest(supplied, token):
            raise HTTPException(status_code=403, detail="Missing or invalid token")
        origin = request.headers.get("origin")
        if origin is not None and origin != allowed_origin:
            raise HTTPException(status_code=403, detail="Invalid origin")

    @app.get("/api/health", dependencies=[Depends(_verify_request)])
    def health() -> dict:
        from alebrijes.diagnostics import torch_build_warning

        warning = torch_build_warning()
        return {"status": "ok", "warnings": [warning] if warning else []}

    @app.get("/api/voices", response_model=VoicesResponse, dependencies=[Depends(_verify_request)])
    def voices() -> VoicesResponse:
        cloned = []
        if voices_dir.exists():
            for pt_file in sorted(voices_dir.glob("*.pt")):
                cloned.append(ClonedVoice(name=pt_file.stem, voice_state_path=str(pt_file)))
        return VoicesResponse(builtin=list(BUILTIN_VOICES), cloned=cloned)

    @app.post("/api/preview", dependencies=[Depends(_verify_request)])
    def preview(payload: PreviewRequest) -> Response:
        # Bounded to a single line/short passage, not an arbitrary text
        # blob — this endpoint exists for "does this voice sound right for
        # this character", not batch synthesis (that's /api/render).
        if len(payload.text) > 500:
            raise HTTPException(status_code=400, detail="Preview text must be 500 characters or fewer.")
        try:
            audio_bytes = synthesize_preview_fn(payload.text, payload.voice_state_path)
        except Exception as exc:  # noqa: BLE001 - reported to the client, not swallowed
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=audio_bytes, media_type="audio/wav")

    @app.post("/api/cast", response_model=CastResponse, dependencies=[Depends(_verify_request)])
    def cast(payload: CastRequest) -> CastResponse:
        book_path = Path(payload.book)
        try:
            book = ingest_book(book_path)
        except NotImplementedError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result = build_cast_file(book)
        out_path = Path(payload.out) if payload.out else book_path.with_suffix(".cast.yaml")
        save_cast(result.cast_file, out_path)

        return CastResponse(
            cast_path=str(out_path),
            book_title=result.cast_file.book_title,
            entries=[
                CastEntryModel(character=e.character, voice_state_path=e.voice_state_path)
                for e in result.cast_file.entries
            ],
            low_confidence_lines=[
                LowConfidenceLine(character=a.character, confidence=a.confidence, text=a.span.text)
                for a in result.low_confidence()
            ],
        )

    @app.put("/api/cast", response_model=CastResponse, dependencies=[Depends(_verify_request)])
    def update_cast(payload: CastUpdateRequest) -> CastResponse:
        cast_path = Path(payload.cast_path)
        try:
            existing = load_cast(cast_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        updated = CastFile(
            book_title=existing.book_title,
            entries=[CastEntry(character=e.character, voice_state_path=e.voice_state_path) for e in payload.entries],
        )
        save_cast(updated, cast_path)

        return CastResponse(
            cast_path=str(cast_path),
            book_title=updated.book_title,
            entries=[
                CastEntryModel(character=e.character, voice_state_path=e.voice_state_path) for e in updated.entries
            ],
            low_confidence_lines=[],
        )

    def _run_render_job(job_id: str, book_path: Path, cast_file: CastFile, out_path: Path, pause_ms: int) -> None:
        job = jobs[job_id]

        def _on_chapter_done(done: int, total: int) -> None:
            with job.lock:
                job.chapters_done = done
                job.chapters_total = total

        try:
            book = ingest_book(book_path)
            with job.lock:
                job.chapters_total = len(book.chapters)
                job.chapter_titles = [c.title or f"Chapter {c.index + 1}" for c in book.chapters]
            result_path = render_fn(
                book, cast_file, out_path, pause_ms=pause_ms, on_chapter_done=_on_chapter_done
            )
            with job.lock:
                job.status = "done"
                job.output_path = str(result_path)
                job.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001 - reported to the client, not swallowed
            with job.lock:
                job.status = "error"
                job.error = str(exc)
                job.finished_at = time.time()

    @app.post("/api/render", response_model=RenderJobStatus, dependencies=[Depends(_verify_request)])
    def render(payload: RenderRequest) -> RenderJobStatus:
        book_path = Path(payload.book)
        try:
            cast_file = load_cast(Path(payload.cast))
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        out_path = Path(payload.out) if payload.out else book_path.with_suffix(".m4b")
        job_id = uuid.uuid4().hex
        jobs[job_id] = _RenderJob()

        thread = threading.Thread(
            target=_run_render_job,
            args=(job_id, book_path, cast_file, out_path, payload.pause_ms),
            daemon=True,
        )
        thread.start()

        return RenderJobStatus(job_id=job_id, status="running")

    @app.get("/api/render/{job_id}", response_model=RenderJobStatus, dependencies=[Depends(_verify_request)])
    def render_status(job_id: str) -> RenderJobStatus:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id")
        with job.lock:
            elapsed = (job.finished_at or time.time()) - job.started_at
            return RenderJobStatus(
                job_id=job_id,
                status=job.status,
                chapters_done=job.chapters_done,
                chapters_total=job.chapters_total,
                chapter_titles=list(job.chapter_titles),
                elapsed_seconds=elapsed,
                output_path=job.output_path,
                error=job.error,
            )

    # Deliberately NOT behind _verify_request: this is just the static app
    # shell (HTML/CSS/JS), no secret embedded in it. The token only ever
    # appears in the URL pywebview navigates to directly (see gui.py) — the
    # frontend JS reads it from the URL and attaches it to every /api/*
    # call itself. Serving the shell without the token is what makes that
    # possible in the first place (the page has to load before it can have
    # a token to present).
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _FRONTEND_HTML_PATH.read_text(encoding="utf-8")

    return app


_FRONTEND_HTML_PATH = Path(__file__).parent / "frontend" / "index.html"


def run_server_blocking(host: str = "127.0.0.1", port: int | None = None) -> None:
    """Blocking dev/manual-test entrypoint — prints the token and port, then
    runs until interrupted. The future pywebview shell won't call this
    directly: it needs the token/port *before* starting the server thread (to
    embed them into the page it loads), so it will instead call
    `generate_token()` / `find_free_port()` / `create_app()` itself and run
    uvicorn in a background thread. Those three building blocks are the
    actual integration points, not this function.
    """
    import uvicorn

    token = generate_token()
    resolved_port = port if port is not None else find_free_port(host)
    app = create_app(token, allowed_origin=f"http://{host}:{resolved_port}")
    print(f"alebrijes server: http://{host}:{resolved_port}  token={token}")
    uvicorn.run(app, host=host, port=resolved_port, log_level="warning")
