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

import asyncio
import base64
import io
import secrets
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from alebrijes.cast import CastEntry, CastFile, load as load_cast, save as save_cast
from alebrijes.cast.build import LOW_CONFIDENCE_THRESHOLD, build_cast_file
from alebrijes.config import BUILTIN_VOICES, Config
from alebrijes.ingest import Book
from alebrijes.preview import synthesize_preview as default_synthesize_preview
from alebrijes.render import NARRATOR, render_book as default_render_book
from alebrijes.session import InteractiveSession, SessionMode
from alebrijes.synth.streaming import SentenceChunker
from alebrijes.transcript import Transcript, load as load_transcript


def _default_build_narrator_agent(cast_file: CastFile) -> object:
    """Builds a `TTSAgent` bound to the book's cast Narrator voice, for
    synthesizing spoken interactive-narration responses in the same voice
    as the rest of the audiobook. Loads the real pocket-tts model, so this
    is slow/heavy by design — callers should treat failure (no model
    available, wrong environment, etc.) as "spoken responses aren't
    available this session", not fatal; see `session_start`'s try/except.
    """
    from alebrijes.render import _load_voice_state
    from alebrijes.synth.agent import TTSAgent
    from pocket_tts import TTSModel

    try:
        model = TTSModel.load_model()
    except AttributeError:
        model = TTSModel()

    voice_key = cast_file.voice_for(NARRATOR)
    voice_path = Path(voice_key)
    state = _load_voice_state(voice_path) if voice_path.exists() else model.get_state_for_audio_prompt(voice_key)
    return TTSAgent(voice=voice_key, model=model, voice_state=state)


def _audio_to_wav_bytes(sample_rate: int, audio: "np.ndarray") -> bytes:
    from alebrijes.synth.agent import write_wav_array

    buffer = io.BytesIO()
    write_wav_array(buffer, sample_rate, audio)
    return buffer.getvalue()


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
    chapter_start: Optional[int] = None
    chapter_end: Optional[int] = None


class BookChaptersRequest(BaseModel):
    book: str


class ChapterInfo(BaseModel):
    index: int  # 0-based
    title: str


class BookChaptersResponse(BaseModel):
    chapters: list[ChapterInfo]


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


class ImportVoiceRequest(BaseModel):
    source: str  # path to an existing voice-state .pt or .safetensors file
    name: str


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


class RagBuildRequest(BaseModel):
    transcript: str
    out_prefix: Optional[str] = None


class RagBuildJobStatus(BaseModel):
    job_id: str
    status: str  # "running" | "done" | "error"
    spans_done: int = 0
    spans_total: int = 0
    output_prefix: Optional[str] = None
    error: Optional[str] = None


@dataclass
class _RagBuildJob:
    status: str = "running"
    spans_done: int = 0
    spans_total: int = 0
    output_prefix: Optional[str] = None
    error: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionAskRequest(BaseModel):
    book_title: str
    character_voice: str = "Narrator"
    position_summary: str
    rag_npz: str
    rag_json: str
    current_seq: int
    question: str
    top_k: int = 5


class SessionAskResponse(BaseModel):
    answer: str
    sources: list[str] = []


class SessionStartRequest(BaseModel):
    m4b: str
    cast: str
    mode: str = "qa"
    checkpoints: list[str] = []
    chat_model: Optional[str] = None  # overrides the server's default Ollama chat model for this session only


class SessionStartResponse(BaseModel):
    session_id: str
    book_title: str
    spoken: bool  # whether a narrator TTS agent was available for spoken responses


class SessionStartLiveRequest(BaseModel):
    book: str
    cast: str
    mode: str = "qa"
    checkpoints: list[str] = []
    lookahead_chapters: int = 2
    chat_model: Optional[str] = None
    rag_npz: Optional[str] = None
    rag_json: Optional[str] = None


IngestFn = Callable[[Path], Book]
RenderFn = Callable[..., Path]
SynthesizePreviewFn = Callable[[str, str], bytes]
EmbedFn = Callable[[str], "np.ndarray"]
ChatStreamFn = Callable[..., Iterator[str]]
ImportVoiceFn = Callable[[Path, str, Path], Path]
BuildNarratorAgentFn = Callable[[CastFile], object]
BuildRenderPipelineFn = Callable[[CastFile], tuple]  # -> (TTSAgent, voice_states dict)


def _default_build_render_pipeline(cast_file: CastFile) -> tuple:
    from alebrijes.render import build_render_agent

    return build_render_agent(cast_file)


def _pin_chat_model(chat_stream_fn: "ChatStreamFn", model: Optional[str]) -> "ChatStreamFn":
    """Wraps `chat_stream_fn` to always use `model`, if given -- lets one
    session request a different Ollama chat model than the server's
    default without every other session being affected."""
    if not model:
        return chat_stream_fn
    return lambda messages: chat_stream_fn(messages, model=model)


def _load_voice_state_dict(source_path: Path) -> dict:
    # .safetensors is loaded and re-saved as a torch-pickle .pt below --
    # every other consumer of a voice state (render.py, the *.pt glob in
    # /api/voices and `alebrijes voices list`) expects a torch-pickle file,
    # so normalizing here at import time keeps that assumption everywhere
    # else instead of teaching every consumer two formats.
    if source_path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(source_path))

    import torch

    try:
        return torch.load(source_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(source_path, map_location="cpu")


def _default_import_voice(source_path: Path, name: str, voices_dir: Path) -> Path:
    # No cloning inference here -- the packaged app ships pocket-tts's
    # ungated model (see PHASES.md), which can't do audio->voice-state
    # cloning at all. Voice states come from elsewhere (e.g. `alebrijes
    # voices add` on a source/dev install, which does have that
    # capability, via a one-time HF login) and get imported here as
    # already-computed .pt/.safetensors files -- validated (by loading),
    # not blindly copied, so a wrong file fails clearly now instead of
    # cryptically the next time this voice is used to render.
    import torch

    try:
        state = _load_voice_state_dict(source_path)
    except Exception as exc:
        raise ValueError(f"'{source_path}' doesn't look like a valid voice-state file: {exc}") from exc

    voices_dir.mkdir(parents=True, exist_ok=True)
    out_path = voices_dir / f"{name}.pt"
    torch.save(state, out_path)
    return out_path


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
    import_voice_fn: ImportVoiceFn = _default_import_voice,
    voices_dir: Optional[Path] = None,
    embed_fn: Optional[EmbedFn] = None,
    chat_stream_fn: Optional[ChatStreamFn] = None,
    list_models_fn: Optional[Callable[[], list[str]]] = None,
    build_narrator_agent_fn: BuildNarratorAgentFn = _default_build_narrator_agent,
    build_render_pipeline_fn: BuildRenderPipelineFn = _default_build_render_pipeline,
) -> FastAPI:
    if voices_dir is None:
        voices_dir = Config().voices_dir
    if embed_fn is None:
        from alebrijes.llm.ollama import OllamaClient

        embed_fn = OllamaClient().embed
    if chat_stream_fn is None:
        from alebrijes.llm.ollama import OllamaClient

        chat_stream_fn = OllamaClient().chat_stream
    if list_models_fn is None:
        from alebrijes.llm.ollama import OllamaClient

        list_models_fn = OllamaClient().list_models
    app = FastAPI(title="alebrijes")
    jobs: dict[str, _RenderJob] = {}
    rag_jobs: dict[str, _RagBuildJob] = {}
    sessions: dict[str, dict] = {}  # session_id -> {"session": InteractiveSession, "agent": object | None}

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

    @app.get("/api/llm/models", dependencies=[Depends(_verify_request)])
    def llm_models() -> dict:
        from alebrijes.llm.ollama import DEFAULT_CHAT_MODEL, OllamaUnavailable

        try:
            names = list_models_fn()
        except OllamaUnavailable as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"models": names, "default": DEFAULT_CHAT_MODEL}

    @app.get("/api/voices", response_model=VoicesResponse, dependencies=[Depends(_verify_request)])
    def voices() -> VoicesResponse:
        cloned = []
        if voices_dir.exists():
            for pt_file in sorted(voices_dir.glob("*.pt")):
                cloned.append(ClonedVoice(name=pt_file.stem, voice_state_path=str(pt_file)))
        return VoicesResponse(builtin=list(BUILTIN_VOICES), cloned=cloned)

    @app.post("/api/voices", response_model=VoicesResponse, dependencies=[Depends(_verify_request)])
    def add_voice(payload: ImportVoiceRequest) -> VoicesResponse:
        source_path = Path(payload.source)
        if not source_path.exists():
            raise HTTPException(status_code=400, detail=f"Voice file not found: {source_path}")
        if source_path.suffix.lower() not in (".pt", ".safetensors"):
            raise HTTPException(
                status_code=400,
                detail=f"'{source_path}' doesn't look like a voice-state file (expected .pt or .safetensors).",
            )
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Voice name is required.")

        try:
            import_voice_fn(source_path, name, voices_dir)
        except Exception as exc:  # noqa: BLE001 - reported to the client, not swallowed
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return voices()

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

    @app.post("/api/book/chapters", response_model=BookChaptersResponse, dependencies=[Depends(_verify_request)])
    def book_chapters(payload: BookChaptersRequest) -> BookChaptersResponse:
        book_path = Path(payload.book)
        try:
            book = ingest_book(book_path)
        except NotImplementedError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return BookChaptersResponse(
            chapters=[
                ChapterInfo(index=c.index, title=c.title or f"Chapter {c.index + 1}") for c in book.chapters
            ]
        )

    @app.post("/api/cast", response_model=CastResponse, dependencies=[Depends(_verify_request)])
    def cast(payload: CastRequest) -> CastResponse:
        book_path = Path(payload.book)
        try:
            book = ingest_book(book_path)
        except NotImplementedError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if payload.chapter_start is not None and payload.chapter_end is not None:
            if payload.chapter_start < 1 or payload.chapter_end < payload.chapter_start:
                raise HTTPException(status_code=400, detail="Invalid chapter range.")
            if payload.chapter_start > len(book.chapters):
                raise HTTPException(status_code=400, detail="Chapter range is out of bounds for this book.")

        result = build_cast_file(book, chapter_start=payload.chapter_start, chapter_end=payload.chapter_end)
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

    def _run_rag_build_job(job_id: str, transcript_path: Path, out_prefix: str) -> None:
        from alebrijes.rag import save as save_rag
        from alebrijes.rag.build import build_rag_index

        job = rag_jobs[job_id]

        def _on_progress(done: int, total: int) -> None:
            with job.lock:
                job.spans_done = done
                job.spans_total = total

        try:
            transcript = load_transcript(transcript_path)
            index = build_rag_index(transcript, embed_fn, on_progress=_on_progress)
            npz_path = Path(f"{out_prefix}.rag.npz")
            json_path = Path(f"{out_prefix}.rag.json")
            save_rag(index, npz_path, json_path)
            with job.lock:
                job.status = "done"
                job.output_prefix = out_prefix
        except Exception as exc:  # noqa: BLE001 - reported to the client, not swallowed
            with job.lock:
                job.status = "error"
                job.error = str(exc)

    @app.post("/api/rag/build", response_model=RagBuildJobStatus, dependencies=[Depends(_verify_request)])
    def rag_build(payload: RagBuildRequest) -> RagBuildJobStatus:
        transcript_path = Path(payload.transcript)
        if not transcript_path.exists():
            raise HTTPException(status_code=400, detail=f"Transcript not found: {transcript_path}")

        if payload.out_prefix:
            out_prefix = payload.out_prefix
        else:
            name = transcript_path.name
            stem = name[: -len(".transcript.json")] if name.endswith(".transcript.json") else transcript_path.stem
            out_prefix = str(transcript_path.parent / stem)

        job_id = uuid.uuid4().hex
        rag_jobs[job_id] = _RagBuildJob()

        thread = threading.Thread(
            target=_run_rag_build_job, args=(job_id, transcript_path, out_prefix), daemon=True
        )
        thread.start()

        return RagBuildJobStatus(job_id=job_id, status="running")

    @app.get(
        "/api/rag/build/{job_id}", response_model=RagBuildJobStatus, dependencies=[Depends(_verify_request)]
    )
    def rag_build_status(job_id: str) -> RagBuildJobStatus:
        job = rag_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id")
        with job.lock:
            return RagBuildJobStatus(
                job_id=job_id,
                status=job.status,
                spans_done=job.spans_done,
                spans_total=job.spans_total,
                output_prefix=job.output_prefix,
                error=job.error,
            )

    @app.post("/api/session/ask", response_model=SessionAskResponse, dependencies=[Depends(_verify_request)])
    def session_ask(payload: SessionAskRequest) -> SessionAskResponse:
        from alebrijes.llm.prompts import Mode, build_messages
        from alebrijes.rag import load as load_rag, search as search_rag

        npz_path = Path(payload.rag_npz)
        json_path = Path(payload.rag_json)
        if not npz_path.exists() or not json_path.exists():
            raise HTTPException(status_code=400, detail="RAG index not found — run `alebrijes rag build` first.")

        index = load_rag(npz_path, json_path)
        query_vector = embed_fn(payload.question)
        hits = search_rag(index, query_vector, max_seq=payload.current_seq, top_k=payload.top_k)

        messages = build_messages(
            mode=Mode.QA,
            book_title=payload.book_title,
            character_voice=payload.character_voice,
            position_summary=payload.position_summary,
            rag_excerpts=[hit.entry.text for hit in hits],
            history=[],
            user_message=payload.question,
        )

        try:
            answer = "".join(chat_stream_fn(messages))
        except Exception as exc:  # noqa: BLE001 - reported to the client, not swallowed
            raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

        return SessionAskResponse(answer=answer, sources=[hit.entry.span_id for hit in hits])

    @app.post("/api/session/start", response_model=SessionStartResponse, dependencies=[Depends(_verify_request)])
    def session_start(payload: SessionStartRequest) -> SessionStartResponse:
        from alebrijes.rag import load as load_rag_index

        m4b_path = Path(payload.m4b)
        transcript_path = m4b_path.with_suffix(".transcript.json")
        rag_npz_path = m4b_path.with_suffix(".rag.npz")
        rag_json_path = m4b_path.with_suffix(".rag.json")
        if not transcript_path.exists():
            raise HTTPException(status_code=400, detail=f"No transcript found for {m4b_path}")
        if not rag_npz_path.exists() or not rag_json_path.exists():
            raise HTTPException(status_code=400, detail="RAG index not found — run `alebrijes rag build` first.")

        try:
            mode = SessionMode(payload.mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {payload.mode!r}")

        try:
            cast_file = load_cast(Path(payload.cast))
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        transcript = load_transcript(transcript_path)
        rag_index = load_rag_index(rag_npz_path, rag_json_path)

        interactive_session = InteractiveSession(
            book_title=transcript.book_title,
            character_voice=NARRATOR,
            transcript=transcript,
            rag_index=rag_index,
            embed_fn=embed_fn,
            chat_stream_fn=_pin_chat_model(chat_stream_fn, payload.chat_model),
            mode=mode,
            checkpoints=payload.checkpoints,
        )

        try:
            agent = build_narrator_agent_fn(cast_file)
        except Exception:  # noqa: BLE001 - spoken responses degrade to text-only, not fatal
            agent = None

        session_id = uuid.uuid4().hex
        sessions[session_id] = {"session": interactive_session, "agent": agent}

        return SessionStartResponse(session_id=session_id, book_title=transcript.book_title, spoken=agent is not None)

    @app.post("/api/session/start_live", response_model=SessionStartResponse, dependencies=[Depends(_verify_request)])
    def session_start_live(payload: SessionStartLiveRequest) -> SessionStartResponse:
        # Live mode: start listening while the book renders instead of
        # waiting for a whole-book `alebrijes render` first. No .m4b/
        # .transcript.json needed -- the transcript is built up chapter by
        # chapter as session_ws's chapter-drain task consumes
        # LookaheadRenderer's output (see session/buffer.py).
        import queue as queue_module

        from alebrijes.rag import RagIndex, load as load_rag_index
        from alebrijes.session.buffer import LookaheadRenderer

        try:
            book = ingest_book(Path(payload.book))
        except Exception as exc:  # noqa: BLE001 - reported to the client, not swallowed
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not book.chapters:
            raise HTTPException(status_code=400, detail="Book has no chapters to render.")

        try:
            cast_file = load_cast(Path(payload.cast))
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            mode = SessionMode(payload.mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {payload.mode!r}")

        try:
            agent, voice_states = build_render_pipeline_fn(cast_file)
        except Exception as exc:  # noqa: BLE001 - live mode has no pre-rendered audio to fall back on
            raise HTTPException(status_code=400, detail=f"Could not prepare narrator voice: {exc}") from exc

        rag_index = RagIndex(entries=[], vectors=np.zeros((0, 0), dtype=np.float32))
        if payload.rag_npz and payload.rag_json:
            npz_path, json_path = Path(payload.rag_npz), Path(payload.rag_json)
            if npz_path.exists() and json_path.exists():
                rag_index = load_rag_index(npz_path, json_path)

        transcript = Transcript(book_title=book.title, chapters=[], spans=[])
        interactive_session = InteractiveSession(
            book_title=book.title,
            character_voice=NARRATOR,
            transcript=transcript,
            rag_index=rag_index,
            embed_fn=embed_fn,
            chat_stream_fn=_pin_chat_model(chat_stream_fn, payload.chat_model),
            mode=mode,
            checkpoints=payload.checkpoints,
        )

        chapter_queue: "queue_module.Queue" = queue_module.Queue()
        total_chapters = len(book.chapters)

        def on_chapter_ready(chapter_audio) -> None:
            chapter_queue.put(chapter_audio)
            if chapter_audio.chapter_index == total_chapters - 1:
                chapter_queue.put(None)  # sentinel: no more chapters coming

        renderer = LookaheadRenderer(
            book,
            cast_file,
            voice_states,
            agent,
            lookahead_chapters=payload.lookahead_chapters,
            on_chapter_ready=on_chapter_ready,
        )
        renderer.start()

        session_id = uuid.uuid4().hex
        sessions[session_id] = {
            "session": interactive_session,
            "agent": agent,
            "renderer": renderer,
            "chapter_queue": chapter_queue,
        }

        return SessionStartResponse(session_id=session_id, book_title=book.title, spoken=True)

    @app.websocket("/api/session/{session_id}/ws")
    async def session_ws(websocket: WebSocket, session_id: str) -> None:
        supplied = websocket.query_params.get("token")
        if not supplied or not secrets.compare_digest(supplied, token):
            await websocket.close(code=4403)
            return
        origin = websocket.headers.get("origin")
        if origin is not None and origin != allowed_origin:
            await websocket.close(code=4403)
            return

        entry = sessions.get(session_id)
        if entry is None:
            await websocket.close(code=4404)
            return

        await websocket.accept()
        interactive_session: InteractiveSession = entry["session"]
        agent = entry["agent"]
        renderer = entry.get("renderer")
        chapter_queue = entry.get("chapter_queue")

        # Two coroutines can both push frames on this socket -- the request/
        # response loop below, and (for a live session) the chapter-drain
        # task. A lock keeps their sends from interleaving mid-frame.
        send_lock = asyncio.Lock()

        async def _send(payload: dict) -> None:
            async with send_lock:
                await websocket.send_json(payload)

        async def _send_audio(audio: "np.ndarray") -> None:
            wav_bytes = _audio_to_wav_bytes(agent.sample_rate, audio)
            await _send({"type": "audio_chunk", "sample_rate": agent.sample_rate, "data": base64.b64encode(wav_bytes).decode("ascii")})

        async def _speak_llm_call(call_with_on_delta):
            """Runs a blocking `InteractiveSession` call (`handle_user_message`
            or `advance_position`) in a worker thread, sending each sentence's
            audio out over the socket the moment it's synthesized instead of
            batching it all up until the whole LLM response is done -- that's
            what actually makes the "lead with a short sentence" guideline
            (narrator_guidelines.md) pay off as low time-to-first-audio,
            rather than just spreading the same total wait out differently.

            `call_with_on_delta` is `lambda on_delta: <the blocking call>(...,
            on_delta=on_delta)`. `on_delta` runs on the worker thread, so it
            hands each send to the event loop via `run_coroutine_threadsafe`
            and blocks on the result -- keeps chunks in order and paced by
            actual network delivery, not fired-and-forgotten from a thread
            that's about to move on to the next sentence.
            """
            loop = asyncio.get_running_loop()
            chunker = SentenceChunker()

            def _on_delta(delta: str) -> None:
                if agent is None:
                    return
                for sentence in chunker.feed(delta):
                    audio = agent.synthesize(sentence)
                    asyncio.run_coroutine_threadsafe(_send_audio(audio), loop).result()

            event = await run_in_threadpool(call_with_on_delta, _on_delta if agent is not None else None)

            if agent is not None:
                tail = chunker.flush()
                if tail:
                    await _send_audio(agent.synthesize(tail))

            return event

        async def _drain_chapters() -> None:
            # Runs alongside the receive loop for a live (still-rendering)
            # session, forwarding each chapter's audio + span timings as
            # LookaheadRenderer finishes it (see session/buffer.py) -- a
            # thread -> queue.Queue -> run_in_threadpool bridge, since the
            # renderer produces chapters on its own background thread.
            while True:
                chapter_audio = await run_in_threadpool(chapter_queue.get)
                if chapter_audio is None:
                    return
                interactive_session.extend_transcript(chapter_audio.timings)
                wav_bytes = _audio_to_wav_bytes(chapter_audio.sample_rate, chapter_audio.audio)
                await _send(
                    {
                        "type": "chapter_ready",
                        "chapter_index": chapter_audio.chapter_index,
                        "title": chapter_audio.title,
                        "sample_rate": chapter_audio.sample_rate,
                        "data": base64.b64encode(wav_bytes).decode("ascii"),
                        "spans": [
                            {
                                "span_id": t.span_id,
                                "chapter_index": t.chapter_index,
                                "paragraph_index": t.paragraph_index,
                                "character": t.character,
                                "kind": t.kind.name,
                                "text": t.text,
                                "start_ms": t.start_ms,
                                "end_ms": t.end_ms,
                            }
                            for t in chapter_audio.timings
                        ],
                    }
                )

        drain_task = asyncio.create_task(_drain_chapters()) if chapter_queue is not None else None

        try:
            while True:
                message = await websocket.receive_json()
                msg_type = message.get("type")

                try:
                    if msg_type == "position":
                        span_id = message["span_id"]
                        event = await _speak_llm_call(
                            lambda on_delta, span_id=span_id: interactive_session.advance_position(
                                span_id, on_delta=on_delta
                            )
                        )
                        if event is not None:
                            await _send({"type": event.kind, "text": event.text, "sources": event.sources})

                    elif msg_type == "chapter_consumed":
                        if renderer is not None:
                            renderer.mark_consumed(int(message["chapter_index"]))

                    elif msg_type in ("chat", "choice"):
                        text = message.get("text", "")
                        event = await _speak_llm_call(
                            lambda on_delta, text=text: interactive_session.handle_user_message(
                                text, on_delta=on_delta
                            )
                        )
                        await _send({"type": "answer", "text": event.text, "sources": event.sources})

                    else:
                        await _send({"type": "error", "detail": f"Unknown message type: {msg_type!r}"})
                except Exception as exc:  # noqa: BLE001 - reported to the client, connection stays open
                    await _send({"type": "error", "detail": str(exc)})
        except WebSocketDisconnect:
            pass
        finally:
            if drain_task is not None:
                drain_task.cancel()
            if renderer is not None:
                renderer.stop()
            sessions.pop(session_id, None)

    @app.get("/api/book/transcript", dependencies=[Depends(_verify_request)])
    def book_transcript(path: str) -> Response:
        # `path` is the rendered .m4b's path; the transcript sidecar lives
        # alongside it at the same convention `render_book` writes to
        # (`output_path.with_suffix(".transcript.json")`) — see transcript.py.
        transcript_path = Path(path).with_suffix(".transcript.json")
        if not transcript_path.exists():
            raise HTTPException(status_code=404, detail=f"No transcript found for {path}")
        return Response(content=transcript_path.read_text(encoding="utf-8"), media_type="application/json")

    @app.get("/api/book/audio", dependencies=[Depends(_verify_request)])
    def book_audio(path: str) -> FileResponse:
        # FileResponse (not a raw Response) so Starlette serves Range
        # requests for scrubbing without us reimplementing that.
        audio_path = Path(path)
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail=f"No audio file found at {path}")
        return FileResponse(audio_path, media_type="audio/mp4")

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

    @app.get("/listen", response_class=HTMLResponse)
    def listen() -> str:
        return _LISTEN_HTML_PATH.read_text(encoding="utf-8")

    return app


_FRONTEND_HTML_PATH = Path(__file__).parent / "frontend" / "index.html"
_LISTEN_HTML_PATH = Path(__file__).parent / "frontend" / "listen.html"


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
