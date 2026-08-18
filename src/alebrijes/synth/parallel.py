"""Reusable helpers for parallel batch TTS synthesis.

pocket_tts's TTSModel.generate_audio is documented as NOT thread/process-safe to
share across concurrent callers: each concurrent generation needs its own model
instance. These helpers give every worker process its own TTSAgent (loaded once,
reused across jobs) instead of sharing one, and merge the resulting per-job WAVs
back into a single file.

Each worker process only needs ~1 CPU core: pocket_tts sets
`torch.set_num_threads(1)` internally, so sizing the worker pool near the core
count (rather than dividing cores across workers) is the right default.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
import wave

try:
    from rich.console import Group
    from rich.live import Live
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

# Two stats persisted between runs so the dashboard can show a useful ETA from
# the very first second, instead of waiting on completions in *this* run:
#
# - audio_seconds_per_char: how many seconds of speech a character of input
#   text tends to produce. This is basically a property of the voice/speaking
#   rate, not the machine, so it barely varies run to run and a job's *total*
#   expected audio length can be predicted from its text length alone.
# - wall_seconds_per_audio_second: how many wall-clock seconds this machine
#   needs to synthesize one second of audio (inverse of the real-time
#   factor). This is hardware/model dependent and is where the real "how much
#   longer" signal comes from.
#
# Best-effort throughout: any failure to read or write this file just means
# we fall back to a plain default guess.
_RATE_STATE_PATH = Path.home() / ".cache" / "alebrijes" / "synthesis_rate.json"
_DEFAULT_AUDIO_SECONDS_PER_CHAR = 0.08  # ~12.5 chars/sec, a typical spoken pace


def _load_historical_rates() -> dict:
    try:
        data = json.loads(_RATE_STATE_PATH.read_text())
        return {
            "audio_seconds_per_char": float(data.get("audio_seconds_per_char"))
            if data.get("audio_seconds_per_char")
            else None,
            "wall_seconds_per_audio_second": float(data.get("wall_seconds_per_audio_second"))
            if data.get("wall_seconds_per_audio_second")
            else None,
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {"audio_seconds_per_char": None, "wall_seconds_per_audio_second": None}


def _save_historical_rates(audio_seconds_per_char: Optional[float], wall_seconds_per_audio_second: Optional[float]) -> None:
    prior = _load_historical_rates()

    def _blend(new: Optional[float], old: Optional[float]) -> Optional[float]:
        if not new or new <= 0:
            return old
        # Exponential moving average: trust the new observation somewhat but
        # don't let one unusually fast/slow run swing the estimate wildly.
        return new if old is None else 0.7 * old + 0.3 * new

    blended = {
        "audio_seconds_per_char": _blend(audio_seconds_per_char, prior["audio_seconds_per_char"]),
        "wall_seconds_per_audio_second": _blend(
            wall_seconds_per_audio_second, prior["wall_seconds_per_audio_second"]
        ),
    }
    try:
        _RATE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _RATE_STATE_PATH.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(blended))
        os.replace(tmp_path, _RATE_STATE_PATH)
    except OSError:
        pass


def recommended_worker_count(job_count: int, *, reserve_cores: int = 1) -> int:
    """Suggest a worker-pool size for `job_count` independent synthesis jobs.

    Never suggests more workers than there are jobs (extra workers would just
    sit idle), and leaves `reserve_cores` free for the calling process/UI.
    """
    cpu_count = os.cpu_count() or 2
    workers = max(1, min(job_count, cpu_count - reserve_cores))
    LOGGER.info(
        "recommended_worker_count: job_count=%s cpu_count=%s reserve_cores=%s -> workers=%s",
        job_count,
        cpu_count,
        reserve_cores,
        workers,
    )
    return workers


def merge_wav_files(input_paths: list[Path], output_path: Path) -> Path:
    """Concatenate 16-bit PCM WAV files (same format) into one output file."""
    if not input_paths:
        raise ValueError("No cached audio files were provided for merge.")

    sample_rate: Optional[int] = None
    channel_count: Optional[int] = None
    sample_width: Optional[int] = None
    merged_frames = bytearray()

    for wav_path in input_paths:
        with wave.open(str(wav_path), "rb") as handle:
            if handle.getcomptype() != "NONE":
                raise ValueError(f"Cached audio is compressed and cannot be merged: {wav_path}")
            if handle.getsampwidth() != 2:
                raise ValueError(f"Cached audio must be 16-bit PCM: {wav_path}")

            current_rate = handle.getframerate()
            current_channels = handle.getnchannels()
            current_width = handle.getsampwidth()

            if sample_rate is None:
                sample_rate = current_rate
                channel_count = current_channels
                sample_width = current_width
            elif (
                current_rate != sample_rate
                or current_channels != channel_count
                or current_width != sample_width
            ):
                raise ValueError(f"Cached audio format mismatch for merge: {wav_path}")

            merged_frames.extend(handle.readframes(handle.getnframes()))

    assert sample_rate is not None
    assert channel_count is not None
    assert sample_width is not None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(channel_count)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(merged_frames))

    LOGGER.info("merge_wav_files: merged %s files -> %s", len(input_paths), output_path)
    return output_path


def _wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def _write_ffmetadata_chapters(
    metadata_path: Path, part_paths: list[Path], chapter_titles: list[str], book_title: Optional[str]
) -> None:
    lines = [";FFMETADATA1"]
    if book_title:
        escaped_title = book_title.replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#")
        lines.append(f"title={escaped_title}")

    cursor_ms = 0
    for wav_path, title in zip(part_paths, chapter_titles):
        start_ms = cursor_ms
        cursor_ms += round(_wav_duration_seconds(wav_path) * 1000)
        escaped = title.replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#")
        lines.extend(
            [
                "",
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start_ms}",
                f"END={cursor_ms}",
                f"title={escaped}",
            ]
        )

    metadata_path.write_text("\n".join(lines) + "\n")


def build_m4b(
    part_paths: list[Path],
    chapter_titles: list[str],
    output_path: Path,
    book_title: Optional[str] = None,
    bitrate: str = "64k",
) -> Path:
    """Concatenates per-chapter WAVs into a chaptered M4B audiobook.

    `part_paths` and `chapter_titles` must be the same length and in the
    desired chapter order. Uses the ffmpeg bundled via `imageio-ffmpeg`
    (see `alebrijes.ffmpeg_util`), not a system install.
    """
    if not part_paths:
        raise ValueError("No audio files were provided to build an M4B from.")
    if len(part_paths) != len(chapter_titles):
        raise ValueError(
            f"part_paths ({len(part_paths)}) and chapter_titles ({len(chapter_titles)}) must be the same length."
        )

    from alebrijes.ffmpeg_util import find_ffmpeg

    ffmpeg = find_ffmpeg()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix="pocket_tts_m4b_"))
    try:
        merged_wav = staging_dir / "merged.wav"
        merge_wav_files(part_paths, merged_wav)

        metadata_path = staging_dir / "chapters.txt"
        _write_ffmetadata_chapters(metadata_path, part_paths, chapter_titles, book_title)

        command = [
            ffmpeg,
            "-y",
            "-i",
            str(merged_wav),
            "-f",
            "ffmetadata",
            "-i",
            str(metadata_path),
            "-map",
            "0:a",
            "-map_metadata",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            bitrate,
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to build M4B: {stderr[-2000:]}") from exc
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    LOGGER.info("build_m4b: merged %s chapters -> %s", len(part_paths), output_path)
    return output_path


# --- Worker pool -----------------------------------------------------------

_worker_agent = None  # type: ignore[var-annotated]
_worker_status_path: Optional[Path] = None


def _write_worker_status(status_path: Optional[Path], **fields) -> None:
    """Overwrite this worker's status file with `fields` (best-effort).

    Written atomically (write to a temp file, then rename) so the poller in
    `run_parallel_synthesis` never reads a half-written file. Used to drive
    the live per-worker dashboard; failures here must never break synthesis.
    """
    if status_path is None:
        return
    payload = {"pid": os.getpid(), "updated_at": time.time(), **fields}
    tmp_path = status_path.with_suffix(status_path.suffix + ".tmp")
    try:
        with open(tmp_path, "w") as handle:
            json.dump(payload, handle)
        os.replace(tmp_path, status_path)
    except OSError:
        pass


def init_pool_worker(voice_state_path: str, log_dir: Optional[str] = None) -> None:
    """ProcessPoolExecutor initializer: load one TTSAgent per worker process.

    `ProcessPoolExecutor` forks, so each worker starts out with the parent's
    already-configured `logging` handlers (e.g. a `StreamHandler` bound to the
    parent's stderr). pocket_tts's own progress messages (timer starts,
    generation speed, ...) go through `logging.getLogger(__name__).info(...)`
    internally, not `print()`, so redirecting this process's stdout/stderr
    would not touch them. When `log_dir` is given, replace this worker's root
    logger handlers with a `FileHandler` writing to `log_dir/worker_<pid>.log`
    so each worker's trace lands in its own file (tailable with `multitail`)
    instead of all workers interleaving into one shared terminal stream.
    """
    global _worker_agent, _worker_status_path
    from alebrijes.synth.agent import TTSAgent

    if log_dir is not None:
        log_path = Path(log_dir) / f"worker_{os.getpid()}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s"))
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)

        _worker_status_path = Path(log_dir) / f"worker_{os.getpid()}.status.json"
        _write_worker_status(_worker_status_path, label=None, status="loading-model")

    def _load_voice_state(state_path: Path):
        import torch

        try:
            return torch.load(state_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(state_path, map_location="cpu")

    voice_state = _load_voice_state(Path(voice_state_path))
    _worker_agent = TTSAgent(voice=voice_state_path, voice_state=voice_state)
    LOGGER.info("init_pool_worker: loaded model pid=%s voice=%s", os.getpid(), voice_state_path)
    _write_worker_status(_worker_status_path, label=None, status="idle")


def _cache_key_for(text: str, voice_state_path: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(text.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(voice_state_path.encode("utf-8"))
    return hasher.hexdigest()


def synthesize_job(
    text: str,
    output_path: str,
    voice_state_path: str,
    cache_dir: Optional[str] = None,
    label: Optional[str] = None,
) -> dict:
    """Runs in a worker process: synthesize `text` and write it to `output_path`.

    Returns stats ({"chars", "duration", "audio_seconds", "cache_hit"}) so the
    parent process can turn them into observed synthesis-speed estimates for
    the ETA display.
    """
    agent = _worker_agent
    if agent is None:
        raise RuntimeError("Pool worker was not initialized (missing init_pool_worker).")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    char_count = len(text)

    cache_wav_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_key = _cache_key_for(text, voice_state_path)
        cache_wav_path = cache_dir_path / f"{cache_key}.wav"
        if cache_wav_path.exists():
            shutil.copy2(cache_wav_path, out_path)
            LOGGER.info("synthesize_job: cache hit pid=%s -> %s", os.getpid(), out_path)
            _write_worker_status(_worker_status_path, label=label, status="cache-hit")
            return {"chars": char_count, "duration": 0.0, "audio_seconds": 0.0, "cache_hit": True}

    if label is not None:
        LOGGER.info("synthesize_job: pid=%s starting job=%s", os.getpid(), label)
    started_at = time.time()
    _write_worker_status(
        _worker_status_path,
        label=label,
        status="running",
        started_at=started_at,
        chars=char_count,
        audio_seconds=0.0,
    )

    def _on_progress(audio_seconds_so_far: float) -> None:
        # Called from inside agent.write_wav as each sentence-chunk of audio
        # is decoded, giving a real (not guessed) measure of how far through
        # this job we are.
        _write_worker_status(
            _worker_status_path,
            label=label,
            status="running",
            started_at=started_at,
            chars=char_count,
            audio_seconds=audio_seconds_so_far,
        )

    tmp_path = out_path.with_suffix(f".{os.getpid()}.tmp.wav")
    agent.write_wav(tmp_path, text, on_progress=_on_progress)
    os.replace(tmp_path, out_path)
    duration = time.time() - started_at

    with wave.open(str(out_path), "rb") as handle:
        audio_seconds = handle.getnframes() / handle.getframerate()

    if cache_wav_path is not None:
        try:
            shutil.copy2(out_path, cache_wav_path)
        except Exception:
            LOGGER.exception("synthesize_job: failed to write cache entry %s", cache_wav_path)

    LOGGER.info("synthesize_job: pid=%s wrote %s in %.1fs", os.getpid(), out_path, duration)
    _write_worker_status(_worker_status_path, label=label, status="done")
    return {"chars": char_count, "duration": duration, "audio_seconds": audio_seconds, "cache_hit": False}


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def _read_worker_statuses(logs_dir: Path) -> list[dict]:
    statuses = []
    for status_file in sorted(logs_dir.glob("worker_*.status.json")):
        try:
            statuses.append(json.loads(status_file.read_text()))
        except (OSError, ValueError):
            continue
    return statuses


def _build_worker_table(statuses: list[dict], audio_seconds_per_char: float, wall_seconds_per_audio_second: Optional[float]) -> "Table":
    """Render a snapshot of every worker's current job from its status file.

    Progress is driven by real audio generated so far (`audio_seconds`,
    updated live as pocket_tts decodes each sentence-chunk), not a guess: a
    job's expected total audio length comes from its text length via
    `audio_seconds_per_char`, and `wall_seconds_per_audio_second` (this
    machine's observed synthesis speed) converts the remaining audio into a
    remaining wall-clock estimate.
    """
    table = Table(expand=True)
    table.add_column("PID", justify="right")
    table.add_column("Job")
    table.add_column("Status")
    table.add_column("Elapsed", justify="right")
    table.add_column("Progress", justify="right")
    table.add_column("Est. left", justify="right")

    now = time.time()
    for data in statuses:
        status = data.get("status", "?")
        started_at = data.get("started_at")
        chars = data.get("chars")
        audio_seconds = data.get("audio_seconds")
        elapsed_secs = now - started_at if status == "running" and started_at else None
        elapsed = f"{elapsed_secs:.0f}s" if elapsed_secs is not None else "-"

        progress_str = "-"
        est_left = "-"
        if status == "running" and chars:
            expected_audio_total = chars * audio_seconds_per_char
            if audio_seconds and expected_audio_total > 0:
                fraction = min(0.99, audio_seconds / expected_audio_total)
                progress_str = f"{fraction * 100:.0f}%"
                if wall_seconds_per_audio_second:
                    remaining_audio = max(0.0, expected_audio_total - audio_seconds)
                    remaining = remaining_audio * wall_seconds_per_audio_second
                    est_left = f"~{_format_duration(remaining)}" if remaining > 1 else "any moment"
            elif audio_seconds == 0.0:
                progress_str = "warming up"

        table.add_row(
            str(data.get("pid", "?")), str(data.get("label") or "-"), status, elapsed, progress_str, est_left
        )
    return table


def _merge_parts(
    part_paths: list[Path],
    output_path: Path,
    output_format: str,
    chapter_titles: Optional[list[str]],
    book_title: Optional[str],
) -> Path:
    if output_format == "m4b":
        if chapter_titles is None or len(chapter_titles) != len(part_paths):
            raise ValueError(
                "chapter_titles must be provided and match jobs 1:1 when output_format='m4b'."
            )
        return build_m4b(part_paths, chapter_titles, output_path, book_title=book_title)
    return merge_wav_files(part_paths, output_path)


def run_parallel_synthesis(
    jobs: list[tuple[str, str]],
    *,
    voice_state_path: str,
    output_path: Path,
    max_workers: int,
    cache_dir: Optional[Path] = None,
    output_format: str = "wav",
    chapter_titles: Optional[list[str]] = None,
    book_title: Optional[str] = None,
) -> Path:
    """Synthesize each (label, text) job and merge the results into `output_path`.

    `jobs` is a list of (label, text) pairs, in the desired output order (label
    is only used for logging/ordering, e.g. a chapter number). Falls back to
    running sequentially, in-process, when parallelism wouldn't help.

    `output_format` is "wav" (default, plain concatenation) or "m4b" (chaptered
    AAC audiobook via ffmpeg); when "m4b", `chapter_titles` must be given, one
    per job, in the same order as `jobs`.
    """
    if not jobs:
        raise ValueError("No synthesis jobs were provided.")

    if max_workers <= 1 or len(jobs) <= 1:
        reason = "max_workers <= 1" if max_workers <= 1 else "only one job"
        LOGGER.info(
            "run_parallel_synthesis: falling back to sequential synthesis (%s); job_count=%s",
            reason,
            len(jobs),
        )
        init_pool_worker(voice_state_path)
        staging_dir = Path(tempfile.mkdtemp(prefix="pocket_tts_parallel_"))
        try:
            part_paths = []
            for label, text in jobs:
                part_path = staging_dir / f"{label}.wav"
                synthesize_job(
                    text, str(part_path), voice_state_path, str(cache_dir) if cache_dir else None, label
                )
                part_paths.append(part_path)
            return _merge_parts(part_paths, output_path, output_format, chapter_titles, book_title)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    LOGGER.info(
        "run_parallel_synthesis: starting parallel synthesis: %s jobs across %s worker processes",
        len(jobs),
        max_workers,
    )
    staging_dir = Path(tempfile.mkdtemp(prefix="pocket_tts_parallel_"))
    logs_dir = staging_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "run_parallel_synthesis: per-worker logs at %s (raw traces, if you need to dig into one job)",
        logs_dir,
    )

    # Prefer a live progress dashboard over scrolling per-worker log lines,
    # but only when there's a real terminal to draw it on; a piped/redirected
    # stdout falls back to the plain "completed X/Y" log lines below.
    use_dashboard = _RICH_AVAILABLE and sys.stdout.isatty()
    progress: Optional["Progress"] = None
    task_id = None
    live: Optional["Live"] = None
    stop_poll = threading.Event()
    poll_thread: Optional[threading.Thread] = None

    total_chars = sum(len(text) for _label, text in jobs)
    # Seed both rates from previous runs on this machine so the dashboard is
    # useful from second one, instead of showing "-" until a job finishes.
    historical = _load_historical_rates()
    rate_lock = threading.Lock()
    rate_state = {
        "audio_seconds_per_char": historical["audio_seconds_per_char"] or _DEFAULT_AUDIO_SECONDS_PER_CHAR,
        "wall_seconds_per_audio_second": historical["wall_seconds_per_audio_second"],
        "completed_chars": 0,
        "completed_wall_seconds": 0.0,
        "completed_audio_seconds": 0.0,
    }

    def _current_speed(statuses: list[dict]) -> tuple[float, Optional[float]]:
        """Blend completed-job totals with live in-progress audio to get the
        current audio_seconds_per_char / wall_seconds_per_audio_second
        estimates, so the speed estimate improves continuously instead of
        jumping only when a job finishes."""
        with rate_lock:
            audio_seconds_per_char = rate_state["audio_seconds_per_char"]
            wall_seconds = rate_state["completed_wall_seconds"]
            audio_seconds = rate_state["completed_audio_seconds"]
        now = time.time()
        for data in statuses:
            if data.get("status") != "running":
                continue
            started_at = data.get("started_at")
            live_audio = data.get("audio_seconds")
            if started_at and live_audio:
                wall_seconds += now - started_at
                audio_seconds += live_audio
        wall_per_audio = (wall_seconds / audio_seconds) if audio_seconds > 0 else None
        return audio_seconds_per_char, wall_per_audio

    def _eta_text(statuses: list[dict]) -> str:
        audio_seconds_per_char, wall_per_audio = _current_speed(statuses)
        with rate_lock:
            remaining_chars = max(0, total_chars - rate_state["completed_chars"])

        # Running jobs contribute only their *remaining* expected audio (their
        # chars are still part of `remaining_chars` above, so subtract those
        # out first to avoid double-counting them below).
        remaining_audio = 0.0
        for data in statuses:
            if data.get("status") == "running" and data.get("chars"):
                chars = data["chars"]
                remaining_chars = max(0, remaining_chars - chars)
                expected_total = chars * audio_seconds_per_char
                audio_so_far = data.get("audio_seconds") or 0.0
                remaining_audio += max(0.0, expected_total - audio_so_far)
        remaining_audio += remaining_chars * audio_seconds_per_char

        if not wall_per_audio:
            return "ETA -:--:--"
        active_workers = max(1, sum(1 for data in statuses if data.get("status") == "running"))
        eta_seconds = remaining_audio * wall_per_audio / active_workers
        speed = f"{1 / wall_per_audio:.1f}x realtime"
        return f"{speed} · ETA {_format_duration(eta_seconds)}"

    if use_dashboard:
        progress = Progress(
            TextColumn("[bold]Synthesizing[/bold]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} jobs"),
            TimeElapsedColumn(),
            TextColumn("{task.fields[eta]}"),
        )
        initial_statuses = _read_worker_statuses(logs_dir)
        task_id = progress.add_task("jobs", total=len(jobs), eta=_eta_text(initial_statuses))

        def _render() -> "Group":
            statuses = _read_worker_statuses(logs_dir)
            audio_seconds_per_char, wall_per_audio = _current_speed(statuses)
            progress.update(task_id, eta=_eta_text(statuses))
            return Group(progress, _build_worker_table(statuses, audio_seconds_per_char, wall_per_audio))

        live = Live(_render(), refresh_per_second=4)
        live.start()

        def _poll_workers() -> None:
            while not stop_poll.wait(0.25):
                live.update(_render())

        poll_thread = threading.Thread(target=_poll_workers, daemon=True)
        poll_thread.start()

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_pool_worker,
            initargs=(voice_state_path, str(logs_dir)),
        ) as executor:
            future_to_label = {}
            for label, text in jobs:
                part_path = staging_dir / f"{label}.wav"
                future = executor.submit(
                    synthesize_job,
                    text,
                    str(part_path),
                    voice_state_path,
                    str(cache_dir) if cache_dir else None,
                    label,
                )
                future_to_label[future] = label

            completed = 0
            for future in as_completed(future_to_label):
                label = future_to_label[future]
                try:
                    result = future.result()
                except Exception:
                    LOGGER.exception("run_parallel_synthesis: job %s failed", label)
                    raise
                completed += 1
                with rate_lock:
                    rate_state["completed_chars"] += result["chars"]
                    if not result["cache_hit"] and result["duration"] > 0 and result["audio_seconds"] > 0:
                        rate_state["completed_wall_seconds"] += result["duration"]
                        rate_state["completed_audio_seconds"] += result["audio_seconds"]
                        observed_ratio = result["audio_seconds"] / result["chars"]
                        rate_state["audio_seconds_per_char"] = (
                            0.7 * rate_state["audio_seconds_per_char"] + 0.3 * observed_ratio
                        )
                if use_dashboard:
                    progress.update(task_id, completed=completed, eta=_eta_text(_read_worker_statuses(logs_dir)))
                else:
                    LOGGER.info(
                        "run_parallel_synthesis: completed %s/%s (job=%s)",
                        completed,
                        len(jobs),
                        label,
                    )

        part_paths = [staging_dir / f"{label}.wav" for label, _text in jobs]
        merged_path = _merge_parts(part_paths, output_path, output_format, chapter_titles, book_title)
    except Exception:
        LOGGER.error("run_parallel_synthesis: aborting merge; staging dir kept at %s", staging_dir)
        raise
    finally:
        if poll_thread is not None:
            stop_poll.set()
            poll_thread.join(timeout=1)
        with rate_lock:
            completed_wall_seconds = rate_state["completed_wall_seconds"]
            completed_audio_seconds = rate_state["completed_audio_seconds"]
            audio_seconds_per_char = rate_state["audio_seconds_per_char"]
        if completed_audio_seconds > 0:
            _save_historical_rates(audio_seconds_per_char, completed_wall_seconds / completed_audio_seconds)
        if live is not None:
            final_statuses = _read_worker_statuses(logs_dir)
            wall_per_audio = (
                completed_wall_seconds / completed_audio_seconds if completed_audio_seconds > 0 else None
            )
            live.update(Group(progress, _build_worker_table(final_statuses, audio_seconds_per_char, wall_per_audio)))
            live.stop()

    shutil.rmtree(staging_dir, ignore_errors=True)
    return merged_path
