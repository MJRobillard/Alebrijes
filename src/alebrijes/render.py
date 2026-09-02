"""Orchestrates cast + synth + assemble into a finished, multi-voice audiobook.

Not one of the six pipeline stages itself — this is the CLI-level glue that
walks a book chapter by chapter, resolves each span's voice, and drives
synthesis + assembly. See PHASES.md for the integration assumptions this was
built against, in particular:

- `alebrijes.synth.parallel.run_parallel_synthesis` assumes one voice for its
  whole worker pool (built for the old repo's single-narrator-per-book case),
  so it isn't reused here. Instead, `synthesize_chapter` follows the proven
  pattern from pocket-tts-agent's `examples/multi_ci_tts.py`
  (`_prepare_voice_states` / `_synthesize_segments`): every cast voice is
  resolved once, then one `TTSAgent`'s cached state is swapped
  (`use_voice_state`) before each line and synthesis runs sequentially. This
  is slower than the parallel single-voice path — an accepted v1 trade-off,
  not a hidden one.
- Attribution is re-derived at render time (deterministic, so this
  reproduces identically) rather than persisted in `cast.yaml`.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from alebrijes.attribute.rules import UNKNOWN, RuleBasedAttributor
from alebrijes.cast import CastFile
from alebrijes.ffmpeg_util import find_ffmpeg
from alebrijes.ingest import Book, Chapter
from alebrijes.segment import Span, SpanKind, segment_paragraph
from alebrijes.synth.agent import TTSAgent, write_wav_array
from alebrijes.synth.parallel import build_m4b
from alebrijes.transcript import ChapterMark, SpanTiming, Transcript
from alebrijes.transcript import save as save_transcript

NARRATOR = "Narrator"
DEFAULT_PAUSE_MS = 250


@dataclass
class VoicedSpan:
    span: Span
    character: str
    voice_prompt: str


def resolve_chapter_spans(chapter: Chapter, cast_file: CastFile) -> list[VoicedSpan]:
    """Segments and attributes one chapter, then resolves each span to a voice.

    Narration always speaks in the Narrator's voice. Dialogue speaks in its
    attributed character's voice, falling back to the Narrator's voice when
    attribution is Unknown, or when it names a character the cast file
    doesn't have an entry for (attribution can find names the cast-review
    step hasn't seen/assigned yet).
    """
    spans = [span for paragraph in chapter.paragraphs for span in segment_paragraph(paragraph, chapter.index)]
    attributions_by_span = {a.span.span_id: a for a in RuleBasedAttributor().attribute(spans)}
    narrator_voice = cast_file.voice_for(NARRATOR)

    voiced: list[VoicedSpan] = []
    for span in spans:
        if span.kind is SpanKind.NARRATION:
            voiced.append(VoicedSpan(span, NARRATOR, narrator_voice))
            continue

        attribution = attributions_by_span.get(span.span_id)
        character = attribution.character if attribution else UNKNOWN
        if character == UNKNOWN:
            voiced.append(VoicedSpan(span, NARRATOR, narrator_voice))
            continue
        try:
            voice_prompt = cast_file.voice_for(character)
        except KeyError:
            voiced.append(VoicedSpan(span, NARRATOR, narrator_voice))
            continue
        voiced.append(VoicedSpan(span, character, voice_prompt))

    return voiced


def _load_voice_state(state_path: Path) -> object:
    import torch

    try:
        return torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(state_path, map_location="cpu")


def resolve_voice_states(cast_file: CastFile, model: object) -> dict[str, tuple[str, object]]:
    """Loads or computes a (voice_prompt, state) pair per cast character, once.

    `voice_state_path` is overloaded (see module docstring): an existing path
    is a saved `.pt` voice-state file; anything else is treated as a pocket-tts
    builtin voice name and resolved through the model directly.
    """
    states: dict[str, tuple[str, object]] = {}
    resolved_cache: dict[str, object] = {}

    for entry in cast_file.entries:
        voice_key = entry.voice_state_path
        if voice_key not in resolved_cache:
            path = Path(voice_key)
            if path.exists():
                resolved_cache[voice_key] = _load_voice_state(path)
            else:
                resolved_cache[voice_key] = model.get_state_for_audio_prompt(voice_key)
        states[entry.character] = (voice_key, resolved_cache[voice_key])

    return states


def synthesize_chapter(
    voiced_spans: list[VoicedSpan],
    voice_states: dict[str, tuple[str, object]],
    agent: TTSAgent,
    pause_ms: int = DEFAULT_PAUSE_MS,
) -> tuple[np.ndarray, list[SpanTiming]]:
    """Synthesizes every span in order, swapping voices per line as needed.

    Returns the concatenated chapter audio plus each span's (start_ms, end_ms)
    within that audio (pauses excluded from the span's own span, but counted
    toward the cursor so later spans still land at their real offset).
    """
    rendered: list[np.ndarray] = []
    timings: list[SpanTiming] = []
    pause_samples = int(agent.sample_rate * (pause_ms / 1000.0))
    cursor_samples = 0

    for voiced in voiced_spans:
        voice_prompt, state = voice_states[voiced.character]
        agent.use_voice_state(voice_prompt, state)
        audio = agent.synthesize(voiced.span.text)
        if audio.size == 0:
            continue
        start_ms = round(cursor_samples / agent.sample_rate * 1000)
        cursor_samples += audio.size
        end_ms = round(cursor_samples / agent.sample_rate * 1000)
        timings.append(
            SpanTiming(
                span_id=voiced.span.span_id,
                chapter_index=voiced.span.chapter_index,
                paragraph_index=voiced.span.paragraph_index,
                character=voiced.character,
                kind=voiced.span.kind,
                text=voiced.span.text,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
        rendered.append(audio)
        if pause_samples > 0:
            rendered.append(np.zeros(pause_samples, dtype=audio.dtype))
            cursor_samples += pause_samples

    if not rendered:
        raise RuntimeError("No audio rendered from chapter.")
    return np.concatenate(rendered), timings


def build_render_agent(cast_file: CastFile, model: object | None = None) -> tuple[TTSAgent, dict[str, tuple[str, object]]]:
    """Loads (or reuses) a pocket-tts model and resolves every cast voice
    once, returning a `TTSAgent` primed with the Narrator's voice plus the
    full `voice_states` map `synthesize_chapter` needs to swap voices per
    line. Shared by `render_book` and by a live interactive session
    (`server.py`), which both need this same setup before synthesizing.
    """
    if model is None:
        from pocket_tts import TTSModel

        try:
            model = TTSModel.load_model()
        except AttributeError:
            model = TTSModel()

    voice_states = resolve_voice_states(cast_file, model)
    narrator_prompt, narrator_state = voice_states[NARRATOR]
    agent = TTSAgent(voice=narrator_prompt, model=model, voice_state=narrator_state)
    return agent, voice_states


def render_book(
    book: Book,
    cast_file: CastFile,
    output_path: Path,
    *,
    pause_ms: int = DEFAULT_PAUSE_MS,
    model: object | None = None,
    on_chapter_done: Callable[[int, int], None] | None = None,
    transcript_path: Path | None = None,
) -> Path:
    """Renders `book` with `cast_file`'s voices to a chaptered .m4b at `output_path`.

    `on_chapter_done(completed, total)` is called after each chapter finishes
    synthesizing — chapter-level granularity only (not per-line), since that's
    what's available without restructuring `synthesize_chapter`'s sequential
    loop. Good enough for a progress bar; not real-time per-line progress.

    Also writes a `Transcript` sidecar (`transcript_path`, defaulting to
    `output_path.with_suffix(".transcript.json")`) mapping every span to its
    whole-book audio position, for a live "now narrating" player view and
    position-bounded retrieval.
    """
    # build_m4b (the very last step) already raises a clear RuntimeError if
    # ffmpeg is missing, but by then every chapter has already been
    # synthesized — the slow part — and the staging dir holding that work
    # gets deleted by the `finally` below regardless. Check up front instead
    # so a missing ffmpeg fails in milliseconds, not after however long a
    # full multi-chapter render takes.
    find_ffmpeg()

    agent, voice_states = build_render_agent(cast_file, model)

    staging_dir = Path(tempfile.mkdtemp(prefix="alebrijes_render_"))
    try:
        part_paths: list[Path] = []
        chapter_titles: list[str] = []
        chapter_marks: list[ChapterMark] = []
        all_timings: list[SpanTiming] = []
        total_chapters = len(book.chapters)
        book_cursor_ms = 0
        for chapter in book.chapters:
            voiced_spans = resolve_chapter_spans(chapter, cast_file)
            audio, chapter_timings = synthesize_chapter(voiced_spans, voice_states, agent, pause_ms=pause_ms)

            part_path = staging_dir / f"chapter_{chapter.index:04d}.wav"
            write_wav_array(part_path, agent.sample_rate, audio)
            part_paths.append(part_path)
            title = chapter.title or f"Chapter {chapter.index + 1}"
            chapter_titles.append(title)

            chapter_start_ms = book_cursor_ms
            chapter_duration_ms = round(audio.size / agent.sample_rate * 1000)
            book_cursor_ms += chapter_duration_ms
            chapter_marks.append(ChapterMark(chapter.index, title, chapter_start_ms, book_cursor_ms))
            for timing in chapter_timings:
                timing.start_ms += chapter_start_ms
                timing.end_ms += chapter_start_ms
            all_timings.extend(chapter_timings)

            if on_chapter_done is not None:
                on_chapter_done(len(part_paths), total_chapters)

        m4b_path = build_m4b(part_paths, chapter_titles, output_path, book_title=book.title)

        resolved_transcript_path = transcript_path or output_path.with_suffix(".transcript.json")
        save_transcript(
            Transcript(book_title=book.title, chapters=chapter_marks, spans=all_timings),
            resolved_transcript_path,
        )

        return m4b_path
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
