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
from alebrijes.ingest import Book, Chapter
from alebrijes.segment import Span, SpanKind, segment_paragraph
from alebrijes.synth.agent import TTSAgent, write_wav_array
from alebrijes.synth.parallel import build_m4b

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
    attributions_by_span = {id(a.span): a for a in RuleBasedAttributor().attribute(spans)}
    narrator_voice = cast_file.voice_for(NARRATOR)

    voiced: list[VoicedSpan] = []
    for span in spans:
        if span.kind is SpanKind.NARRATION:
            voiced.append(VoicedSpan(span, NARRATOR, narrator_voice))
            continue

        attribution = attributions_by_span.get(id(span))
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
) -> np.ndarray:
    """Synthesizes every span in order, swapping voices per line as needed."""
    rendered: list[np.ndarray] = []
    pause_samples = int(agent.sample_rate * (pause_ms / 1000.0))

    for voiced in voiced_spans:
        voice_prompt, state = voice_states[voiced.character]
        agent.use_voice_state(voice_prompt, state)
        audio = agent.synthesize(voiced.span.text)
        if audio.size == 0:
            continue
        rendered.append(audio)
        if pause_samples > 0:
            rendered.append(np.zeros(pause_samples, dtype=audio.dtype))

    if not rendered:
        raise RuntimeError("No audio rendered from chapter.")
    return np.concatenate(rendered)


def render_book(
    book: Book,
    cast_file: CastFile,
    output_path: Path,
    *,
    pause_ms: int = DEFAULT_PAUSE_MS,
    model: object | None = None,
    on_chapter_done: Callable[[int, int], None] | None = None,
) -> Path:
    """Renders `book` with `cast_file`'s voices to a chaptered .m4b at `output_path`.

    `on_chapter_done(completed, total)` is called after each chapter finishes
    synthesizing — chapter-level granularity only (not per-line), since that's
    what's available without restructuring `synthesize_chapter`'s sequential
    loop. Good enough for a progress bar; not real-time per-line progress.
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

    staging_dir = Path(tempfile.mkdtemp(prefix="alebrijes_render_"))
    try:
        part_paths: list[Path] = []
        chapter_titles: list[str] = []
        total_chapters = len(book.chapters)
        for chapter in book.chapters:
            voiced_spans = resolve_chapter_spans(chapter, cast_file)
            audio = synthesize_chapter(voiced_spans, voice_states, agent, pause_ms=pause_ms)

            part_path = staging_dir / f"chapter_{chapter.index:04d}.wav"
            write_wav_array(part_path, agent.sample_rate, audio)
            part_paths.append(part_path)
            chapter_titles.append(chapter.title or f"Chapter {chapter.index + 1}")
            if on_chapter_done is not None:
                on_chapter_done(len(part_paths), total_chapters)

        return build_m4b(part_paths, chapter_titles, output_path, book_title=book.title)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
