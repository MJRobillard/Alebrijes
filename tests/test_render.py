from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alebrijes.cast import CastEntry, CastFile
from alebrijes.ingest import Chapter, Paragraph
from alebrijes.render import NARRATOR, resolve_chapter_spans, resolve_voice_states, synthesize_chapter
from alebrijes.segment import SpanKind


def _cast_file() -> CastFile:
    return CastFile(
        book_title="Fixture Book",
        entries=[
            CastEntry(character=NARRATOR, voice_state_path="alba"),
            CastEntry(character="Colin", voice_state_path="javert"),
        ],
    )


def test_narration_resolves_to_narrator_voice() -> None:
    chapter = Chapter(title="Ch1", index=0, paragraphs=[Paragraph(text="The room was quiet.", index=0)])

    voiced = resolve_chapter_spans(chapter, _cast_file())

    assert len(voiced) == 1
    assert voiced[0].character == NARRATOR
    assert voiced[0].voice_prompt == "alba"
    assert voiced[0].span.kind is SpanKind.NARRATION


def test_dialogue_resolves_to_attributed_character_voice() -> None:
    chapter = Chapter(
        title="Ch1",
        index=0,
        paragraphs=[Paragraph(text='Colin said, "Let\'s take another path."', index=0)],
    )

    voiced = resolve_chapter_spans(chapter, _cast_file())

    dialogue = [v for v in voiced if v.span.kind is SpanKind.DIALOGUE]
    assert len(dialogue) == 1
    assert dialogue[0].character == "Colin"
    assert dialogue[0].voice_prompt == "javert"


def test_unresolvable_dialogue_falls_back_to_narrator_voice() -> None:
    chapter = Chapter(
        title="Ch1",
        index=0,
        paragraphs=[Paragraph(text='Someone spoke from the dark: "Where has everyone gone?"', index=0)],
    )

    voiced = resolve_chapter_spans(chapter, _cast_file())

    dialogue = [v for v in voiced if v.span.kind is SpanKind.DIALOGUE]
    assert len(dialogue) == 1
    assert dialogue[0].character == NARRATOR
    assert dialogue[0].voice_prompt == "alba"


def test_character_missing_from_cast_falls_back_to_narrator_voice() -> None:
    cast_file = CastFile(book_title="Fixture Book", entries=[CastEntry(character=NARRATOR, voice_state_path="alba")])
    chapter = Chapter(
        title="Ch1",
        index=0,
        paragraphs=[Paragraph(text='Colin said, "Let\'s take another path."', index=0)],
    )

    voiced = resolve_chapter_spans(chapter, cast_file)

    dialogue = [v for v in voiced if v.span.kind is SpanKind.DIALOGUE]
    assert dialogue[0].character == NARRATOR
    assert dialogue[0].voice_prompt == "alba"


class _FakeModel:
    def get_state_for_audio_prompt(self, conditioning: object) -> object:
        return {"conditioning": conditioning}


def test_resolve_voice_states_treats_existing_path_as_pt_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    voice_path = tmp_path / "colin.pt"
    voice_path.write_bytes(b"fake-state")
    monkeypatch.setattr("alebrijes.render._load_voice_state", lambda path: {"loaded_from": str(path)})

    cast_file = CastFile(
        book_title="Fixture Book",
        entries=[
            CastEntry(character=NARRATOR, voice_state_path="alba"),
            CastEntry(character="Colin", voice_state_path=str(voice_path)),
        ],
    )

    states = resolve_voice_states(cast_file, _FakeModel())

    assert states[NARRATOR] == ("alba", {"conditioning": "alba"})
    assert states["Colin"] == (str(voice_path), {"loaded_from": str(voice_path)})


def test_resolve_voice_states_caches_repeated_voice_keys() -> None:
    class CountingModel:
        def __init__(self) -> None:
            self.calls = 0

        def get_state_for_audio_prompt(self, conditioning: object) -> object:
            self.calls += 1
            return {"conditioning": conditioning}

    model = CountingModel()
    cast_file = CastFile(
        book_title="Fixture Book",
        entries=[
            CastEntry(character=NARRATOR, voice_state_path="alba"),
            CastEntry(character="Colin", voice_state_path="alba"),
        ],
    )

    resolve_voice_states(cast_file, model)

    assert model.calls == 1


class _RecordingAgent:
    sample_rate = 1000

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []  # (voice_prompt, text)
        self._voice_prompt: str | None = None

    def use_voice_state(self, voice: str, state: object) -> None:
        self._voice_prompt = voice

    def synthesize(self, text: str) -> np.ndarray:
        self.calls.append((self._voice_prompt, text))
        return np.ones(4, dtype=np.int16)


def test_synthesize_chapter_preserves_order_and_uses_correct_voice_per_line() -> None:
    from alebrijes.render import VoicedSpan
    from alebrijes.segment import Span

    voiced_spans = [
        VoicedSpan(Span("Narration one.", SpanKind.NARRATION, 0, 0, "0.0.0"), NARRATOR, "alba"),
        VoicedSpan(Span("Dialogue line.", SpanKind.DIALOGUE, 0, 1, "0.1.0"), "Colin", "javert"),
        VoicedSpan(Span("Narration two.", SpanKind.NARRATION, 0, 2, "0.2.0"), NARRATOR, "alba"),
    ]
    voice_states = {NARRATOR: ("alba", object()), "Colin": ("javert", object())}
    agent = _RecordingAgent()

    audio, timings = synthesize_chapter(voiced_spans, voice_states, agent, pause_ms=0)

    assert agent.calls == [
        ("alba", "Narration one."),
        ("javert", "Dialogue line."),
        ("alba", "Narration two."),
    ]
    assert audio.size == 4 * 3  # three lines, no pause samples requested
    assert [t.span_id for t in timings] == ["0.0.0", "0.1.0", "0.2.0"]
    assert [t.character for t in timings] == [NARRATOR, "Colin", NARRATOR]


def test_synthesize_chapter_inserts_silence_between_lines() -> None:
    from alebrijes.render import VoicedSpan
    from alebrijes.segment import Span

    voiced_spans = [
        VoicedSpan(Span("One.", SpanKind.NARRATION, 0, 0, "0.0.0"), NARRATOR, "alba"),
        VoicedSpan(Span("Two.", SpanKind.NARRATION, 0, 1, "0.1.0"), NARRATOR, "alba"),
    ]
    voice_states = {NARRATOR: ("alba", object())}
    agent = _RecordingAgent()
    agent.sample_rate = 1000

    audio, timings = synthesize_chapter(voiced_spans, voice_states, agent, pause_ms=250)

    # A pause follows every rendered line (including the last), matching
    # pocket-tts-agent's _synthesize_segments: 2 lines x (4 audio + 250 pause).
    assert audio.size == (4 + 250) * 2


def test_synthesize_chapter_raises_when_nothing_rendered() -> None:
    agent = _RecordingAgent()

    with pytest.raises(RuntimeError, match="No audio rendered"):
        synthesize_chapter([], {}, agent)


def test_render_book_fails_fast_when_ffmpeg_missing_before_any_synthesis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from alebrijes.ingest import Book
    from alebrijes.render import render_book

    def _raise():
        raise RuntimeError("ffmpeg could not be located via imageio-ffmpeg: fake failure")

    monkeypatch.setattr("alebrijes.render.find_ffmpeg", _raise)

    # No model= passed on purpose: if the ffmpeg check didn't run first,
    # this would go on to actually load the real pocket-tts model (slow,
    # network-dependent) instead of failing immediately with the clear
    # ffmpeg error -- so a fast, clean failure here proves the check runs
    # before any of that, not just that it exists somewhere in the function.
    book = Book(title="Fixture", chapters=[])
    cast_file = CastFile(book_title="Fixture", entries=[CastEntry(character=NARRATOR, voice_state_path="alba")])

    with pytest.raises(RuntimeError, match="could not be located"):
        render_book(book, cast_file, tmp_path / "out.m4b")
