"""Adapted from pocket-tts-agent's tests/test_agent_voice_prompt.py.

Same assertions against the ported `alebrijes.synth.agent` module — this is
what proves the port (import-path updates only) didn't change behavior.
"""

from __future__ import annotations

import wave
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from alebrijes.synth.agent import TTSAgent, _resolve_voice_prompt, _voice_cache_key, create_voice_state, write_wav_array


def test_resolve_voice_prompt_returns_existing_local_file(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"wav")

    resolved = _resolve_voice_prompt(prompt)

    assert resolved == prompt


def test_resolve_voice_prompt_downloads_direct_wav_without_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    downloaded: list[tuple[str, Path]] = []

    def fake_urlretrieve(url: str, destination: Path) -> tuple[str, object]:
        path = Path(destination)
        path.write_bytes(b"wav")
        downloaded.append((url, path))
        return str(path), None

    monkeypatch.setattr("alebrijes.synth.agent.urlretrieve", fake_urlretrieve)

    resolved = _resolve_voice_prompt(
        "https://example.com/voice.wav",
        cache_dir=tmp_path,
    )

    cache_key = _voice_cache_key("https://example.com/voice.wav", None, None)
    assert resolved == tmp_path / f"{cache_key}.wav"
    assert resolved.read_bytes() == b"wav"
    assert downloaded == [("https://example.com/voice.wav", tmp_path / f"{cache_key}.tmp" / "source.wav")]


def test_resolve_voice_prompt_requires_clip_window_for_youtube(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="clip_start and clip_end"):
        _resolve_voice_prompt("https://www.youtube.com/watch?v=abc123", cache_dir=tmp_path)


def test_resolve_voice_prompt_converts_youtube_clip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    subprocess_calls: list[list[str]] = []

    def fake_which(name: str) -> str:
        return f"/usr/bin/{name}"

    def fake_run(command: list[str], check: bool, stdout: int, stderr: int) -> None:
        subprocess_calls.append(command)
        if "yt-dlp" in command[0]:
            output_dir = Path(command[command.index("-o") + 1]).parent
            (output_dir / "source.webm").write_bytes(b"audio")
        else:
            Path(command[-1]).write_bytes(b"wav")

    monkeypatch.setattr("alebrijes.synth.agent.shutil.which", fake_which)
    monkeypatch.setattr("alebrijes.synth.agent.subprocess.run", fake_run)

    resolved = _resolve_voice_prompt(
        "https://youtu.be/example",
        clip_start=5.0,
        clip_end=10.0,
        cache_dir=tmp_path,
    )

    assert resolved.exists()
    assert resolved.read_bytes() == b"wav"
    assert len(subprocess_calls) == 2
    assert subprocess_calls[0][0].endswith("yt-dlp")
    assert subprocess_calls[1][0].endswith("ffmpeg")


def test_create_voice_state_uses_resolved_prompt(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"wav")

    class FakeModel:
        def get_state_for_audio_prompt(self, conditioning: object) -> object:
            return {"conditioning": conditioning}

    state = create_voice_state(prompt, model=FakeModel(), voice_cache_dir=tmp_path)

    assert state == {"conditioning": prompt}


def test_agent_init_uses_provided_voice_state_without_recomputing() -> None:
    import torch

    calls: list[object] = []

    class FakeModel:
        sample_rate = 24000
        # TTSAgent.__init__ primes a real model's stateful submodules via
        # init_states(self._model.flow_lm, ...) whenever a voice_state is
        # supplied directly (see synth/agent.py). init_states only needs
        # flow_lm.named_modules(); a bare nn.Module satisfies that without
        # pulling in the real (heavy) flow_lm.
        flow_lm = torch.nn.Module()

        def get_state_for_audio_prompt(self, conditioning: object) -> object:
            calls.append(conditioning)
            return {"conditioning": conditioning}

        def generate_audio(self, state: object, text: str) -> object:
            return [0.0]

    supplied_state = object()
    agent = TTSAgent(voice="cached", voice_state=supplied_state, model=FakeModel())

    assert agent.voice == "cached"
    assert agent._voice_cache is not None
    assert agent._voice_cache.state is supplied_state
    assert calls == []


def test_agent_stream_chunks_yields_each_chunk_before_concatenating() -> None:
    class FakeModel:
        sample_rate = 24000

        def get_state_for_audio_prompt(self, conditioning: object) -> object:
            return {"conditioning": conditioning}

        def generate_audio_stream(self, state: object, text: str):
            yield np.ones(3, dtype=np.float32)
            yield np.ones(2, dtype=np.float32) * 2

    agent = TTSAgent(voice="alba", model=FakeModel())

    chunks = list(agent.stream_chunks("hello"))

    assert len(chunks) == 2
    assert chunks[0].tolist() == [1.0, 1.0, 1.0]
    assert chunks[1].tolist() == [2.0, 2.0]


def test_synthesize_streaming_concatenates_stream_chunks_and_reports_progress() -> None:
    class FakeModel:
        sample_rate = 2

        def get_state_for_audio_prompt(self, conditioning: object) -> object:
            return {"conditioning": conditioning}

        def generate_audio_stream(self, state: object, text: str):
            yield np.ones(2, dtype=np.float32)
            yield np.ones(2, dtype=np.float32)

    agent = TTSAgent(voice="alba", model=FakeModel())
    progress: list[float] = []

    audio = agent.synthesize_streaming("hello", on_progress=progress.append)

    assert audio.shape[0] == 4
    assert progress == [1.0, 2.0]


def test_write_wav_array_accepts_a_file_like_object() -> None:
    # Regression test: write_wav_array used to unconditionally str(path) its
    # destination, which for a file-like object like BytesIO silently wrote
    # to a garbage on-disk filename derived from the object's repr (e.g.
    # "<_io.BytesIO object at 0x...>") instead of writing into the buffer or
    # raising a clear error.
    buffer = BytesIO()
    write_wav_array(buffer, 24000, np.ones(10, dtype=np.int16))

    with wave.open(BytesIO(buffer.getvalue()), "rb") as handle:
        assert handle.getnframes() == 10
        assert handle.getframerate() == 24000
