from __future__ import annotations

import wave
from io import BytesIO
from pathlib import Path

import numpy as np

import alebrijes.preview as preview


class _FakeModel:
    sample_rate = 24000

    def __init__(self) -> None:
        import torch

        # TTSAgent.__init__ primes a real model's stateful submodules via
        # init_states(self._model.flow_lm, ...) when a voice_state is
        # supplied directly; init_states only needs flow_lm.named_modules(),
        # so a bare nn.Module satisfies that (see test_synth_agent.py).
        self.flow_lm = torch.nn.Module()
        self.resolved_prompts: list[object] = []

    def get_state_for_audio_prompt(self, conditioning: object) -> object:
        self.resolved_prompts.append(conditioning)
        return {"conditioning": conditioning}

    def generate_audio(self, state: object, text: str) -> np.ndarray:
        return np.ones(50, dtype=np.float32) * 0.5


def _reset_cache() -> None:
    preview._voice_state_cache.clear()


def test_synthesize_preview_returns_valid_wav_bytes() -> None:
    _reset_cache()
    audio_bytes = preview.synthesize_preview("Hello there.", "alba", model=_FakeModel())

    with wave.open(BytesIO(audio_bytes), "rb") as handle:
        assert handle.getnframes() == 50
        assert handle.getframerate() == 24000


def test_synthesize_preview_resolves_builtin_voice_by_name() -> None:
    _reset_cache()
    fake_model = _FakeModel()
    preview.synthesize_preview("Hello there.", "javert", model=fake_model)

    assert fake_model.resolved_prompts == ["javert"]


def test_synthesize_preview_loads_pt_file_for_existing_path(tmp_path: Path, monkeypatch) -> None:
    _reset_cache()
    voice_path = tmp_path / "colin.pt"
    voice_path.write_bytes(b"fake-state")
    monkeypatch.setattr(preview, "_load_voice_state", lambda path: {"loaded_from": str(path)})

    fake_model = _FakeModel()
    preview.synthesize_preview("Hello there.", str(voice_path), model=fake_model)

    # Builtin resolution (get_state_for_audio_prompt) must NOT have been used
    # for a real file path.
    assert fake_model.resolved_prompts == []


def test_synthesize_preview_caches_voice_state_across_calls() -> None:
    _reset_cache()
    fake_model = _FakeModel()
    preview.synthesize_preview("First line.", "alba", model=fake_model)
    preview.synthesize_preview("Second line.", "alba", model=fake_model)

    assert fake_model.resolved_prompts == ["alba"]
