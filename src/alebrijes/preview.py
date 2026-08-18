"""Short-clip synthesis for the GUI's inline low-confidence-line playback.

Not one of the six pipeline stages — a small, focused module so the caching
it needs is scoped to what previews actually require: previews are called
repeatedly against one long-lived server process (unlike `render_book`,
which loads a model once per book render), so the model and each voice's
resolved state are cached at module scope across calls, not re-loaded per
request. Reloading the model on every preview click would make each one
take as long as the first.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from alebrijes.synth.agent import TTSAgent, write_wav_array

_model = None
_voice_state_cache: dict[str, object] = {}


def _load_voice_state(path: Path) -> object:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _get_model() -> object:
    global _model
    if _model is None:
        from pocket_tts import TTSModel

        try:
            _model = TTSModel.load_model()
        except AttributeError:
            _model = TTSModel()
    return _model


def synthesize_preview(text: str, voice_state_path: str, model: Optional[object] = None) -> bytes:
    """Synthesizes `text` in the given voice and returns it as WAV bytes."""
    resolved_model = model if model is not None else _get_model()

    if voice_state_path not in _voice_state_cache:
        path = Path(voice_state_path)
        if path.exists():
            _voice_state_cache[voice_state_path] = _load_voice_state(path)
        else:
            _voice_state_cache[voice_state_path] = resolved_model.get_state_for_audio_prompt(voice_state_path)
    state = _voice_state_cache[voice_state_path]

    agent = TTSAgent(voice=voice_state_path, model=resolved_model, voice_state=state)
    audio = agent.synthesize(text)

    buffer = io.BytesIO()
    write_wav_array(buffer, agent.sample_rate, audio)
    return buffer.getvalue()
