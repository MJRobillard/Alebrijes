from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable, Optional, Union
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np

VoicePrompt = Union[str, Path]

_DEFAULT_VOICE_CACHE_DIR = Path(tempfile.gettempdir()) / "pocket_tts_agent" / "voice_prompts"


def write_wav_array(path: Union[str, Path, object], sample_rate: int, audio: np.ndarray) -> None:
    """Writes a raw audio array (float or int16) to a 16-bit PCM WAV file.

    `path` accepts a str/Path destination or an open binary file-like object
    (e.g. `io.BytesIO`, for in-memory previews that never touch disk) — only
    str/Path go through `str(path)`; a file-like object is passed to scipy
    as-is, since stringifying it would silently write to a garbage filename
    derived from its repr instead of raising a clear error.

    Public so callers assembling audio from multiple synthesize() calls (e.g.
    a multi-voice chapter render) can write it out without reaching into
    TTSAgent internals — see synth.agent.TTSAgent.write_wav for the
    single-call text-to-file convenience path.
    """
    from scipy.io import wavfile as _wavfile

    pcm_audio = np.asarray(audio)
    if pcm_audio.dtype.kind == "f":
        pcm_audio = np.nan_to_num(pcm_audio, nan=0.0, posinf=1.0, neginf=-1.0)
        pcm_audio = np.clip(pcm_audio, -1.0, 1.0)
        pcm_audio = (pcm_audio * 32767.0).astype(np.int16)
    elif pcm_audio.dtype != np.int16:
        pcm_audio = pcm_audio.astype(np.int16)

    destination = str(path) if isinstance(path, (str, Path)) else path
    _wavfile.write(destination, sample_rate, pcm_audio)


_write_wav = write_wav_array


@dataclass
class _VoiceCache:
    voice: VoicePrompt
    state: object


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_youtube_url(value: str) -> bool:
    host = urlparse(value).netloc.lower()
    return host.endswith("youtube.com") or host.endswith("youtu.be")


def _voice_cache_key(voice: str, clip_start: Optional[float], clip_end: Optional[float]) -> str:
    token = f"{voice}\n{clip_start!r}\n{clip_end!r}"
    return sha256(token.encode("utf-8")).hexdigest()[:16]


def _require_tool(name: str) -> str:
    executable = shutil.which(name)
    if executable:
        return executable
    raise RuntimeError(f"{name} is required for remote voice prompts but was not found on PATH.")


def _yt_dlp_command() -> list[str]:
    executable = shutil.which("yt-dlp")
    if executable:
        return [executable]

    try:
        import yt_dlp  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "yt-dlp is required for YouTube voice prompts. Install it with `python -m pip install yt-dlp` "
            "or add the `yt-dlp` executable to PATH."
        ) from exc

    return ["python", "-m", "yt_dlp"]


def _validate_clip_window(clip_start: Optional[float], clip_end: Optional[float]) -> None:
    if clip_start is None or clip_end is None:
        raise ValueError("Remote clip prompts require both clip_start and clip_end values.")
    if clip_start < 0:
        raise ValueError("clip_start must be greater than or equal to 0.")
    if clip_end <= clip_start:
        raise ValueError("clip_end must be greater than clip_start.")


def _download_direct_audio(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, destination)
    return destination


def _download_youtube_audio(url: str, destination: Path) -> Path:
    yt_dlp_command = _yt_dlp_command()
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_template = destination.parent / "source.%(ext)s"
    subprocess.run(
        [
            *yt_dlp_command,
            "--no-playlist",
            "--no-progress",
            "--quiet",
            "-f",
            "bestaudio/best",
            "-o",
            str(output_template),
            url,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    matches = sorted(destination.parent.glob("source.*"))
    if not matches:
        raise RuntimeError(f"yt-dlp did not produce a local audio file for {url}")
    return matches[0]


def _convert_audio_to_wav(source: Path, destination: Path, clip_start: Optional[float], clip_end: Optional[float]) -> Path:
    ffmpeg = _require_tool("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-y"]
    if clip_start is not None:
        command.extend(["-ss", f"{clip_start:.3f}"])
    command.extend(["-i", str(source)])
    if clip_end is not None and clip_start is not None:
        command.extend(["-t", f"{clip_end - clip_start:.3f}"])
    command.extend(["-ac", "1", "-ar", "24000", str(destination)])
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return destination


def _resolve_voice_prompt(
    voice: VoicePrompt,
    clip_start: Optional[float] = None,
    clip_end: Optional[float] = None,
    cache_dir: Path = _DEFAULT_VOICE_CACHE_DIR,
) -> VoicePrompt:
    voice_str = str(voice)
    voice_path = Path(voice_str)
    if voice_path.exists():
        return voice_path
    if not _is_url(voice_str):
        return voice_str

    cache_key = _voice_cache_key(voice_str, clip_start, clip_end)
    cached_wav = cache_dir / f"{cache_key}.wav"
    if cached_wav.exists():
        return cached_wav

    temp_dir = cache_dir / f"{cache_key}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        if _is_youtube_url(voice_str):
            _validate_clip_window(clip_start, clip_end)
            source_path = _download_youtube_audio(voice_str, temp_dir / "source")
            return _convert_audio_to_wav(source_path, cached_wav, clip_start, clip_end)

        parsed = urlparse(voice_str)
        suffix = Path(parsed.path).suffix.lower() or ".bin"
        source_path = _download_direct_audio(voice_str, temp_dir / f"source{suffix}")
        if suffix == ".wav" and clip_start is None and clip_end is None:
            cached_wav.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(cached_wav))
            return cached_wav
        return _convert_audio_to_wav(source_path, cached_wav, clip_start, clip_end)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def create_voice_state(
    voice: VoicePrompt,
    clip_start: Optional[float] = None,
    clip_end: Optional[float] = None,
    model: Optional[object] = None,
    voice_cache_dir: Union[str, Path] = _DEFAULT_VOICE_CACHE_DIR,
) -> object:
    if model is None:
        from pocket_tts import TTSModel

        try:
            model = TTSModel.load_model()
        except AttributeError:
            model = TTSModel()

    conditioning = _resolve_voice_prompt(
        voice,
        clip_start=clip_start,
        clip_end=clip_end,
        cache_dir=Path(voice_cache_dir),
    )
    return model.get_state_for_audio_prompt(conditioning)


class TTSAgent:
    """Thin wrapper that caches model and voice state for repeated synthesis."""

    def __init__(
        self,
        voice: VoicePrompt = "alba",
        model: Optional[object] = None,
        voice_state: Optional[object] = None,
        voice_start: Optional[float] = None,
        voice_end: Optional[float] = None,
        voice_cache_dir: Union[str, Path] = _DEFAULT_VOICE_CACHE_DIR,
    ) -> None:
        if model is None:
            from pocket_tts import TTSModel

            try:
                model = TTSModel.load_model()
            except AttributeError:
                
                model = TTSModel()
        self._model = model
        self._voice_start = voice_start
        self._voice_end = voice_end
        self._voice_cache_dir = Path(voice_cache_dir)
        self._voice_cache: Optional[_VoiceCache] = None
        if voice_state is not None:
            # A voice state loaded from disk (or passed in directly) was built
            # against a different model instance/process. The model's stateful
            # submodules need their `_module_absolute_name` primed via
            # init_states() before generation can look up state by name, even
            # though we don't want its returned (empty) state itself.
            from pocket_tts.modules.stateful_module import init_states

            init_states(self._model.flow_lm, batch_size=1, sequence_length=1000)
            self._voice_cache = _VoiceCache(voice=voice, state=voice_state)
        else:
            self.set_voice(voice, clip_start=voice_start, clip_end=voice_end)

    @property
    def sample_rate(self) -> int:
        return int(self._model.sample_rate)

    @property
    def voice(self) -> VoicePrompt:
        """Return the currently configured voice prompt."""
        if not self._voice_cache:
            raise RuntimeError("Voice state is not initialized")
        return self._voice_cache.voice

    def set_voice(
        self,
        voice: VoicePrompt,
        clip_start: Optional[float] = None,
        clip_end: Optional[float] = None,
    ) -> None:
        """Load and cache the voice state for future calls."""
        resolved_start = self._voice_start if clip_start is None else clip_start
        resolved_end = self._voice_end if clip_end is None else clip_end
        conditioning = _resolve_voice_prompt(
            voice,
            clip_start=resolved_start,
            clip_end=resolved_end,
            cache_dir=self._voice_cache_dir,
        )

        state = self._model.get_state_for_audio_prompt(conditioning)
        self._voice_cache = _VoiceCache(voice=voice, state=state)

    def use_voice_state(self, voice: VoicePrompt, state: object) -> None:
        """Switch to an already-computed voice state without recomputing it.

        For rendering a multi-voice cast: one agent's cached state is swapped
        before each line instead of resolving/recomputing per line, or
        keeping one TTSAgent per character alive at once.
        """
        self._voice_cache = _VoiceCache(voice=voice, state=state)

    def synthesize(self, text: str) -> np.ndarray:
        """Generate audio for the provided text using the cached voice."""
        if not self._voice_cache:
            raise RuntimeError("Voice state is not initialized")
        audio = self._model.generate_audio(self._voice_cache.state, text)
        if isinstance(audio, np.ndarray):
            return audio
        return audio.numpy()

    def stream_chunks(self, text: str) -> Iterator[np.ndarray]:
        """Yields each raw audio chunk as pocket_tts decodes it, instead of
        buffering the whole utterance before returning anything.

        pocket_tts generates text sentence-by-sentence internally and yields
        each chunk of decoded audio as soon as it's ready
        (`generate_audio_stream`); `generate_audio`/`synthesize` just buffer
        all of those chunks before returning. A caller that can play/send
        audio incrementally (e.g. server.py's live-session sentence
        synthesis) gets a much lower time-to-first-sound this way than
        waiting for a whole sentence's audio at once -- `synthesize_streaming`
        below is for a caller that only wants progress reporting but still
        needs the concatenated result.
        """
        if not self._voice_cache:
            raise RuntimeError("Voice state is not initialized")
        for chunk in self._model.generate_audio_stream(self._voice_cache.state, text):
            yield chunk.numpy() if not isinstance(chunk, np.ndarray) else chunk

    def synthesize_streaming(
        self, text: str, on_progress: Optional[Callable[[float], None]] = None
    ) -> np.ndarray:
        """Like `synthesize`, but calls `on_progress(audio_seconds_generated_so_far)`
        after every audio chunk pocket_tts decodes.

        Consuming `stream_chunks` ourselves and concatenating gives callers a
        real, continuously-updating measure of progress through a single long
        job instead of only an all-or-nothing "done" signal.
        """
        chunks: list[np.ndarray] = []
        total_samples = 0
        for chunk_array in self.stream_chunks(text):
            chunks.append(chunk_array)
            total_samples += chunk_array.shape[-1]
            if on_progress is not None:
                on_progress(total_samples / self.sample_rate)
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def write_wav(
        self,
        path: Union[str, Path],
        text: str,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """Generate audio and write it to a WAV file.

        Pass `on_progress` to receive live audio-seconds-generated updates
        while a (possibly long) job is running; see `synthesize_streaming`.
        """
        audio = self.synthesize_streaming(text, on_progress) if on_progress else self.synthesize(text)
        _write_wav(path, self.sample_rate, audio)
        return Path(path)
