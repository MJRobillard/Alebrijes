from __future__ import annotations

import wave
from pathlib import Path

import pytest

from alebrijes.assemble import merge_wav_files


def _write_wav(path: Path, frames: bytes, *, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)


def test_merge_wav_files_concatenates_in_order(tmp_path: Path) -> None:
    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    _write_wav(first, b"\x01\x00" * 4)
    _write_wav(second, b"\x02\x00" * 4)

    output = tmp_path / "merged.wav"
    merge_wav_files([first, second], output)

    with wave.open(str(output), "rb") as handle:
        assert handle.readframes(handle.getnframes()) == (b"\x01\x00" * 4) + (b"\x02\x00" * 4)


def test_merge_wav_files_rejects_format_mismatch(tmp_path: Path) -> None:
    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    _write_wav(first, b"\x01\x00" * 4, sample_rate=24000)
    _write_wav(second, b"\x02\x00" * 4, sample_rate=16000)

    with pytest.raises(ValueError, match="format mismatch"):
        merge_wav_files([first, second], tmp_path / "merged.wav")


def test_merge_wav_files_requires_input() -> None:
    with pytest.raises(ValueError, match="No cached audio"):
        merge_wav_files([], Path("unused.wav"))
