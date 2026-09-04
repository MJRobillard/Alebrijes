from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from alebrijes.session.fillibuster import list_fillibuster_queue, next_fillibuster


def _write_wav(path: Path, sample_rate: int, value: int, mtime: float) -> None:
    from scipy.io import wavfile

    audio = np.full(10, value, dtype=np.int16)
    wavfile.write(path, sample_rate, audio)
    os.utime(path, (mtime, mtime))


def test_missing_directory_returns_none(tmp_path: Path) -> None:
    assert next_fillibuster(tmp_path / "does_not_exist") is None


def test_empty_directory_returns_none(tmp_path: Path) -> None:
    assert next_fillibuster(tmp_path) is None


def test_pops_oldest_file_first(tmp_path: Path) -> None:
    _write_wav(tmp_path / "a.wav", 16000, value=1, mtime=100)
    _write_wav(tmp_path / "b.wav", 16000, value=2, mtime=200)
    _write_wav(tmp_path / "c.wav", 16000, value=3, mtime=300)

    result = next_fillibuster(tmp_path)

    assert result is not None
    assert result.path.name == "a.wav"
    assert result.sample_rate == 16000
    assert (result.audio == 1).all()


def test_rotates_popped_file_to_the_back(tmp_path: Path) -> None:
    _write_wav(tmp_path / "a.wav", 16000, value=1, mtime=100)
    _write_wav(tmp_path / "b.wav", 16000, value=2, mtime=200)

    first = next_fillibuster(tmp_path)
    second = next_fillibuster(tmp_path)
    third = next_fillibuster(tmp_path)

    assert first is not None and first.path.name == "a.wav"
    assert second is not None and second.path.name == "b.wav"
    # a.wav was rotated to the back after the first pop, so a full cycle
    # returns to it -- confirming this is a reusable pool, not consumed-once.
    assert third is not None and third.path.name == "a.wav"


def test_ignores_non_wav_files(tmp_path: Path) -> None:
    _write_wav(tmp_path / "a.wav", 16000, value=1, mtime=100)
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("not audio")

    result = next_fillibuster(tmp_path)

    assert result is not None
    assert result.path.name == "a.wav"


def test_repeated_pops_never_return_the_same_file_twice_in_a_row(tmp_path: Path) -> None:
    for i, name in enumerate(["a", "b", "c"]):
        _write_wav(tmp_path / f"{name}.wav", 16000, value=i, mtime=100 + i)

    seen = [next_fillibuster(tmp_path).path.name for _ in range(6)]  # type: ignore[union-attr]

    for previous, current in zip(seen, seen[1:]):
        assert previous != current


def test_list_fillibuster_queue_returns_empty_for_missing_or_empty_directory(tmp_path: Path) -> None:
    assert list_fillibuster_queue(tmp_path / "does_not_exist") == []
    assert list_fillibuster_queue(tmp_path) == []


def test_list_fillibuster_queue_matches_pop_order_without_consuming(tmp_path: Path) -> None:
    _write_wav(tmp_path / "a.wav", 16000, value=1, mtime=100)
    _write_wav(tmp_path / "b.wav", 16000, value=2, mtime=200)

    assert list_fillibuster_queue(tmp_path) == ["a.wav", "b.wav"]
    # A peek, not a pop -- calling it doesn't rotate anything.
    assert list_fillibuster_queue(tmp_path) == ["a.wav", "b.wav"]

    popped = next_fillibuster(tmp_path)
    assert popped.path.name == "a.wav"
    assert list_fillibuster_queue(tmp_path) == ["b.wav", "a.wav"]
