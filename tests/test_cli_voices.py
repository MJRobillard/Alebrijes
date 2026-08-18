from __future__ import annotations

from pathlib import Path

import pytest

from alebrijes.cli import main


def test_voices_add_rejects_non_audio_extension(tmp_path: Path) -> None:
    fake_source = tmp_path / "clip.mp4"
    fake_source.write_bytes(b"not really audio")

    with pytest.raises(ValueError, match="audio file"):
        main(["voices", "add", "--source", str(fake_source), "--name", "test", "--voices-dir", str(tmp_path)])


def test_voices_add_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.wav"

    with pytest.raises(FileNotFoundError):
        main(["voices", "add", "--source", str(missing), "--name", "test", "--voices-dir", str(tmp_path)])


def test_voices_add_rejects_url_looking_source(tmp_path: Path) -> None:
    # A URL is never an existing local path, so it's rejected as "missing"
    # before the audio-extension check even runs — either way, no network
    # call is ever attempted for `voices add`.
    with pytest.raises(FileNotFoundError):
        main(
            [
                "voices",
                "add",
                "--source",
                "https://example.com/clip.wav",
                "--name",
                "test",
                "--voices-dir",
                str(tmp_path),
            ]
        )


def test_voices_list_reports_empty_directory(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    exit_code = main(["voices", "list", "--voices-dir", str(tmp_path / "empty")])

    assert exit_code == 0
    assert "No cloned voices found" in capsys.readouterr().out


def test_voices_list_reports_saved_voices(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    (tmp_path / "narrator.pt").write_bytes(b"fake-state")
    (tmp_path / "villain.pt").write_bytes(b"fake-state")

    exit_code = main(["voices", "list", "--voices-dir", str(tmp_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "narrator" in output
    assert "villain" in output
