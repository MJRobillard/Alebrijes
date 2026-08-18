from __future__ import annotations

import sys
import types

import pytest

from alebrijes.ffmpeg_util import find_ffmpeg


def test_find_ffmpeg_returns_the_imageio_ffmpeg_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.SimpleNamespace(get_ffmpeg_exe=lambda: "/fake/path/to/ffmpeg")
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_module)

    assert find_ffmpeg() == "/fake/path/to/ffmpeg"


def test_find_ffmpeg_raises_clear_error_when_imageio_ffmpeg_cannot_locate_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise():
        raise RuntimeError("no ffmpeg exe could be found")

    fake_module = types.SimpleNamespace(get_ffmpeg_exe=_raise)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_module)

    with pytest.raises(RuntimeError, match="could not be located"):
        find_ffmpeg()


def test_find_ffmpeg_raises_clear_error_when_package_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)  # simulates ImportError on import

    with pytest.raises(RuntimeError, match="not installed"):
        find_ffmpeg()
