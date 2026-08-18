from __future__ import annotations

from pathlib import Path

import pytest

from alebrijes.hf_bundle import configure_bundled_hf_cache


def test_noop_when_not_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr("sys.frozen", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    configure_bundled_hf_cache()

    import os

    assert "HF_HOME" not in os.environ
    assert "HF_HUB_OFFLINE" not in os.environ


def test_points_hf_home_at_bundled_cache_when_frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundled_cache = tmp_path / "hf_cache"
    bundled_cache.mkdir()
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys._MEIPASS", str(tmp_path), raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    configure_bundled_hf_cache()

    import os

    assert os.environ["HF_HOME"] == str(bundled_cache)
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_noop_when_frozen_but_bundle_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys._MEIPASS", str(tmp_path), raising=False)  # no hf_cache/ written here
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    configure_bundled_hf_cache()

    import os

    assert "HF_HOME" not in os.environ
