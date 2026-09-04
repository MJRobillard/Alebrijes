from __future__ import annotations

import logging

from alebrijes.logging_config import configure_logging


def _reset_root_logger() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.WARNING)


def test_configure_logging_defaults_to_info(monkeypatch) -> None:
    monkeypatch.delenv("ALEBRIJES_LOG_LEVEL", raising=False)
    _reset_root_logger()

    configure_logging()

    assert logging.getLogger().level == logging.INFO
    assert logging.getLogger().handlers  # a real handler is attached, not left to lastResort


def test_configure_logging_respects_env_var_override(monkeypatch) -> None:
    monkeypatch.setenv("ALEBRIJES_LOG_LEVEL", "DEBUG")
    _reset_root_logger()

    configure_logging()

    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_ignores_unknown_level_name(monkeypatch) -> None:
    monkeypatch.setenv("ALEBRIJES_LOG_LEVEL", "not-a-real-level")
    _reset_root_logger()

    configure_logging()

    assert logging.getLogger().level == logging.INFO
