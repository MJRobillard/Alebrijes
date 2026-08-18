from __future__ import annotations

import sys
import types

from alebrijes.gui import FileApi, build_launch_url


def test_build_launch_url_embeds_token_as_query_param() -> None:
    url = build_launch_url("127.0.0.1", 8123, "sekrit")
    assert url == "http://127.0.0.1:8123/?token=sekrit"


def test_file_api_pick_file_returns_first_selected_path(monkeypatch) -> None:
    fake_window = types.SimpleNamespace(
        create_file_dialog=lambda dialog_type, file_types=(): ("/home/user/book.epub",)
    )
    fake_file_dialog = types.SimpleNamespace(OPEN="OPEN")
    fake_webview_module = types.SimpleNamespace(windows=[fake_window], FileDialog=fake_file_dialog)
    monkeypatch.setitem(sys.modules, "webview", fake_webview_module)

    result = FileApi().pick_file()

    assert result == "/home/user/book.epub"


def test_file_api_pick_file_returns_none_when_dialog_cancelled(monkeypatch) -> None:
    fake_window = types.SimpleNamespace(create_file_dialog=lambda dialog_type, file_types=(): None)
    fake_file_dialog = types.SimpleNamespace(OPEN="OPEN")
    fake_webview_module = types.SimpleNamespace(windows=[fake_window], FileDialog=fake_file_dialog)
    monkeypatch.setitem(sys.modules, "webview", fake_webview_module)

    result = FileApi().pick_file()

    assert result is None
