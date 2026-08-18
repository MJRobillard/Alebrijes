"""pywebview desktop shell launcher (Phase 3).

Starts the hardened local FastAPI server (`alebrijes.server`) in a background
thread, then opens a native pywebview window pointed at it with the
per-launch token passed as a URL query parameter — the same pattern
Jupyter's own token auth uses. The index page itself is servable without a
token (it's just the app shell, no secret in it); only `/api/*` routes
require it. The token only ever appears in the URL pywebview navigates to
directly, never disclosed anywhere else, so a page an attacker controls has
no way to learn it.

The one thing routed outside the HTTP API entirely: the native file-open
dialogs (`FileApi.pick_file` / `FileApi.pick_audio_file`), exposed via
pywebview's own JS bridge instead of an HTTP endpoint — they're an OS-shell
capability with no network path, not application logic, so they don't
belong behind the token/Origin layer.
"""

from __future__ import annotations

import threading
from typing import Optional

from alebrijes.config import AUDIO_EXTENSIONS
from alebrijes.server import create_app, find_free_port, generate_token

_BOOK_FILE_TYPES = ("Book files (*.epub;*.pdf;*.txt)", "All files (*.*)")
_AUDIO_FILE_TYPES = (
    "Audio files (" + ";".join(f"*{ext}" for ext in sorted(AUDIO_EXTENSIONS)) + ")",
    "All files (*.*)",
)


class FileApi:
    """Exposed to the frontend as `window.pywebview.api`."""

    def pick_file(self) -> Optional[str]:
        return self._pick(_BOOK_FILE_TYPES)

    def pick_audio_file(self) -> Optional[str]:
        return self._pick(_AUDIO_FILE_TYPES)

    def _pick(self, file_types: tuple[str, ...]) -> Optional[str]:
        import webview

        window = webview.windows[0]
        result = window.create_file_dialog(webview.FileDialog.OPEN, file_types=file_types)
        if not result:
            return None
        return result[0]


def build_launch_url(host: str, port: int, token: str) -> str:
    return f"http://{host}:{port}/?token={token}"


def main() -> None:
    import uvicorn
    import webview

    from alebrijes.diagnostics import torch_build_warning

    warning = torch_build_warning()
    if warning:
        print(f"[alebrijes] warning: {warning}")

    host = "127.0.0.1"
    token = generate_token()
    port = find_free_port(host)
    app = create_app(token, allowed_origin=f"http://{host}:{port}")

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    webview.create_window(
        "Alebrijes",
        url=build_launch_url(host, port, token),
        js_api=FileApi(),
        width=1000,
        height=760,
        min_size=(700, 500),
    )
    webview.start()

    server.should_exit = True
    server_thread.join(timeout=3)


if __name__ == "__main__":
    main()
