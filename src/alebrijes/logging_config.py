"""Makes `logging.getLogger(__name__).info/warning(...)` calls anywhere in
this package actually show up in the terminal the app was launched from.

Without this, the standard library's root logger has no handler at all --
Python's own fallback (`logging.lastResort`) only prints WARNING and above,
silently swallowing every INFO/DEBUG call (e.g. `llm.ollama.keep_warm`'s
"reached Ollama again" recovery message never appeared). uvicorn's own
`log_level=` setting (see `gui.py`, `server.run_server_blocking`) only
configures uvicorn's *own* loggers via its internal dictConfig -- it has no
effect on this package's loggers, which is the actual gap this closes.
"""

from __future__ import annotations

import logging
import os

_DEFAULT_LEVEL = "INFO"


def configure_logging() -> None:
    """Call once, early, from each real entrypoint (gui.py's `main`,
    `server.run_server_blocking`) -- not from library code, which should
    never call `basicConfig` itself (that's an application concern, and
    calling it twice is a silent no-op the second time, masking a level
    override). `ALEBRIJES_LOG_LEVEL` (e.g. `DEBUG`) overrides the default
    without a code change, same env-var-overridable pattern as
    `llm.ollama.DEFAULT_CHAT_MODEL`.
    """
    level_name = os.environ.get("ALEBRIJES_LOG_LEVEL", _DEFAULT_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
