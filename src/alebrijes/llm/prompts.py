"""Builds the `messages` list sent to Ollama's `/api/chat` for each mode.

One common system-prompt base (book/position context + the narrator
guidelines + position-bounded RAG excerpts) plus a small composable
mode-specific fragment — not four independent prompt templates. Both live
in `narrator_guidelines.md`: a YAML frontmatter block holds the four
short per-mode fragments (structured, since there's one per `Mode`), and
the Markdown body below it holds the long-form shared guidelines (plain
prose reads better as Markdown than as a YAML block scalar). Editing
either doesn't need a code change or restart — the file's read fresh each
time a prompt is built.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

import yaml

_GUIDELINES_PATH = Path(__file__).parent / "narrator_guidelines.md"
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)


class Mode(str, Enum):
    QA = "qa"
    CYOA = "cyoa"
    CHECKPOINT = "checkpoint"
    LEARNING = "learning"


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str


def _read_guidelines_file() -> tuple[str, dict]:
    """Splits `narrator_guidelines.md` into (markdown body, frontmatter
    dict). A missing/malformed frontmatter block is treated as "no
    frontmatter" (empty dict) rather than an error, so the file still works
    as pure prose if someone strips it."""
    raw = _GUIDELINES_PATH.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return raw, {}
    frontmatter = yaml.safe_load(match.group(1)) or {}
    return raw[match.end() :], frontmatter


def load_guidelines() -> str:
    body, _ = _read_guidelines_file()
    return body.strip()


def load_mode_fragments() -> dict[Mode, str]:
    _, frontmatter = _read_guidelines_file()
    modes = frontmatter.get("modes", {})
    return {Mode(key): value.strip() for key, value in modes.items()}


def build_system_prompt(
    *,
    book_title: str,
    character_voice: str,
    position_summary: str,
    rag_excerpts: Sequence[str],
    mode: Mode,
    guidelines: str | None = None,
    mode_fragments: dict[Mode, str] | None = None,
) -> str:
    guidelines = guidelines if guidelines is not None else load_guidelines()
    mode_fragments = mode_fragments if mode_fragments is not None else load_mode_fragments()

    excerpts_block = (
        "\n".join(f"- {excerpt}" for excerpt in rag_excerpts)
        if rag_excerpts
        else "(no relevant excerpts retrieved)"
    )

    return (
        f"{guidelines}\n\n"
        "---\n\n"
        f"Book: {book_title}\n"
        f"You are speaking as: {character_voice}\n"
        f"Listener's current position: {position_summary}\n\n"
        f"Retrieved excerpts (bounded to the listener's current position):\n{excerpts_block}\n\n"
        f"{mode_fragments[mode]}"
    )


def build_messages(
    *,
    mode: Mode,
    book_title: str,
    character_voice: str,
    position_summary: str,
    rag_excerpts: Sequence[str],
    history: Sequence[ChatTurn],
    user_message: str,
    guidelines: str | None = None,
) -> list[dict]:
    system_prompt = build_system_prompt(
        book_title=book_title,
        character_voice=character_voice,
        position_summary=position_summary,
        rag_excerpts=rag_excerpts,
        mode=mode,
        guidelines=guidelines,
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend({"role": turn.role, "content": turn.content} for turn in history)
    messages.append({"role": "user", "content": user_message})
    return messages
