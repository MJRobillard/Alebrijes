"""Builds the `messages` list sent to Ollama's `/api/chat` for each mode.

One common system-prompt base (book/position context + the narrator
guidelines + position-bounded RAG excerpts) plus a small composable
mode-specific fragment — not four independent prompt templates. See
`narrator_guidelines.md` for the shared spoiler-safety/grounding rules this
always includes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

_GUIDELINES_PATH = Path(__file__).parent / "narrator_guidelines.md"


class Mode(str, Enum):
    QA = "qa"
    CYOA = "cyoa"
    CHECKPOINT = "checkpoint"
    LEARNING = "learning"


_MODE_FRAGMENTS: dict[Mode, str] = {
    Mode.QA: (
        "The listener has interrupted playback to ask a question. Answer it "
        "directly and briefly, then let them return to listening — you are "
        "not continuing the story here, just answering."
    ),
    Mode.CYOA: (
        "The listener is improvising a choice or action. Continue the scene "
        "in the book's voice, incorporating their input, and stay tonally "
        "consistent with the book. If a steer-back beat is provided below, "
        "work it in as a natural in-story turn rather than an abrupt jump."
    ),
    Mode.CHECKPOINT: (
        "You have reached a natural pause point. Ask the listener what they "
        "think should happen next, or what choice the story's protagonist "
        "should make, as an in-story beat."
    ),
    Mode.LEARNING: (
        "Ask the listener a comprehension question about what's happened so "
        "far, or, if they've just answered one, react to their specific "
        "answer in character before continuing."
    ),
}


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str


def load_guidelines() -> str:
    return _GUIDELINES_PATH.read_text(encoding="utf-8")


def build_system_prompt(
    *,
    book_title: str,
    character_voice: str,
    position_summary: str,
    rag_excerpts: Sequence[str],
    mode: Mode,
    guidelines: str | None = None,
) -> str:
    guidelines = guidelines if guidelines is not None else load_guidelines()

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
        f"{_MODE_FRAGMENTS[mode]}"
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
