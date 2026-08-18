"""Cast stage: character -> cloned-voice mapping, persisted as an editable file.

This is the contract between attribution, a future cast-review UI, and synth:
rule-based attribution will be right most of the time and wrong on ambiguous
lines, so its output is never fed straight into synthesis. It's persisted here
first, with low-confidence spans flagged, so a wrong guess is a five-minute
manual edit to a YAML file instead of a silent quality bug baked into the
rendered audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class CastEntry:
    character: str
    voice_state_path: str  # path to a saved .pt voice state


@dataclass
class CastFile:
    book_title: str
    entries: list[CastEntry] = field(default_factory=list)

    def voice_for(self, character: str) -> str:
        for entry in self.entries:
            if entry.character == character:
                return entry.voice_state_path
        raise KeyError(f"No voice cast for character {character!r}")


def load(path: Path) -> CastFile:
    data = yaml.safe_load(path.read_text())
    entries = [CastEntry(**entry) for entry in data.get("entries", [])]
    return CastFile(book_title=data["book_title"], entries=entries)


def save(cast_file: CastFile, path: Path) -> None:
    payload = {
        "book_title": cast_file.book_title,
        "entries": [
            {"character": entry.character, "voice_state_path": entry.voice_state_path}
            for entry in cast_file.entries
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
