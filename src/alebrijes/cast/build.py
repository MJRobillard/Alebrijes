"""Builds a CastFile from a parsed Book by running segment + attribute over it.

Kept separate from the CLI (so it's testable without spawning a subprocess)
and from the CastEntry/CastFile data model in `alebrijes.cast` (so that stays
a plain, dependency-light module with no attribution logic in it).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from alebrijes.attribute import Attribution
from alebrijes.attribute.rules import UNKNOWN, RuleBasedAttributor
from alebrijes.cast import CastEntry, CastFile
from alebrijes.config import BUILTIN_VOICES
from alebrijes.ingest import Book
from alebrijes.segment import segment_paragraph

NARRATOR = "Narrator"
LOW_CONFIDENCE_THRESHOLD = 0.5


def _voice_for(character: str) -> str:
    # Placeholder voice assignment: a deterministic pick from pocket-tts's
    # built-in voices, ported from pocket-tts-agent's examples/multi_ci_tts.py
    # (_assign_voice_for_speaker). Meant to be overridden with real cloned
    # voices before rendering — this just gives every detected character
    # *something* distinct out of the box.
    if character == NARRATOR:
        return BUILTIN_VOICES[0]
    digest = hashlib.md5(character.lower().encode("utf-8")).hexdigest()
    return BUILTIN_VOICES[int(digest, 16) % len(BUILTIN_VOICES)]


@dataclass
class CastResult:
    cast_file: CastFile
    attributions: list[Attribution]

    def low_confidence(self) -> list[Attribution]:
        return [a for a in self.attributions if a.confidence < LOW_CONFIDENCE_THRESHOLD]


def build_cast_file(book: Book) -> CastResult:
    attributor = RuleBasedAttributor()
    all_attributions: list[Attribution] = []
    characters: list[str] = []

    for chapter in book.chapters:
        spans = [span for paragraph in chapter.paragraphs for span in segment_paragraph(paragraph, chapter.index)]
        for attribution in attributor.attribute(spans):
            all_attributions.append(attribution)
            if attribution.character != UNKNOWN and attribution.character not in characters:
                characters.append(attribution.character)

    entries = [CastEntry(character=NARRATOR, voice_state_path=_voice_for(NARRATOR))]
    entries += [CastEntry(character=name, voice_state_path=_voice_for(name)) for name in characters]

    return CastResult(cast_file=CastFile(book_title=book.title, entries=entries), attributions=all_attributions)
