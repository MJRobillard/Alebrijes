"""Deterministic attribution for the `NAME: text` script ingest format
(`ingest/script.py`) — no NLP guessing needed, since the speaker is already
explicit in the source, unlike `attribute/rules.py`'s heuristics for prose.

Combines segmentation and attribution into one pass (unlike the prose path's
`segment_paragraph` + `RuleBasedAttributor`, which are separate stages):
each script line already is exactly one span with an already-known speaker,
so there's nothing to guess or split — see `cast/build.py` and `render.py`
for where this replaces the prose path when `Book.source_format == "script"`.
"""

from __future__ import annotations

import re

from alebrijes.attribute import Attribution
from alebrijes.ingest import Chapter
from alebrijes.segment import Span, SpanKind

NARRATOR = "Narrator"
_LINE_RE = re.compile(r"^([A-Z][A-Z0-9' ]*):\s*(.*)$")


def attribute_script_chapter(chapter: Chapter) -> tuple[list[Span], list[Attribution]]:
    spans: list[Span] = []
    attributions: list[Attribution] = []

    for paragraph in chapter.paragraphs:
        match = _LINE_RE.match(paragraph.text.strip())
        if not match:
            raise ValueError(f"Script line doesn't match 'NAME: text': {paragraph.text!r}")

        raw_name, text = match.group(1).strip(), match.group(2).strip()
        is_narrator = raw_name.upper() == "NARRATOR"
        kind = SpanKind.NARRATION if is_narrator else SpanKind.DIALOGUE
        character = NARRATOR if is_narrator else raw_name.title()

        span = Span(
            text=text,
            kind=kind,
            chapter_index=chapter.index,
            paragraph_index=paragraph.index,
            span_id=f"{chapter.index}.{paragraph.index}.0",
        )
        spans.append(span)
        attributions.append(Attribution(span=span, character=character, confidence=1.0))

    return spans, attributions
