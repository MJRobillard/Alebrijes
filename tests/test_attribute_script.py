from __future__ import annotations

import pytest

from alebrijes.attribute.script import attribute_script_chapter
from alebrijes.ingest import Chapter, Paragraph
from alebrijes.segment import SpanKind


def _chapter(lines: list[str]) -> Chapter:
    return Chapter(title="Ch1", index=0, paragraphs=[Paragraph(text=line, index=i) for i, line in enumerate(lines)])


def test_narrator_line_becomes_narration_attributed_to_narrator() -> None:
    spans, attributions = attribute_script_chapter(_chapter(["NARRATOR: The room was quiet."]))

    assert spans[0].kind is SpanKind.NARRATION
    assert spans[0].text == "The room was quiet."
    assert attributions[0].character == "Narrator"
    assert attributions[0].confidence == 1.0


def test_character_line_becomes_dialogue_attributed_to_that_character() -> None:
    spans, attributions = attribute_script_chapter(_chapter(["COLIN: Let's take another path."]))

    assert spans[0].kind is SpanKind.DIALOGUE
    assert spans[0].text == "Let's take another path."
    assert attributions[0].character == "Colin"
    assert attributions[0].confidence == 1.0


def test_span_id_encodes_chapter_and_paragraph_position() -> None:
    spans, _ = attribute_script_chapter(_chapter(["NARRATOR: One.", "COLIN: Two."]))

    assert spans[0].span_id == "0.0.0"
    assert spans[1].span_id == "0.1.0"


def test_spans_and_attributions_are_in_paragraph_order() -> None:
    spans, attributions = attribute_script_chapter(
        _chapter(["NARRATOR: One.", "COLIN: Two.", "SABLE: Three."])
    )

    assert [s.text for s in spans] == ["One.", "Two.", "Three."]
    assert [a.character for a in attributions] == ["Narrator", "Colin", "Sable"]
    assert [a.span is s for a, s in zip(attributions, spans)] == [True, True, True]


def test_raises_on_malformed_paragraph() -> None:
    with pytest.raises(ValueError, match="NAME: text"):
        attribute_script_chapter(_chapter(["not a valid script line"]))
