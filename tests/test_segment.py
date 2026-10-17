from __future__ import annotations

from alebrijes.ingest import Paragraph
from alebrijes.segment import SpanKind, segment_paragraph


def test_paragraph_with_no_quotes_is_one_narration_span() -> None:
    paragraph = Paragraph(text="He walked into the room and sat down.", index=0)

    spans = segment_paragraph(paragraph, chapter_index=0)

    assert len(spans) == 1
    assert spans[0].kind == SpanKind.NARRATION
    assert spans[0].text == "He walked into the room and sat down."


def test_quote_near_speech_marker_is_dialogue() -> None:
    paragraph = Paragraph(text='Colin looked up and said, "Let\'s take another path."', index=0)

    spans = segment_paragraph(paragraph, chapter_index=0)

    dialogue = [s for s in spans if s.kind == SpanKind.DIALOGUE]
    assert len(dialogue) == 1
    assert dialogue[0].text == "Let's take another path."


def test_short_quoted_word_is_emphasis_not_dialogue() -> None:
    paragraph = Paragraph(text='They were troubled by the "twilight" for years.', index=0)

    spans = segment_paragraph(paragraph, chapter_index=0)

    assert all(s.kind == SpanKind.NARRATION for s in spans)


def test_quote_after_trailing_colon_is_dialogue_even_without_speech_verb() -> None:
    paragraph = Paragraph(text='He turned to face them: "We have nothing left to give."', index=0)

    spans = segment_paragraph(paragraph, chapter_index=0)

    dialogue = [s for s in spans if s.kind == SpanKind.DIALOGUE]
    assert len(dialogue) == 1
    assert dialogue[0].text == "We have nothing left to give."


def test_mixed_paragraph_preserves_order_and_text() -> None:
    paragraph = Paragraph(
        text='Derrick looked ahead and said, "This is the Waning Forest." He fell silent afterward.',
        index=2,
    )

    spans = segment_paragraph(paragraph, chapter_index=1)

    kinds = [s.kind for s in spans]
    assert kinds == [SpanKind.NARRATION, SpanKind.DIALOGUE, SpanKind.NARRATION]
    assert spans[1].text == "This is the Waning Forest."
    assert all(s.chapter_index == 1 and s.paragraph_index == 2 for s in spans)


def test_paragraph_opening_quote_is_dialogue_without_a_speech_tag() -> None:
    paragraph = Paragraph(text='"Then we shall have to go without him."', index=0)

    spans = segment_paragraph(paragraph, chapter_index=0)

    assert [s.kind for s in spans] == [SpanKind.DIALOGUE]


def test_mid_paragraph_quote_without_tag_is_still_narration() -> None:
    paragraph = Paragraph(text='The sign over the door read "Closed for the season" in faded paint.', index=0)

    spans = segment_paragraph(paragraph, chapter_index=0)

    assert all(s.kind == SpanKind.NARRATION for s in spans)
