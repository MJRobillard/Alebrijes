from __future__ import annotations

from alebrijes.attribute.rules import UNKNOWN, RuleBasedAttributor
from alebrijes.segment import Span, SpanKind


def _narration(text: str, paragraph: int = 0) -> Span:
    return Span(text=text, kind=SpanKind.NARRATION, chapter_index=0, paragraph_index=paragraph)


def _dialogue(text: str, paragraph: int = 0) -> Span:
    return Span(text=text, kind=SpanKind.DIALOGUE, chapter_index=0, paragraph_index=paragraph)


def test_named_subject_before_quote_is_high_confidence() -> None:
    spans = [_narration("Colin said,"), _dialogue("Let's take another path.")]

    attributions = RuleBasedAttributor().attribute(spans)

    [attribution] = attributions
    assert attribution.character == "Colin"
    assert attribution.confidence >= 0.8


def test_named_subject_after_quote_inverted_order_is_high_confidence() -> None:
    spans = [_dialogue("Let's take another path."), _narration("said Colin.")]

    attributions = RuleBasedAttributor().attribute(spans)

    [attribution] = attributions
    assert attribution.character == "Colin"
    assert attribution.confidence >= 0.8


def test_pronoun_subject_resolves_to_most_recently_mentioned_character() -> None:
    spans = [_narration("Audrey walked in. She said,"), _dialogue("Hello.")]

    attributions = RuleBasedAttributor().attribute(spans)

    [attribution] = attributions
    assert attribution.character == "Audrey"
    assert 0.4 <= attribution.confidence < 0.8


def test_untagged_back_and_forth_alternates_between_established_speakers() -> None:
    spans = [
        _narration("Colin said,", paragraph=0),
        _dialogue("I will go first.", paragraph=0),
        _narration("Audrey replied,", paragraph=1),
        _dialogue("Then I will follow.", paragraph=1),
        _dialogue("Let's move quickly.", paragraph=2),
        _dialogue("Agreed.", paragraph=3),
    ]

    attributions = RuleBasedAttributor().attribute(spans)
    characters = [a.character for a in attributions]
    confidences = [a.confidence for a in attributions]

    assert characters == ["Colin", "Audrey", "Colin", "Audrey"]
    assert confidences[0] >= 0.8 and confidences[1] >= 0.8
    assert confidences[2] < 0.5 and confidences[3] < 0.5


def test_unresolvable_dialogue_is_unknown_with_zero_confidence() -> None:
    spans = [_narration("The room was silent."), _dialogue("Where is everyone?")]

    attributions = RuleBasedAttributor().attribute(spans)

    [attribution] = attributions
    assert attribution.character == UNKNOWN
    assert attribution.confidence == 0.0


def test_narration_spans_are_never_attributed() -> None:
    spans = [
        _narration("Colin said,"),
        _dialogue("Let's take another path."),
        _narration("He turned and walked away."),
    ]

    attributions = RuleBasedAttributor().attribute(spans)

    assert len(attributions) == 1
    assert attributions[0].span.kind == SpanKind.DIALOGUE
