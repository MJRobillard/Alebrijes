from __future__ import annotations

from alebrijes.attribute.rules import UNKNOWN, RuleBasedAttributor
from alebrijes.segment import Span, SpanKind


def _narration(text: str, paragraph: int = 0) -> Span:
    return Span(
        text=text,
        kind=SpanKind.NARRATION,
        chapter_index=0,
        paragraph_index=paragraph,
        span_id=f"0.{paragraph}.0",
    )


def _dialogue(text: str, paragraph: int = 0) -> Span:
    return Span(
        text=text,
        kind=SpanKind.DIALOGUE,
        chapter_index=0,
        paragraph_index=paragraph,
        span_id=f"0.{paragraph}.0",
    )


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


def test_labeled_line_with_no_speech_verb_attributes_to_named_label() -> None:
    # No quotes and no "said"/"asked" verb for segment.py or the dependency
    # parse to key off of -- just an ALL-CAPS "NAME:" label, e.g. a pasted
    # dialogue exchange. Should not fall through to Unknown.
    spans = [
        _narration("FELICITY: Except Bayesian Optimization actually converges.", paragraph=0),
        _narration("SARA: So if I were a hyperparameter, which one would I be?", paragraph=1),
        _narration("FELICITY: You would be the spin speed.", paragraph=2),
    ]

    attributions = RuleBasedAttributor().attribute(spans)
    characters = [a.character for a in attributions]

    assert characters == ["Felicity", "Sara", "Felicity"]
    assert all(a.confidence >= 0.8 for a in attributions)


def test_narration_spans_are_never_attributed() -> None:
    spans = [
        _narration("Colin said,"),
        _dialogue("Let's take another path."),
        _narration("He turned and walked away."),
    ]

    attributions = RuleBasedAttributor().attribute(spans)

    assert len(attributions) == 1
    assert attributions[0].span.kind == SpanKind.DIALOGUE


def test_titled_name_keeps_its_title() -> None:
    # A bare "Bennet" would merge Mr. and Mrs. Bennet into one cast entry.
    spans = [_dialogue("Have you heard the news?"), _narration("said Mrs. Bennet.")]

    [attribution] = RuleBasedAttributor().attribute(spans)

    assert attribution.character == "Mrs. Bennet"
    assert attribution.confidence >= 0.8


def test_common_noun_subject_is_not_cast_as_a_character() -> None:
    spans = [_narration("Audrey walked in."), _dialogue("Where have you been?"), _narration("cried his wife.")]

    [attribution] = RuleBasedAttributor().attribute(spans)

    assert attribution.character != "wife"
