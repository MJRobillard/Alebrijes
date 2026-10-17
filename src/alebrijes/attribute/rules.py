"""Rule-based Attributor: spaCy dependency parse + novel-specific fallbacks.

Borrows the core proven technique from sayswho (github.com/afriedman412/sayswho,
MIT) — using spaCy's dependency parse to find a speech verb's syntactic
subject, which correctly handles both word orders ("Colin said" / "said
Colin") that a naive proximity/regex heuristic gets wrong. sayswho itself is
tuned for short news articles and leans on spaCy's experimental
(version-pinned) coreference component for pronoun resolution; neither fits a
novel's longer, denser dialogue exchanges or this project's "no torch, no
heavy model" goal for this stage, so pronoun subjects and untagged
back-and-forth exchanges are resolved here with two novel-specific heuristics
instead:

- a pronoun subject ("she said") resolves to whichever named character was
  most recently mentioned in narration;
- an untagged dialogue span (no speech verb in the neighboring narration at
  all — a common back-and-forth exchange) alternates between the two most
  recently established speakers.

Anything neither can resolve becomes "Unknown" at confidence 0.0, which the
(planned) cast-review UI exists to catch — this attributor is meant to be
right most of the time and to fail loudly, not silently, the rest.
"""

from __future__ import annotations

import functools
import re

from alebrijes.attribute import Attribution
from alebrijes.segment import Span, SpanKind

# A bare "NAME: text" label at the start of a narration span -- the same
# ALL-CAPS convention as ingest/script.py's strict format, but appearing
# inline in prose (no surrounding quotes for segment.py's quote-based split
# to key off of), e.g. a pasted dialogue exchange with no narration verbs at
# all. Matched here so those turns aren't silently left as unattributed
# narration.
_LABELED_LINE_RE = re.compile(r"^([A-Z][A-Z0-9' ]*):\s*(.+)$")

_SPEECH_LEMMAS = {
    "say",
    "ask",
    "reply",
    "blurt",
    "shout",
    "yell",
    "tell",
    "write",
    "chant",
    "whisper",
    "mutter",
    "cry",
    "exclaim",
    "murmur",
    "call",
    "answer",
    "interrupt",
    "continue",
    "add",
}

_PRONOUN_SUBJECTS = {"he", "she", "they", "i", "we"}

_TITLES = {"mr", "mrs", "miss", "ms", "dr", "sir", "lady", "lord", "colonel", "captain", "aunt", "uncle"}

UNKNOWN = "Unknown"

_NAMED_CONFIDENCE = 0.9
_PRONOUN_CONFIDENCE = 0.55
_ALTERNATION_CONFIDENCE = 0.35


@functools.lru_cache(maxsize=1)
def _nlp():
    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "Rule-based attribution needs spaCy's English model. Install it with "
            "`python -m spacy download en_core_web_sm`."
        ) from exc


def _full_name(token) -> str:
    """Expand a name's head token to its compound modifiers, so "Mrs. Bennet
    said" yields "Mrs. Bennet", not "Bennet" -- which would merge Mr. and Mrs.
    Bennet into one cast entry (found by evals/attribution/run_pdnc.py)."""
    parts = [t for t in token.lefts if t.dep_ in ("compound", "flat")] + [token]
    return " ".join(t.text for t in parts)


def _speech_verb_subject(text: str) -> str | None:
    """Return who a speech verb's syntactic subject names, if any.

    Handles both "Colin said" and "said Colin" — the dependency parse finds
    the verb's nsubj regardless of word order, unlike proximity/regex.
    Returns a full proper name, or a bare pronoun from `_PRONOUN_SUBJECTS`
    for the caller to resolve. A common-noun subject ("cried his wife",
    "her mother said") names nobody this attributor can resolve, so it's
    skipped rather than cast as a character called "wife".
    """
    doc = _nlp()(text)
    for token in doc:
        if token.pos_ != "VERB" or token.lemma_ not in _SPEECH_LEMMAS:
            continue
        for child in token.children:
            if child.dep_ not in ("nsubj", "nsubjpass"):
                continue
            if child.text.lower() in _PRONOUN_SUBJECTS:
                return child.text
            if child.pos_ == "PROPN":
                return _full_name(child)
    return None


def _person_entities(text: str) -> list[str]:
    """PERSON entities, with a directly preceding title kept ("Mrs. Bennet",
    not spaCy's bare "Bennet") for the same reason as `_full_name`."""
    doc = _nlp()(text)
    people = []
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        start = ent.start
        if start > 0 and doc[start - 1].text.rstrip(".").lower() in _TITLES:
            start -= 1
        people.append(doc[start : ent.end].text)
    return people


class RuleBasedAttributor:
    """Local, no-network speaker attribution: spaCy parse + fiction heuristics."""

    def attribute(self, spans: list[Span]) -> list[Attribution]:
        attributions: list[Attribution] = []
        last_mentioned: str | None = None
        speaker_history: list[str] = []  # distinct speakers, most recent last

        for index, span in enumerate(spans):
            if span.kind is SpanKind.NARRATION:
                labeled = _LABELED_LINE_RE.match(span.text)
                if labeled:
                    character = labeled.group(1).strip().title()
                    last_mentioned = character
                    if character in speaker_history:
                        speaker_history.remove(character)
                    speaker_history.append(character)

                    dialogue_span = Span(
                        text=labeled.group(2).strip(),
                        kind=SpanKind.DIALOGUE,
                        chapter_index=span.chapter_index,
                        paragraph_index=span.paragraph_index,
                        span_id=span.span_id,
                    )
                    attributions.append(
                        Attribution(span=dialogue_span, character=character, confidence=_NAMED_CONFIDENCE)
                    )
                    continue

                people = _person_entities(span.text)
                if people:
                    last_mentioned = people[-1]
                continue

            character, confidence = self._attribute_dialogue(spans, index, last_mentioned, speaker_history)
            if character != UNKNOWN:
                last_mentioned = character
                if character in speaker_history:
                    speaker_history.remove(character)
                speaker_history.append(character)

            attributions.append(Attribution(span=span, character=character, confidence=confidence))

        return attributions

    def _attribute_dialogue(
        self,
        spans: list[Span],
        index: int,
        last_mentioned: str | None,
        speaker_history: list[str],
    ) -> tuple[str, float]:
        for neighbor_index in (index - 1, index + 1):
            if not (0 <= neighbor_index < len(spans)):
                continue
            neighbor = spans[neighbor_index]
            if neighbor.kind is not SpanKind.NARRATION:
                continue
            subject = _speech_verb_subject(neighbor.text)
            if subject is None:
                continue
            if subject.lower() in _PRONOUN_SUBJECTS:
                if last_mentioned is not None:
                    return last_mentioned, _PRONOUN_CONFIDENCE
                continue
            return subject, _NAMED_CONFIDENCE

        if len(speaker_history) >= 2:
            return speaker_history[-2], _ALTERNATION_CONFIDENCE

        return UNKNOWN, 0.0
