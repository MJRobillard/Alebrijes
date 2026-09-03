from __future__ import annotations

from alebrijes.cast.build import NARRATOR, build_cast_file
from alebrijes.ingest import Book, Chapter, Paragraph


def _fixture_book() -> Book:
    chapter = Chapter(
        title="Chapter One",
        index=0,
        paragraphs=[
            Paragraph(text='Colin said, "Let\'s take another path."', index=0),
            Paragraph(text='Audrey replied, "Agreed, it is safer."', index=1),
            Paragraph(text="The room fell quiet.", index=2),
            Paragraph(text='Someone spoke from the dark: "Where has everyone gone?"', index=3),
        ],
    )
    return Book(title="Fixture Book", chapters=[chapter])


def test_build_cast_file_includes_narrator_and_detected_characters() -> None:
    result = build_cast_file(_fixture_book())

    names = [entry.character for entry in result.cast_file.entries]
    assert names[0] == NARRATOR
    assert "Colin" in names
    assert "Audrey" in names
    assert all(entry.voice_state_path for entry in result.cast_file.entries)


def test_build_cast_file_does_not_duplicate_narrator_for_script_format() -> None:
    # A script with an explicit NARRATOR: line produces an
    # Attribution(character="Narrator", ...) alongside the other characters
    # -- NARRATOR must not also get added via the generic characters-seen
    # accumulation on top of its hardcoded first entry.
    chapter = Chapter(
        title="Chapter One",
        index=0,
        paragraphs=[
            Paragraph(text="NARRATOR: It began.", index=0),
            Paragraph(text="FELICITY: Hello.", index=1),
            Paragraph(text="NARRATOR: And so it continued.", index=2),
        ],
    )
    book = Book(title="Fixture Book", chapters=[chapter], source_format="script")

    result = build_cast_file(book)

    names = [entry.character for entry in result.cast_file.entries]
    assert names == [NARRATOR, "Felicity"]


def test_build_cast_file_reports_low_confidence_attributions() -> None:
    # The 4th line has no speech verb and no named character nearby, so it can
    # only be resolved (if at all) by the low-confidence alternation fallback
    # — exactly the kind of line the cast-review step exists to catch.
    result = build_cast_file(_fixture_book())

    low_confidence = result.low_confidence()
    assert len(low_confidence) >= 1
    assert all(a.confidence < 0.5 for a in low_confidence)


def test_build_cast_file_assigns_stable_voice_per_character() -> None:
    first = build_cast_file(_fixture_book())
    second = build_cast_file(_fixture_book())

    voices_first = {e.character: e.voice_state_path for e in first.cast_file.entries}
    voices_second = {e.character: e.voice_state_path for e in second.cast_file.entries}
    assert voices_first == voices_second
