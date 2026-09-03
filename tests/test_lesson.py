from __future__ import annotations

from pathlib import Path

from alebrijes.lesson import (
    QuestionResult,
    QuizProgress,
    QuizQuestion,
    load_coffee_premise,
    load_progress,
    load_quiz,
    parse_choice,
    save_progress,
)


def test_load_quiz_parses_questions_by_id(tmp_path: Path) -> None:
    path = tmp_path / "book.quiz.yaml"
    path.write_text(
        "questions:\n"
        "  q1:\n"
        "    prompt: \"What is an IAM role?\"\n"
        "    options:\n"
        "      A: \"A user\"\n"
        "      B: \"A delegated identity\"\n"
        "    correct: B\n",
        encoding="utf-8",
    )

    questions = load_quiz(path)

    assert set(questions.keys()) == {"q1"}
    assert questions["q1"].prompt == "What is an IAM role?"
    assert questions["q1"].options == {"A": "A user", "B": "A delegated identity"}
    assert questions["q1"].correct == "B"
    assert questions["q1"].praise is None


def test_load_quiz_parses_optional_praise(tmp_path: Path) -> None:
    path = tmp_path / "book.quiz.yaml"
    path.write_text(
        "questions:\n"
        "  q1:\n"
        "    prompt: \"What is an IAM role?\"\n"
        "    options:\n"
        "      A: \"A user\"\n"
        "      B: \"A delegated identity\"\n"
        "    correct: B\n"
        "    praise: \"You nailed it.\"\n",
        encoding="utf-8",
    )

    questions = load_quiz(path)

    assert questions["q1"].praise == "You nailed it."


def test_load_quiz_on_empty_file_returns_empty_dict(tmp_path: Path) -> None:
    path = tmp_path / "book.quiz.yaml"
    path.write_text("", encoding="utf-8")

    assert load_quiz(path) == {}


def test_load_progress_on_missing_file_returns_empty_progress(tmp_path: Path) -> None:
    progress = load_progress(tmp_path / "missing.progress.json", book_title="Fixture")

    assert progress.book_title == "Fixture"
    assert progress.answered == {}
    assert progress.score == (0, 0)


def test_save_then_load_progress_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "book.progress.json"
    progress = QuizProgress(
        book_title="Fixture",
        answered={
            "q1": QuestionResult(correct=True, attempts=1, last_answer="B"),
            "q2": QuestionResult(correct=False, attempts=2, last_answer="A"),
        },
    )

    save_progress(progress, path)
    loaded = load_progress(path)

    assert loaded == progress
    assert loaded.score == (1, 2)


def test_save_progress_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "book.progress.json"
    save_progress(QuizProgress(book_title="Fixture"), path)

    assert path.exists()


def test_parse_choice_extracts_standalone_letter() -> None:
    assert parse_choice("B") == "B"
    assert parse_choice("b") == "B"
    assert parse_choice("I think B") == "B"
    assert parse_choice("b) that one") == "B"


def test_parse_choice_takes_the_last_letter_mentioned() -> None:
    assert parse_choice("Between A and B... I'll go with C") == "C"


def test_parse_choice_returns_none_when_unparseable() -> None:
    assert parse_choice("I'm not sure") is None
    assert parse_choice("") is None


def test_parse_choice_does_not_match_letters_inside_words() -> None:
    assert parse_choice("Actually never mind") is None


def test_load_coffee_premise_returns_stripped_text(tmp_path: Path) -> None:
    path = tmp_path / "book.coffee.md"
    path.write_text("\n  A mysterious fog rolls into town.  \n\n", encoding="utf-8")

    assert load_coffee_premise(path) == "A mysterious fog rolls into town."
