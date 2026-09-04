"""Round-trip check for the checkpoint-author Skill's output contract
(.claude/skills/checkpoint-author/SKILL.md) -- fully offline, no model call.

`aws_iam_basics.script.txt` / `.quiz.yaml` under tests/fixtures/ are
representative output in the skill's target format (hand-authored following
the skill's own instructions, since a Skill can't be invoked inside a
pytest run) alongside `aws_iam_basics.questions.yaml`, the input question
bank the skill was given. This doesn't test the skill's live placement
judgment -- it tests that output in this shape is exactly what
`ingest.script.parse` and `lesson.load_quiz` actually accept, which is the
one thing that must never silently break.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from alebrijes.ingest.script import parse
from alebrijes.lesson import load_quiz

FIXTURES = Path(__file__).parent / "fixtures"
SCRIPT_PATH = FIXTURES / "aws_iam_basics.script.txt"
QUIZ_PATH = FIXTURES / "aws_iam_basics.quiz.yaml"
INPUT_QUESTIONS_PATH = FIXTURES / "aws_iam_basics.questions.yaml"


def test_script_output_parses_with_no_exceptions() -> None:
    book = parse(SCRIPT_PATH)

    assert book.source_format == "script"
    assert book.chapters


def test_quiz_yaml_output_loads_with_no_exceptions() -> None:
    questions = load_quiz(QUIZ_PATH)

    assert questions


def test_every_checkpoint_points_at_a_question_that_actually_exists() -> None:
    book = parse(SCRIPT_PATH)
    questions = load_quiz(QUIZ_PATH)

    assert book.quiz_checkpoints, "the script placed no QUIZ markers at all"
    for span_id, question_id in book.quiz_checkpoints.items():
        assert question_id in questions, (
            f"checkpoint at {span_id} references question {question_id!r}, "
            "which isn't in the quiz.yaml -- a checkpoint pointing at a "
            "missing question is exactly the silent breakage this test "
            "exists to catch"
        )


def test_every_input_question_was_placed_exactly_once() -> None:
    book = parse(SCRIPT_PATH)
    input_questions = yaml.safe_load(INPUT_QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]

    placed_ids = list(book.quiz_checkpoints.values())

    assert set(placed_ids) == set(input_questions), "every input question must appear in the output"
    assert len(placed_ids) == len(set(placed_ids)), "no question should be placed at more than one checkpoint"


def test_quiz_yaml_matches_input_question_content() -> None:
    """The skill's job is placement, not rewriting the questions -- prompt,
    options, and the correct answer should come through unchanged."""
    questions = load_quiz(QUIZ_PATH)
    input_questions = yaml.safe_load(INPUT_QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]

    for question_id, expected in input_questions.items():
        actual = questions[question_id]
        assert actual.prompt == expected["prompt"]
        assert actual.options == expected["options"]
        assert actual.correct == expected["correct"]
