"""Lesson content: practice-question quizzes, persisted score, and a
freely-written coffee/mystery premise for CYOA grounding.

Mirrors transcript.py's dataclass + load/save pattern. A quiz's running
score is always *derived* from `QuizProgress.answered`
(`QuizProgress.score`), never stored redundantly, so it can't drift out of
sync with the per-question record.

Naming note: this is the *quiz* score, unrelated to this repo's RAG
(retrieval) system in `alebrijes.rag` -- never call it "RAG score", that
name is already taken by something else entirely.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


@dataclass
class QuizQuestion:
    id: str
    prompt: str
    options: dict[str, str]  # {"A": "...", "B": "...", ...}
    correct: str  # a key of `options`
    # Optional author-written reaction for a correct answer, spoken instead
    # of going through the LLM (see session/__init__.py's handle_user_message
    # -- a correct answer is graded deterministically and never needs the
    # LLM to author a reaction). Falls back to a generic rotation when unset.
    praise: Optional[str] = None


@dataclass
class QuestionResult:
    correct: bool
    attempts: int
    last_answer: str


@dataclass
class QuizProgress:
    book_title: str
    answered: dict[str, QuestionResult] = field(default_factory=dict)

    @property
    def score(self) -> tuple[int, int]:
        correct = sum(1 for result in self.answered.values() if result.correct)
        return correct, len(self.answered)


def load_quiz(path: Path) -> dict[str, QuizQuestion]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    questions = data.get("questions", {}) or {}
    return {
        question_id: QuizQuestion(
            id=question_id,
            prompt=question["prompt"],
            options=question["options"],
            correct=question["correct"],
            praise=question.get("praise"),
        )
        for question_id, question in questions.items()
    }


def load_progress(path: Path, *, book_title: str = "") -> QuizProgress:
    """Missing file -> a fresh, empty progress record (a lesson's first run
    has no prior progress on disk yet -- not an error)."""
    if not path.exists():
        return QuizProgress(book_title=book_title)

    data = json.loads(path.read_text(encoding="utf-8"))
    answered = {
        question_id: QuestionResult(**result) for question_id, result in data.get("answered", {}).items()
    }
    return QuizProgress(book_title=data.get("book_title", book_title), answered=answered)


def save_progress(progress: QuizProgress, path: Path) -> None:
    payload = {
        "book_title": progress.book_title,
        "answered": {
            question_id: {"correct": r.correct, "attempts": r.attempts, "last_answer": r.last_answer}
            for question_id, r in progress.answered.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def parse_choice(text: str) -> Optional[str]:
    """Extracts a standalone A/B/C/D letter from free-form user text, e.g.
    "B", "I think B", "b)" -> "B". Takes the *last* match ("Between A and
    B... I'll go with B" -> "B") since a spoken/typed answer often walks
    through options before landing on a final one. Returns None if nothing
    parses as a choice at all."""
    matches = _CHOICE_RE.findall(text.strip())
    return matches[-1].upper() if matches else None


def load_coffee_premise(path: Path) -> str:
    """Reads a coffee/mystery premise file verbatim -- same
    read-and-return-text pattern as llm/prompts.py's load_guidelines()."""
    return path.read_text(encoding="utf-8").strip()
