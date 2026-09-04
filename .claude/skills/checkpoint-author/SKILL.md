---
name: checkpoint-author
description: >-
  Converts a source study document plus a set of practice questions into an
  Alebrijes interactive-narration script (NAME: text lines, # chapter
  headings, QUIZ: markers) and its companion <name>.quiz.yaml. Use when the
  user wants to turn study material (e.g. AWS/technical notes) and practice
  questions into a lesson-mode Alebrijes book, or asks to "author
  checkpoints", "place quiz markers", or "preprocess a script" for
  interactive narration.
---

# Checkpoint Author

Turns a plain source document + a question bank into files Alebrijes's
existing pipeline ingests as-is (`ingest/script.py`, `lesson.py`) — this
skill produces no new file format, only content in formats that already
exist and are already parsed by the app.

## Inputs

- A source document (prose/notes, any length).
- A set of practice questions, each with a prompt, 2–4 lettered options,
  and which single letter is correct. Optional per-question praise line for
  a correct answer. (Multi-select answers aren't supported by the current
  grading code — never emit more than one correct letter.)
- Optional granularity: `chapter` (default — at most one checkpoint per
  chapter, near its end) or `dense` (a checkpoint after roughly every
  significant subsection). Ask if unclear and it matters for the doc's
  length.

If the user hasn't supplied the source doc or the questions inline, ask for
them — don't invent either.

## Output

Two files, named after the source doc (e.g. `aws-notes.txt` →
`aws-notes.script.txt` + `aws-notes.quiz.yaml`), written next to the source
unless told otherwise.

### 1. The script file — exact grammar, no deviation

- `# <Chapter Title>` starts a new chapter (Markdown-heading style).
- Every other non-blank line is `NAME: text`, name in ALL CAPS. For study
  material this is almost always `NARRATOR: <paragraph>` — split the source
  into paragraph-sized NARRATOR lines; don't dump a whole chapter as one
  giant line.
- `QUIZ: <id>` attaches to the line immediately **before** it — put it
  right after the NARRATOR line you want the checkpoint to follow.
- **No other syntax is supported.** No comments, no blank-line-separated
  metadata, nothing but plain text after the `NAME:` prefix. A line that
  doesn't match `# `, `QUIZ: `, or `NAME: text` makes `ingest/script.py`
  raise `ValueError` — see that module's docstring for the full grammar if
  anything here is ambiguous.
- `QUIZ` is reserved — never use it as a speaker name.

### 2. The quiz.yaml file

```yaml
questions:
  <id>:
    prompt: "..."
    options:
      A: "..."
      B: "..."
      C: "..."
      D: "..."
    correct: B
    praise: "..."   # optional
```

One entry per question the script references via `QUIZ: <id>` — ids must
match exactly, letter-for-letter.

## Placement

No foreshadowing requirement — a question doesn't need to be directly
answered by the exact passage it's attached to; inference is part of the
intended learning experience. Place each question at a chapter/section
boundary that's topically related to it, and be able to say what passage or
section title grounded that placement if asked — but don't force an exact
match if there's no clean one; a topically-adjacent placement is fine.

Every input question must be placed exactly once. Never drop a question,
never duplicate one across two checkpoints.

## Self-check before finishing

Run this against your own output before calling the task done — it mirrors
Alebrijes's actual ingest path, so a failure here means real playback would
also fail:

```python
from pathlib import Path
from alebrijes.ingest.script import parse
from alebrijes.lesson import load_quiz

book = parse(Path("<script output path>"))
questions = load_quiz(Path("<quiz.yaml output path>"))

assert book.quiz_checkpoints, "no QUIZ markers were placed"
for span_id, question_id in book.quiz_checkpoints.items():
    assert question_id in questions, f"{span_id} points at missing question {question_id!r}"

placed_ids = set(book.quiz_checkpoints.values())
assert placed_ids == set(questions), "every input question must be placed exactly once, no drops/dupes"
```

If this raises anything, fix the output and re-run — don't hand back a
script that doesn't parse or references a missing question.
