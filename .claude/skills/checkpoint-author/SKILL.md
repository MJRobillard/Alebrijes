---
name: checkpoint-author
description: >-
  Converts a source study document plus a set of practice questions into an
  Alebrijes interactive-narration script (NAME: text lines, # chapter
  headings, QUIZ: markers) and its companion <name>.quiz.yaml, with each
  question placed at the true end of its chapter -- the point where
  playback is already idle between chapters, so the frontend just asks it
  there, no mid-chapter interruption logic needed. Use when the user wants
  to turn study material (e.g. AWS/technical notes) and practice questions
  into a chapter-quiz Alebrijes book, or asks to "author checkpoints",
  "place quiz markers", or "preprocess a script" for interactive narration.
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

One checkpoint per chapter, always at the chapter's true end (see
Placement) — this isn't a tunable granularity. If there are more questions
than natural chapter divisions in the source doc, split the doc into more
chapters rather than stacking multiple questions into one chapter.

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

## Placement — always the chapter's true last line

Every `QUIZ:` marker goes immediately after the **last** `NARRATOR:` (or
other speaker) line of its chapter — the line directly before the next `#`
heading, or the last line of the file for the final chapter. Never mid-
chapter. This is the whole point of this format: a chapter boundary is
already a natural pause (nothing else is playing), so the frontend can just
ask the question there with no interrupt/resume machinery — a mid-chapter
placement would need that machinery and defeats the design.

No foreshadowing requirement — a question doesn't need to be directly
answered by the chapter it's attached to; inference is part of the intended
learning experience. Pick whichever chapter is topically closest to the
question and be able to say why if asked, but the *position* within that
chapter is never a judgment call — always the last line, no exceptions.

One question per chapter (see Inputs). Every input question must be placed
exactly once — never dropped, never duplicated across two checkpoints.

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
    # Must be the chapter's true last line -- the whole point of this
    # format (see Placement, above).
    chapter_index, paragraph_index, _ = span_id.split(".")
    chapter = book.chapters[int(chapter_index)]
    last_index = chapter.paragraphs[-1].index
    assert int(paragraph_index) == last_index, (
        f"{span_id} isn't the last line of chapter {chapter_index} "
        f"(last is index {last_index}) -- move the QUIZ marker there"
    )

placed_ids = set(book.quiz_checkpoints.values())
assert placed_ids == set(questions), "every input question must be placed exactly once, no drops/dupes"

chapters_used = [span_id.split(".")[0] for span_id in book.quiz_checkpoints]
assert len(chapters_used) == len(set(chapters_used)), "at most one checkpoint per chapter"
```

If this raises anything, fix the output and re-run — don't hand back a
script that doesn't parse or references a missing question.
