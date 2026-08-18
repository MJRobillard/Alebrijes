# Rebuild roadmap

Six phases, each shippable on its own, each with a defined "done" so the next
phase never starts on an unverified foundation.

## Phase 0 — Repo hygiene

Clean repo from commit one: no committed media, voice states, caches, or
generated output; a real `.gitignore`; one `pyproject.toml` as the single
source of truth for entry points.

**Verification:** `git status` is clean on a fresh clone. Done as of this
scaffold's initial commit.

## Phase 1 — Core library

The six pipeline stages as real modules (`ingest`, `segment`, `attribute`,
`cast`, `synth`, `assemble`), each independently testable, sharing one config
object instead of per-script argparse defaults.

Done:
- `segment` — narration/dialogue span splitting, ported and generalized from
  pocket-tts-agent's `examples/multi_ci_tts.py`.
- `attribute` — `RuleBasedAttributor` in `alebrijes/attribute/rules.py`: spaCy
  dependency parse to find a speech verb's syntactic subject, handling both
  word orders, plus two fiction-specific fallback heuristics:
  most-recently-mentioned character for pronoun subjects, and alternation
  between the two most recently established speakers for untagged
  back-and-forth dialogue. Needs `python -m spacy download en_core_web_sm`
  once, locally — no network calls per line, no torch dependency for this
  stage.
- `ingest/epub.py` — OPF/spine parsing, HTML text extraction, and
  chapter-header detection (incl. the image-alt-text fallback), ported from
  `examples/ci_tts_from_epub.py`. Left behind: that file's `--book`
  multi-book-omnibus filtering, built for one specific EPUB. Falls back to a
  single chapter when no chapter headers are found rather than dropping
  content. Hardened against documented real-world EPUB failure modes: DRM
  (detected via `META-INF/encryption.xml`, the signal DeDRM tooling keys
  off), malformed/missing XML, manifest entries missing from the archive,
  and OPFs missing namespace declarations — each a clear error or a graceful
  skip instead of a confusing crash.
- `ingest/pdf.py` — ported header/footer-stripping from
  `examples/ci_tts_from_pdf.py`, plus mitigations for pypdf's documented weak
  spots: `extraction_mode="layout"` for its long-standing missing-space/
  reading-order issue, NFKC ligature normalization, a word-wrap dehyphenation
  heuristic (documented limitation in its own docstring — can't perfectly
  distinguish a wrapped word from a real hyphenated compound), a
  blank-password decrypt attempt before reporting real password protection,
  and a clear error for scanned/image-only PDFs (OCR is out of scope).
  Chapter detection reuses epub's heading heuristic.
- `ingest/txt.py` — blank-line paragraph splitting, falling back to
  sentence-count reflow (ported from `examples/reflow_txt_paragraphs.py`) for
  wall-of-text files with no blank lines.
- `cast/build.py` — `build_cast_file(book)` runs segment + attribute over
  every chapter and produces a `CastFile` with a placeholder voice per
  detected character (deterministic pick from pocket-tts's built-in voices),
  plus the full attribution list so low-confidence lines can be surfaced.
  Wired into the CLI: `alebrijes cast <book>` works for `.epub`/`.pdf`/`.txt`.

Remaining work: validate attribution accuracy against real chapters (the
Project Dialogism Novel Corpus — 36k annotated quotes across 22 novels — is a
good benchmark for this once there's a reason to tune the heuristics rather
than just extend coverage); the `--book` multi-book-omnibus EPUB case is
still unported.

**Verification:** `pytest` green, including `test_scaffold.py` (every stage
imports, CLI `--help` works, unsupported formats fail loudly),
`test_segment.py` / `test_attribute_rules.py` (dialogue detection and
rule-based attribution behave as designed), `test_ingest_epub.py` /
`test_ingest_pdf.py` / `test_ingest_txt.py` (extraction and edge cases —
DRM, malformed XML, encryption, scanned PDFs, encoding fallback — against
synthetic/generated fixtures, no committed binaries), `test_cast_build.py` /
`test_cli_cast.py` (end-to-end cast generation across all three formats), and
`test_synth_agent.py` / `test_assemble.py` (ported synth/assemble logic
behaves identically to pocket-tts-agent's tested originals).

## Phase 2 — CLI is the application

One `alebrijes` command (`voices`, `cast`, `render` subcommands) replacing
pocket-tts-agent's four overlapping scripts (`tts-menu`, `ci-tts`,
`ci-tts-from-epub`, `epub-reader`). Installable and usable end to end with no
GUI.

`cast`, `render`, and `voices` are all done. `alebrijes cast book.epub`
followed by `alebrijes render book.epub --cast cast.yaml` was verified for
real (a local pocket-tts model + ffmpeg, not just unit tests) against a
fixture book: a correctly-chaptered, multi-voice `.m4b` with per-line voice
switching between the narrator and two attributed characters, including a
mixed cast (one builtin voice, one real cloned `.pt`). `voices add` was
verified against a real local mp3. `render.py`'s multi-voice synthesis is
sequential, not parallel (see its module docstring) — an accepted v1
trade-off, revisit if render time becomes a real bottleneck on full-length
books. `voices add` only accepts local audio files by design — no URL/YouTube
support at the CLI layer (the underlying `create_voice_state` still supports
those if called directly from Python).

**Verification:** `pipx install .` (or `uv tool install .`) from a fresh
clone, then `alebrijes cast book.epub` followed by `alebrijes render
book.epub --cast cast.yaml` produces a playable `.m4b`, with no source
checkout beyond the installed package.

## Phase 3 — Desktop GUI rewrite

Replace pocket-tts-agent's 1,800-line Tkinter monolith with PySide6 (Qt): an
MVVM split where the view talks only to a thin app-state class, which talks
only to the Phase 1 library. The one new screen that matters: cast review —
character list, voice picker, inline playback of low-confidence attributions.

**Verification:** manual walkthrough — open a book, review/adjust cast,
render — with no script editing or filesystem knowledge required. Checked off
by hand; UI correctness isn't unit-testable.

## Phase 4 — Packaging & distribution

PyInstaller or briefcase build into a signed installer, bundling or
first-run-fetching model weights with a visible progress step.

**Verification:** the built installer launches and renders a book on a
machine with no Python/dev environment installed.

## Phase 5 — Mobile (gated)

Not started until Phase 3 has real usage showing people want this away from
their desktop. Local voice cloning/synthesis is compute-heavy and does not
belong on a phone — this would be a thin client (library browsing, cast
review) against a local/self-hosted render server, not a port of the
renderer itself.

**Verification:** an explicit go/no-go decision recorded here before any code
is written, based on actual Phase 3 usage evidence.
