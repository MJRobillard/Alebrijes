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

Done: `segment` (narration/dialogue span splitting, ported and generalized
from pocket-tts-agent's `examples/multi_ci_tts.py`) and `attribute`
(`RuleBasedAttributor` in `alebrijes/attribute/rules.py` — spaCy dependency
parse to find a speech verb's syntactic subject, handling both word orders,
plus two fiction-specific fallback heuristics: most-recently-mentioned
character for pronoun subjects, and alternation between the two most recently
established speakers for untagged back-and-forth dialogue). Needs
`python -m spacy download en_core_web_sm` once, locally — no network calls
per line, no torch dependency for this stage.

Remaining work: port epub/pdf/txt ingest from pocket-tts-agent's
`examples/ci_tts_from_epub.py` / `ci_tts_from_pdf.py`; wire `cast` generation
(run `RuleBasedAttributor` over a book's segments, write the resulting
`cast.yaml`) into the CLI's `cast` subcommand; validate attribution accuracy
against real chapters (the Project Dialogism Novel Corpus — 36k annotated
quotes across 22 novels — is a good benchmark for this later).

**Verification:** `pytest` green, including `test_scaffold.py` (every stage
imports, CLI `--help` works, stubs fail loudly), `test_segment.py` /
`test_attribute_rules.py` (dialogue detection and rule-based attribution
behave as designed), and `test_synth_agent.py` /
`test_assemble.py` (ported synth/assemble logic behaves identically to
pocket-tts-agent's tested originals).

## Phase 2 — CLI is the application

One `alebrijes` command (`voices`, `cast`, `render` subcommands) replacing
pocket-tts-agent's four overlapping scripts (`tts-menu`, `ci-tts`,
`ci-tts-from-epub`, `epub-reader`). Installable and usable end to end with no
GUI.

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
