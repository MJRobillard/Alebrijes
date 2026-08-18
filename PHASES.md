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

Replace pocket-tts-agent's 1,800-line Tkinter monolith with **pywebview +
a local FastAPI backend**, not PySide6/Qt as originally sketched — revisited
once "easy to set up and download on both Mac and Windows" became the
explicit priority. Decision record:

- **UI shell: pywebview.** A native OS window (WKWebView on macOS,
  WebView2/Edge engine on Windows) hosting an HTML/CSS/JS frontend — not the
  user's actual browser, no URL bar/tabs, indistinguishable from a normal
  desktop app. Chosen over Electron (its bundled Chromium adds ~150MB on top
  of an already-nontrivial ML payload) and over PySide6 (same PyInstaller
  packaging story, but Qt widget styling is more work to make look good than
  HTML/CSS, and an HTML frontend is directly reusable for the Phase 5
  thin-client idea).
- **Backend: local FastAPI**, not pywebview's built-in JS↔Python bridge —
  chosen specifically for real render-progress streaming (a render takes
  minutes; SSE/WebSockets map naturally onto that, matching the progress
  dashboard already in `synth/parallel.py`) and because the same frontend
  can talk to `localhost` today and a remote render server later
  (Phase 5) with zero rewrite. The bridge option was the safer default (no
  listening socket at all — see below) and remains the fallback if the
  hardening below turns out to be more than this needs.
- **Required hardening** (a local HTTP server is a real, well-documented
  attack surface — DNS rebinding lets a malicious webpage in the user's
  actual browser drive an unauthenticated localhost API; this bitten real
  apps, e.g. early Jupyter): bind strictly to `127.0.0.1`, never `0.0.0.0`;
  bind to an OS-assigned ephemeral port (`port=0`), not a fixed one, to avoid
  collisions; require a random per-launch secret token embedded in the page
  pywebview loads and sent on every request (the Jupyter pattern); validate
  the `Origin` header on every request. All four are required together, not
  optional extras — this is what closes the gap with the no-server option.

Architecture: MVVM-equivalent split where the frontend only calls the local
API, which only calls the Phase 1 library — no pipeline logic in
frontend/route-handler code. The one new screen that matters: cast review —
character list, voice picker, inline playback of low-confidence attributions.

**Backend done** (`alebrijes/server.py`): all four hardening points
implemented and verified against a real socket, not just `TestClient` —
confirmed unreachable from the LAN interface, all four behaviors (missing
token, wrong token, wrong `Origin`, missing `Origin`) checked over real HTTP.
Routes: `/api/health`, `/api/cast` (synchronous), `/api/render` (starts a
background thread, returns a `job_id` immediately) + `/api/render/{job_id}`
(polls status; chapter-level progress via `render_book`'s new
`on_chapter_done` callback — not per-line; real SSE/WebSocket streaming is a
documented future refinement, not built yet). `create_app()` takes
injectable `ingest_book`/`render_fn` so tests don't need the real pocket-tts
model loaded.

**Shell done** (`alebrijes/gui.py`, `alebrijes/frontend/index.html`):
launcher starts the server thread, then opens a pywebview window navigated
to `http://host:port/?token=...` (the Jupyter token-in-URL pattern) — the
index route is deliberately unauthenticated (the page has to load before it
can present a token) but verified to never leak the server's own token into
that response. A single-file vanilla-JS frontend (no framework) covers the
full flow: choose book (native OS file dialog via `FileApi.pick_file`,
routed outside the HTTP API since it's an OS-shell capability, not
application logic) → build cast → **review/edit each character's voice
inline, save** (new `PUT /api/cast`) → render with live chapter-progress
polling. `alebrijes gui` / `alebrijes-gui` both launch it.

**Verified on real hardware**: `alebrijes gui` run for real on Windows —
window launched, pywebview's file dialog worked (after fixing a deprecated
API call caught by this run — `OPEN_DIALOG` → `FileDialog.OPEN`), and a
render triggered pocket-tts's first-run model download correctly. This
sandbox couldn't do this itself (Linux without PyGObject for the GTK
backend, and Linux isn't a shipping target anyway) — real-hardware
verification from the user closed that gap.

That same run surfaced a real Phase 4 gap: a plain `pip install -e ".[dev]"`
pulled `torch==2.6.0+cu118` with `torch.cuda.is_available()` False — the
CUDA-build-with-no-GPU scenario the Phase 4 sizing assumed wouldn't happen.
Fixed two ways: install docs now say to install the CPU wheel explicitly
first (README), and `alebrijes/diagnostics.py` surfaces a clear warning
(`alebrijes gui`'s startup print, and `/api/health`'s `warnings` field,
shown as a banner in the frontend) when it happens anyway, rather than
silently eating the extra size/load time. The Phase 4 CI build pipeline
below must enforce the CPU wheel too, not just document it — a signed
installer built without that step ships the same oversized/slower artifact.

Remaining work: the render-progress bar is chapter-granularity only
(matches the backend's current resolution); no inline playback of
low-confidence lines yet (the review list shows text + confidence, not audio
preview) — both documented as known v1 gaps, not oversights.

**Verification:** manual walkthrough — open a book, review/adjust cast,
render (watching live progress over the local API) — with no script editing
or filesystem knowledge required. Checked off by hand once the shell exists;
UI correctness isn't unit-testable. The FastAPI routes themselves are tested
with `TestClient`, no GUI involved, same as the rest of this codebase's
pure-logic-first pattern.

## Phase 4 — Packaging & distribution

PyInstaller into a signed installer (`.dmg`/`.pkg` on macOS, `.exe`/`.msi` on
Windows), one per OS built on that OS's own CI runner. Decision record from
sizing this out concretely:

- **Bundle the full offline stack, don't first-run-download it.** CPU-only
  torch (~190MB — the CUDA build this dev machine happened to have installed
  is 1.8GB and irrelevant; no consumer Mac has an NVIDIA GPU and Apple
  Silicon uses MPS, not CUDA) + the pocket-tts model weights (~225MB,
  measured) + spacy/ffmpeg/numpy/scipy/etc. comes to roughly **550–700MB**
  total — a real but ordinary download. Given this project's whole
  differentiator is "local and private," a single "it just works, no
  internet needed after install" download beats a smaller installer that
  needs a progress bar and network access before first use.
- **The build venv must install CPU-only torch explicitly** — confirmed for
  real (Windows test, Phase 3 verification): a plain `pip install` in a
  fresh venv resolved a CUDA build (`torch==2.6.0+cu118`) with no GPU to use
  it, not the CPU wheel the sizing above assumes. The CI step must be
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  *before* installing the rest of the app, on both the macOS and Windows
  runners — this is a build-pipeline requirement, not just an install-docs
  note (see README), since a signed installer built without it ships the
  same oversized/slower artifact to every user, silently.
- **macOS: notarization is mandatory**, not optional — Gatekeeper blocks or
  scares off unsigned apps on current macOS. Needs an Apple Developer ID
  ($99/year) and, specific to PyInstaller, one extra entitlement
  (`com.apple.security.cs.allow-unsigned-executable-memory`) for its
  unsigned-memory-mapping trick.
- **Windows: a standard OV code-signing cert (~$150–250/year) is enough** —
  EV certificates stopped bypassing SmartScreen in 2024, so paying the EV
  premium purely to dodge SmartScreen warnings isn't justified anymore.
  SmartScreen reputation still builds gradually with download counts either
  way. Certs are now capped at 1-year validity (a Feb 2026 rule change), so
  annual renewal is unavoidable regardless of cert type.
- **Budget ~$250–350/year ongoing** for both platforms' signing, plus CI time
  for macOS notarization — this is a recurring cost, not a one-time setup
  step.

**Verification:** the built installer launches and renders a book on a
machine with no Python/dev environment installed, with no Gatekeeper/
SmartScreen block on a clean machine (tested via a VM with no prior trust of
the signing identity, not just the developer's own already-trusted machine).

### Build pipeline

`pyinstaller` runs in `--onedir` mode, not `--onefile` — at ~550–700MB,
`--onefile` would re-extract the whole payload into a temp dir on *every*
launch (a multi-second delay each time), where `--onedir` unpacks once at
install time. Built separately on `macos-latest` and `windows-latest` GitHub
Actions runners — a Mac app can't be cross-built from Windows or vice versa.

1. `pyinstaller` (onedir) bundles the FastAPI backend + pywebview shell +
   torch/spacy/pocket-tts weights/ffmpeg into a platform-specific app folder.
2. **macOS**: `codesign` the `.app` with the Developer ID cert → `xcrun
   notarytool submit` → `xcrun stapler staple` (ticket must be stapled, not
   just submitted, so the app still passes Gatekeeper offline) → wrap in a
   `.dmg` via `create-dmg` for the drag-to-Applications experience.
3. **Windows**: `signtool sign` the `.exe` → build the installer with Inno
   Setup (a single `.iss` script) — produces the Start Menu shortcut and the
   "Add or Remove Programs" uninstaller entry, which a bare exe/zip doesn't
   give you.
4. A signed-but-unwrapped zip of the same `.app`/`.exe` folder is also worth
   publishing as a secondary "portable" download (no install footprint, no
   admin rights needed, e.g. for running off a USB drive) — it costs nothing
   extra since it's the same signed artifact, just zipped instead of
   installer-wrapped. Not the primary/recommended download: no Gatekeeper
   drag-in experience, no Start Menu entry, no uninstaller. Zipping never
   substitutes for signing — Gatekeeper and SmartScreen both check the
   binary's signature at launch regardless of how it was delivered.
5. Both installers (plus the optional portable zips) attached to a GitHub
   Release; the README/landing page links directly to "Download for Mac" /
   "Download for Windows" pointing at those release assets.

## Phase 5 — Mobile (gated)

Not started until Phase 3 has real usage showing people want this away from
their desktop. Local voice cloning/synthesis is compute-heavy and does not
belong on a phone — this would be a thin client (library browsing, cast
review) against a local/self-hosted render server, not a port of the
renderer itself.

**Verification:** an explicit go/no-go decision recorded here before any code
is written, based on actual Phase 3 usage evidence.
