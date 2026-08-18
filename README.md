# alebrijes

Local, multi-voice book-to-audiobook engine. Cloned voices are cast to characters
by a rule-based speaker-attribution pass, then rendered with a local TTS model —
no cloud dependency, no per-character cost.

Successor to `pocket-tts-agent` (`~/projects/audio`), which proved the core
idea (local voice cloning, cached voice states, parallel synthesis, chaptered
`.m4b` output) but grew as a pile of scripts. This repo is a clean rebuild
around a six-stage pipeline: **ingest → segment → attribute → cast → synth →
assemble**. See `PHASES.md` for the rollout plan and verification checklist.

## Status

The full pipeline (ingest/segment/attribute/cast/synth/assemble) works for
`.epub`/`.pdf`/`.txt` via the `alebrijes` CLI, and a desktop GUI
(`alebrijes gui`) built on pywebview + a local FastAPI backend is up and
verified end to end on Windows. See `PHASES.md` for the rollout plan, what's
done, and what's next.

## Project layout

- `src/alebrijes/` — the pipeline, one module per stage, plus `server.py` /
  `gui.py` / `frontend/` for the desktop app
- `tests/` — unit tests, no model downloads required except spaCy's English
  model for `attribute` (see below)
- `PHASES.md` — the rebuild roadmap and per-phase verification

## Install

**`ffmpeg` must be on PATH** — not pip-installable, not bundled yet (that's
a Phase 4 packaging concern), and needed for the final `.m4b` build step.
`alebrijes render` checks for it up front and fails fast with a clear error
if it's missing, rather than after synthesizing every chapter first.

```bash
winget install ffmpeg     # Windows
brew install ffmpeg       # macOS
```

**Use a dedicated virtual environment** — not your global/system Python.
Confirmed on a real test machine that already had other CUDA-pinned ML
tooling installed (`unsloth-zoo`, `xformers`, `torchaudio`, `torchvision`):
installing into the global environment both pulled in a CUDA torch build
this project doesn't want (see below) *and* broke those other projects'
pinned CUDA torch versions when corrected. A venv isolates Alebrijes from
whatever else is on the machine, in both directions.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

**Then install the CPU-only torch build first, before everything else** —
this matters: `pip install -e ".[dev]"` alone pulls torch from the default
PyPI index with no preference for CPU vs. CUDA, and on a machine with a
CUDA-capable Python/OS combination but no actual NVIDIA GPU, that resolves to
a CUDA build carrying real, needless size and load-time overhead (confirmed
on a real Windows test machine: `torch==2.6.0+cu118` installed with
`torch.cuda.is_available()` False). Installing the CPU wheel explicitly
first avoids this — pip won't touch an already-satisfied dependency on the
next step.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
python -m spacy download en_core_web_sm  # required for alebrijes.attribute
```

If you skip this and end up with a CUDA build anyway, `alebrijes gui` and
`/api/health` both surface a warning about it (`alebrijes/diagnostics.py`) —
not silent, just not prevented by default.

## Testing

```bash
pytest
```
