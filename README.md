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

**`ffmpeg` is handled automatically** — bundled via the `imageio-ffmpeg`
pip package (a real static binary, not a system dependency), so the `pip
install -e ".[dev]"` step above is all that's needed. No `winget`/`brew`
step, and this is also how Phase 4's packaged installer will get it — same
mechanism for a source install and the shipped app, not two different
answers to "where does ffmpeg come from."

**A Hugging Face login is only needed if you want to clone your own voices
from audio** (`alebrijes voices add`) — not for normal use. pocket-tts has
two model variants: a gated, cloning-capable one
(`huggingface.co/kyutai/pocket-tts`) and an open, non-gated one
(`kyutai/pocket-tts-without-voice-cloining`) used for everything else
(rendering, casting, previewing, and using imported `.pt` voice files).
Confirmed empirically: `alebrijes cast`/`render`/the GUI all work with zero
Hugging Face account or login — pocket-tts automatically falls back to the
open model when the gated one isn't accessible.

If you *do* want to clone a voice from an audio sample yourself (as opposed
to importing an already-made `.pt` file via the GUI's "Import a voice"
screen), `alebrijes voices add` needs the gated model, once:

1. Create a free account at huggingface.co (if you don't have one).
2. Visit `https://huggingface.co/kyutai/pocket-tts` and click "Agree and
   access repository."
3. Authenticate locally so downloads are authorized:
   ```bash
   pip install huggingface_hub
   huggingface-cli login   # paste an access token from huggingface.co/settings/tokens
   ```

Phase 4's packaged installer bundles the open model directly — it never
touches the gate at all, so end users of the shipped app need no Hugging
Face account under any circumstance (see `PHASES.md`).

## Testing

```bash
pytest
```
