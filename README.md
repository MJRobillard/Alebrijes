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
(rendering, casting, previewing, and using imported `.pt`/`.safetensors` voice files).
Confirmed empirically: `alebrijes cast`/`render`/the GUI all work with zero
Hugging Face account or login — pocket-tts automatically falls back to the
open model when the gated one isn't accessible.

If you *do* want to clone a voice from an audio sample yourself (as opposed
to importing an already-made `.pt`/`.safetensors` file via the GUI's "Import a voice"
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

## Interactive narration (`/listen`)

On top of the offline render pipeline, the desktop app's `/listen` page adds
a live listening mode: the current line is shown as it plays, you can
interrupt with a chat question, branch into an improvised choose-your-own-
adventure, or have the narrator pause at checkpoints for your reaction or a
comprehension question. This talks to a **local Ollama server** — nothing
leaves the machine.

**Setup**: install [Ollama](https://ollama.com), then pull the two models
this uses:
```bash
ollama pull hermes3:8b          # chat/narration
ollama pull nomic-embed-text    # embeddings for spoiler-safe retrieval
```
`ollama serve` needs to be running before you start a session. If you want
different models (e.g. something smaller/faster on a CPU-only machine), set
these environment variables before launching the app, or just pick a
different pulled model from the dropdown in the session panel itself:
```bash
export ALEBRIJES_CHAT_MODEL=deepseek-r1:1.5b     # macOS/Linux
export ALEBRIJES_EMBED_MODEL=nomic-embed-text
# Windows (PowerShell): $env:ALEBRIJES_CHAT_MODEL = "deepseek-r1:1.5b"
```

**Prompting instructions live in `src/alebrijes/llm/narrator_guidelines.md`**
— not buried in Python, specifically so it can be read and edited without
touching code. It's Markdown with a YAML frontmatter block: the frontmatter
holds the four short per-mode fragments (structured, since there's exactly
one per interaction mode), and the Markdown body below it holds the
long-form guidelines shared by every mode (plain prose, since that reads
easier than a YAML block scalar). `llm/prompts.py` reads both fresh on
every prompt build.

**Frontmatter — one fragment per mode**, appended after the shared
guidelines and RAG excerpts to steer that specific interaction:

```yaml
---
modes:
  qa: >
    ...
  cyoa: >
    ...
  checkpoint: >
    ...
  learning: >
    ...
---
```
Edit `qa` if free-form questions aren't getting answered the way you want,
`cyoa` if improvised branches drift off-tone or don't steer back to the
book cleanly, `checkpoint` if pause-for-a-choice prompts don't read as a
natural in-story beat, or `learning` if comprehension questions or the
reaction to a listener's answer feel off.

**Markdown body — six shared sections**, included in every mode's prompt
regardless of which fragment above also applies:

| Section | Governs | Edit it if... |
|---|---|---|
| **No spoilers** | Refusing to reveal/imply anything past the listener's current position | Answers still leak future plot, or over-refuse things that already happened |
| **Ground answers in the retrieved excerpts** | Only using the RAG passages actually retrieved, not invented lore | The model is hallucinating details not in the book |
| **Checkpoints and comprehension questions** | How pause-for-choice and quiz-style prompts are phrased, and how answers get reacted to | Checkpoint/Learning mode feels like a quiz show instead of an in-story beat |
| **Improvised (choose-your-own-adventure) branches** | Tone consistency during improvisation, and how it steers back to canon | CYOA drifts off-tone, or the return-to-plot feels abrupt |
| **Stay in voice** | Never breaking character as "an AI" | The narrator ever refers to itself as an assistant/model |
| **Lead with a short sentence** | The first sentence being brief, so sentence-chunked audio (see below) starts playing almost immediately | Responses still open with a long sentence before saying anything |

Changes to this file take effect on the next session — no restart needed
beyond starting a new session, since it's read fresh each time
`llm/prompts.py` builds a system prompt.

**Spoken responses stream sentence-by-sentence**: as the model generates a
reply, each completed sentence is synthesized (in the book's own cloned
Narrator voice, via `synth/streaming.py`) and sent to the browser
immediately — not after the whole response finishes — so playback starts
as soon as the first sentence is ready. This is exactly why the "lead with
a short sentence" guideline above matters: a short opener means less time
sitting in silence before you hear anything.

## Testing

```bash
pytest
```
