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

Early scaffold. `synth` and `assemble` are ported from pocket-tts-agent and
working. `ingest`, `segment`, `attribute`, and `cast` are stubbed interfaces —
see `PHASES.md` for what's next.

## Project layout

- `src/alebrijes/` — the pipeline, one module per stage
- `tests/` — unit tests, no model downloads required
- `PHASES.md` — the rebuild roadmap and per-phase verification

## Install

```bash
pip install -e ".[dev]"
```

## Testing

```bash
pytest
```
