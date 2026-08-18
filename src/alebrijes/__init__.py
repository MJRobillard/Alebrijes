"""Alebrijes: local, multi-voice book-to-audiobook pipeline.

Pipeline stages, in order: ingest -> segment -> attribute -> cast -> synth -> assemble.
Each stage is its own module under `alebrijes.*` with a narrow, typed interface so
it can be tested and swapped independently. See PHASES.md at the repo root for the
rollout plan and per-phase verification.
"""

__version__ = "0.1.0"
