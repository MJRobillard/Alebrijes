"""Attribute stage: assign a character label + confidence to each dialogue span.

This is where the rule-based speaker-attribution system plugs in. It's kept
behind a `Protocol` rather than a concrete class so the rule-based
implementation (assumed to already exist and work) can be dropped in without
this module needing to know its internals — and so a smarter attributor can
replace it later without touching `cast` or `synth`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from alebrijes.segment import Span


@dataclass
class Attribution:
    span: Span
    character: str
    confidence: float  # 0.0-1.0; low-confidence attributions get flagged for cast review


class Attributor(Protocol):
    def attribute(self, spans: list[Span]) -> list[Attribution]: ...
