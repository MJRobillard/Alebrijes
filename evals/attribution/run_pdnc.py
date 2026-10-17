"""Speaker-attribution eval against the Project Dialogism Novel Corpus (PDNC).

PDNC (Vishnubhotla et al., LREC 2022, CC-BY 4.0) hand-annotates the speaker
of every quotation in 28 public-domain novels. This runs each novel's text
through the real pipeline -- `ingest.parse_book` -> `segment_paragraph` ->
`RuleBasedAttributor`, chapter by chapter, exactly as `cast/build.py` does --
then aligns the predicted dialogue spans against PDNC's gold sub-quotations
and scores them.

The corpus is not vendored; clone it and pass its path:

    git clone --depth 1 https://github.com/Priya22/project-dialogism-novel-corpus.git
    python evals/attribution/run_pdnc.py --pdnc project-dialogism-novel-corpus

Unit of evaluation: one gold *sub-quotation* (PDNC splits a quote interrupted
by a speech tag -- `"Yes," said Jane, "of course."` -- into two), since that
is what this pipeline renders as one voiced line.

A prediction counts as correct only if it resolves to exactly the gold
character: predicted names are looked up against every character's PDNC
aliases (an exact alias first, else the name's last non-title token), and a
name that still matches more than one character (e.g. a bare "Bennet", when
both Mr. and Mrs. Bennet exist) is scored as ambiguous, not correct.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from alebrijes.attribute.rules import UNKNOWN, RuleBasedAttributor
from alebrijes.cast.build import LOW_CONFIDENCE_THRESHOLD
from alebrijes.ingest import parse_book
from alebrijes.segment import SpanKind, segment_paragraph

# How far ahead of the last aligned span to look for a gold sub-quotation's
# match. Gold quotes the segmenter skipped (e.g. two-word quotes, which it
# treats as emphasis by design) shouldn't drag the alignment pointer forward.
_ALIGN_WINDOW = 40

_TITLES = {"mr", "mrs", "miss", "ms", "dr", "sir", "lady", "lord", "colonel", "captain", "the"}


def _key(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


@dataclass
class Gold:
    quote_id: str
    text: str
    speaker: str
    quote_type: str


@dataclass
class Pred:
    text: str
    character: str
    confidence: float


def load_gold(novel_dir: Path) -> list[Gold]:
    rows = list(csv.DictReader(open(novel_dir / "quotation_info.csv", encoding="utf-8")))
    gold = []
    for row in rows:
        spans = ast.literal_eval(row["quoteByteSpans"])
        subs = ast.literal_eval(row["subQuotationList"])
        for (start, _), sub in sorted(zip(spans, subs)):
            if _key(sub):
                gold.append((start, Gold(row["quoteID"], sub, row["speaker"], row["quoteType"])))
    return [g for _, g in sorted(gold, key=lambda pair: pair[0])]


AliasIndex = tuple[dict[str, set[str]], dict[str, set[str]]]


def load_alias_index(novel_dir: Path) -> AliasIndex:
    """(exact alias -> PDNC main names, alias token -> main names), keyed by `_key`."""
    exact: dict[str, set[str]] = defaultdict(set)
    tokens: dict[str, set[str]] = defaultdict(set)
    for row in csv.DictReader(open(novel_dir / "character_info.csv", encoding="utf-8")):
        main = row["Main Name"]
        for alias in set(ast.literal_eval(row["Aliases"])) | {main}:
            exact[_key(alias)].add(main)
            for token in _key(alias).split():
                if token not in _TITLES:
                    tokens[token].add(main)
    return exact, tokens


def resolve(name: str, alias_index: AliasIndex) -> set[str]:
    """An exact alias match wins (PDNC lists "Darcy" as an alias of Mr. Darcy
    specifically); otherwise fall back to the name's last non-title token,
    which may legitimately match several characters (a bare "Bennet")."""
    exact, tokens = alias_index
    key = _key(name)
    if key in exact:
        return exact[key]
    words = [t for t in key.split() if t not in _TITLES]
    return tokens.get(words[-1], set()) if words else set()


def predict(novel_text: Path) -> tuple[list[Pred], int]:
    book = parse_book(novel_text)
    attributor = RuleBasedAttributor()
    preds = []
    for chapter in book.chapters:
        spans = [s for p in chapter.paragraphs for s in segment_paragraph(p, chapter.index)]
        for a in attributor.attribute(spans):
            if a.span.kind is SpanKind.DIALOGUE:
                preds.append(Pred(a.span.text, a.character, a.confidence))
    return preds, len(book.chapters)


def align(gold: list[Gold], preds: list[Pred]) -> list[Pred | None]:
    """Order-preserving greedy alignment by normalized text (exact, then containment)."""
    pred_keys = [_key(p.text) for p in preds]
    out: list[Pred | None] = []
    cursor = 0
    for g in gold:
        gk = _key(g.text)
        hit = None
        for j in range(cursor, min(cursor + _ALIGN_WINDOW, len(preds))):
            pk = pred_keys[j]
            if pk == gk or (len(gk) >= 12 and (gk in pk or pk in gk) and min(len(gk), len(pk)) / max(len(gk), len(pk)) >= 0.5):
                hit = j
                break
        if hit is None:
            out.append(None)
        else:
            out.append(preds[hit])
            cursor = hit + 1
    return out


def tier(confidence: float) -> str:
    if confidence >= 0.9:
        return "named speech-verb subject (0.9)"
    if confidence >= 0.5:
        return "pronoun -> last mention (0.55)"
    if confidence > 0:
        return "turn alternation (0.35)"
    return "unknown (0.0)"


def score(gold: list[Gold], aligned: list[Pred | None], alias_index) -> dict:
    c = Counter()
    by_tier = defaultdict(Counter)
    by_type = defaultdict(Counter)
    for g, p in zip(gold, aligned):
        c["gold"] += 1
        by_type[g.quote_type]["gold"] += 1
        if p is None:
            continue
        c["detected"] += 1
        by_type[g.quote_type]["detected"] += 1
        t = tier(p.confidence)
        by_tier[t]["n"] += 1
        if p.character == UNKNOWN:
            outcome = "unknown"
        else:
            candidates = resolve(p.character, alias_index)
            outcome = "correct" if candidates == {g.speaker} else ("ambiguous" if g.speaker in candidates else "wrong")
        c[outcome] += 1
        by_tier[t][outcome] += 1
        by_type[g.quote_type][outcome] += 1
        auto = p.confidence >= LOW_CONFIDENCE_THRESHOLD
        c["auto_accepted" if auto else "flagged_for_review"] += 1
        if auto and outcome == "correct":
            c["auto_accepted_correct"] += 1
    return {"counts": dict(c), "by_tier": {k: dict(v) for k, v in by_tier.items()},
            "by_quote_type": {k: dict(v) for k, v in by_type.items()}}


def merge(results: list[dict]) -> dict:
    total = {"counts": Counter(), "by_tier": defaultdict(Counter), "by_quote_type": defaultdict(Counter)}
    for r in results:
        total["counts"].update(r["counts"])
        for section in ("by_tier", "by_quote_type"):
            for k, v in r[section].items():
                total[section][k].update(v)
    return {"counts": dict(total["counts"]),
            "by_tier": {k: dict(v) for k, v in total["by_tier"].items()},
            "by_quote_type": {k: dict(v) for k, v in total["by_quote_type"].items()}}


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:5.1f}%" if d else "   n/a"


def report(name: str, r: dict) -> str:
    c = Counter(r["counts"])
    lines = [
        f"== {name}",
        f"  gold sub-quotations:            {c['gold']}",
        f"  detected as dialogue:           {c['detected']} ({_pct(c['detected'], c['gold'])} recall)",
        f"  speaker correct | detected:     {_pct(c['correct'], c['detected'])}",
        f"  speaker correct | all gold:     {_pct(c['correct'], c['gold'])}  (end to end)",
        f"  auto-accepted (conf >= {LOW_CONFIDENCE_THRESHOLD}):     {_pct(c['auto_accepted'], c['detected'])} of detected, "
        f"{_pct(c['auto_accepted_correct'], c['auto_accepted'])} correct",
        f"  flagged for review (< {LOW_CONFIDENCE_THRESHOLD}):     {_pct(c['flagged_for_review'], c['detected'])} of detected",
        f"  wrong / ambiguous / unknown:    {_pct(c['wrong'], c['detected'])} / {_pct(c['ambiguous'], c['detected'])} / {_pct(c['unknown'], c['detected'])}",
        "  by rule tier (accuracy, share of detected):",
    ]
    for t, v in sorted(r["by_tier"].items(), key=lambda kv: -kv[1].get("n", 0)):
        lines.append(f"    {t:34s} {_pct(v.get('correct', 0), v['n'])}  ({_pct(v['n'], c['detected'])})")
    lines.append("  by PDNC quote type (detected recall, accuracy | detected):")
    for t, v in sorted(r["by_quote_type"].items()):
        lines.append(f"    {t:12s} n={v['gold']:6d}  {_pct(v.get('detected', 0), v['gold'])}  {_pct(v.get('correct', 0), v.get('detected', 0))}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pdnc", type=Path, required=True, help="Path to a clone of the PDNC repo")
    parser.add_argument("--novels", nargs="*", help="Folder names under data/ (default: all)")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results.json")
    args = parser.parse_args()

    data = args.pdnc / "data"
    novels = args.novels or sorted(d.name for d in data.iterdir() if (d / "quotation_info.csv").exists())
    per_novel = {}
    for name in novels:
        t0 = time.monotonic()
        gold = load_gold(data / name)
        preds, n_chapters = predict(data / name / "novel_text.txt")
        result = score(gold, align(gold, preds), load_alias_index(data / name))
        result["chapters_detected"] = n_chapters
        result["seconds"] = round(time.monotonic() - t0, 1)
        per_novel[name] = result
        print(report(f"{name} ({n_chapters} chapters, {result['seconds']}s)", result), flush=True)

    overall = merge(list(per_novel.values()))
    print(report(f"ALL ({len(per_novel)} novels)", overall))
    args.out.write_text(json.dumps({"overall": overall, "novels": per_novel}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
