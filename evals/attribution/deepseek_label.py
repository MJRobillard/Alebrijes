"""LLM speaker attribution + comprehension Q&A over a PDNC novel, via DeepSeek.

Two outputs per gold quotation:
1. **Attribution:** the speaker (and addressees), scored against PDNC's
   hand annotations. This is the LLM upper bound to compare
   `RuleBasedAttributor` against (see run_pdnc.py), measured on PDNC's gold
   quote boundaries so it isolates *who said it* from dialogue detection.
2. **A comprehension question + answer** grounded in the text at that
   quote, with a verbatim evidence span. Saved as a dataset for later
   evaluating smaller local models (e.g. the interactive narrator's Ollama
   models).

Caching instead of batching: DeepSeek has no batch API, but it caches
repeated prompt prefixes automatically. Every request shares one fixed
prefix (instructions, then the character roster and the *entire* novel with
each quote tagged by its ID), and only the final message (which quote IDs to
label) changes. After the first request, the novel is read from cache at
about 2% of the uncached price.

    python evals/attribution/deepseek_label.py --pdnc project-dialogism-novel-corpus \
        --novel PrideAndPrejudice --env-file ../scripter/.env [--limit 80]

Resumable: results append to labels.jsonl per batch, and a re-run skips
quote IDs that are already labeled.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from run_pdnc import _key, load_alias_index  # noqa: E402

API_URL = "https://api.deepseek.com/chat/completions"

# USD per 1M tokens: (cache hit, cache miss, output), peak rates. Off-peak is
# half. From api-docs.deepseek.com/quick_start/pricing, checked 2026-10-17.
PRICES = {"deepseek-flash": (0.006, 0.30, 1.20), "deepseek-v4-pro": (0.044, 1.32, 3.96)}
PEAK_UTC_HOURS = {1, 2, 3, 6, 7, 8, 9}  # Mon-Fri only

SYSTEM_PROMPT = """You are annotating a novel for a dataset. The user gives you the
novel's character list and full text. Every quotation in the text is tagged
with an ID marker like ⟦Q12⟧ placed at its start; a quotation interrupted by
narration ("Yes," said Jane, "of course.") carries the same ID on each part.

For each quotation ID you are asked about, return:
- "speaker": who speaks it, written EXACTLY as one of the main names in the
  character list (the part before " | aliases:"), or "Unknown" if the text
  truly does not determine it. Use the surrounding conversation (turn-taking,
  who was addressed, who replies), not just nearby speech verbs.
- "addressees": main names of who the quotation is spoken to (may be empty).
- "question": one reading-comprehension question about the story at that
  moment that needs the surrounding scene to answer: why the speaker says
  it, what led up to it, how others react, what it implies about a
  character or relationship. Avoid questions whose answer is just the
  quotation's own words restated. It must be answerable from the text up
  to and including that quotation (no later plot), and must NOT ask who is
  speaking. Vary the question type across quotations.
- "answer": a concise, correct answer in 1-2 sentences.
- "evidence": a short span copied verbatim from the novel text (at most 25
  words, quotation marks and ID markers left out) that supports the answer.

Respond in json only, exactly in this shape, with one result per requested ID
and in the same order:
{"results": [{"id": "Q12", "speaker": "Elizabeth Bennet", "addressees": ["Mr. Darcy"],
  "question": "...", "answer": "...", "evidence": "..."}]}"""


@dataclass
class Quote:
    quote_id: str
    text: str
    speaker: str
    addressees: list[str]
    quote_type: str
    spans: list[tuple[int, int]]
    subs: list[str]


def load_quotes(novel_dir: Path) -> list[Quote]:
    quotes = []
    for row in csv.DictReader(open(novel_dir / "quotation_info.csv", encoding="utf-8")):
        quotes.append(
            Quote(
                quote_id=row["quoteID"],
                text=row["quoteText"],
                speaker=row["speaker"],
                addressees=ast.literal_eval(row["addressees"]) if row["addressees"].startswith("[") else [],
                quote_type=row["quoteType"],
                spans=[tuple(s) for s in ast.literal_eval(row["quoteByteSpans"])],
                subs=ast.literal_eval(row["subQuotationList"]),
            )
        )
    return quotes


def tag_novel(text: str, quotes: list[Quote]) -> tuple[str, int]:
    """Inserts a ⟦ID⟧ marker at the start of every sub-quotation. PDNC spans
    are character offsets for ~98% of sub-quotations; the rest are found by
    searching near the stated offset. Returns (tagged text, parts untagged)."""
    inserts: list[tuple[int, str]] = []
    missed = 0
    for q in quotes:
        for (start, end), sub in zip(q.spans, q.subs):
            if text[start:end] != sub:
                found = text.find(sub, max(0, start - 500), end + 500)
                if found < 0:
                    missed += 1
                    continue
                start = found
            inserts.append((start, f"⟦{q.quote_id}⟧"))
    pieces, cursor = [], 0
    for pos, marker in sorted(inserts):
        pieces.append(text[cursor:pos])
        pieces.append(marker)
        cursor = pos
    pieces.append(text[cursor:])
    return "".join(pieces), missed


def roster(novel_dir: Path) -> tuple[str, set[str]]:
    lines, mains = [], set()
    for row in csv.DictReader(open(novel_dir / "character_info.csv", encoding="utf-8")):
        main = row["Main Name"]
        aliases = sorted(set(ast.literal_eval(row["Aliases"])) - {main})
        mains.add(main)
        lines.append(f"- {main}" + (f" | aliases: {', '.join(aliases)}" if aliases else ""))
    return "\n".join(lines), mains


def canonical(name: str, alias_index) -> str:
    """Maps any alias to its PDNC main name when that's unambiguous."""
    exact, _ = alias_index
    mains = exact.get(_key(name), set())
    return next(iter(mains)) if len(mains) == 1 else name


class Usage:
    def __init__(self, model: str) -> None:
        self.model = model
        self.lock = threading.Lock()
        self.totals = {"requests": 0, "cache_hit": 0, "cache_miss": 0, "output": 0, "reasoning": 0, "usd": 0.0}

    def add(self, usage: dict) -> None:
        hit, miss = usage.get("prompt_cache_hit_tokens", 0), usage.get("prompt_cache_miss_tokens", 0)
        out = usage.get("completion_tokens", 0)
        now = datetime.now(timezone.utc)
        peak = now.weekday() < 5 and now.hour in PEAK_UTC_HOURS
        p_hit, p_miss, p_out = (p * (1 if peak else 0.5) / 1e6 for p in PRICES[self.model])
        with self.lock:
            t = self.totals
            t["requests"] += 1
            t["cache_hit"] += hit
            t["cache_miss"] += miss
            t["output"] += out
            t["reasoning"] += usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            t["usd"] += hit * p_hit + miss * p_miss + out * p_out


def call(client: httpx.Client, args, messages: list[dict]) -> tuple[dict, dict]:
    body = {
        "model": args.model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": 32000,
    }
    if args.effort == "none":
        body["thinking"] = {"type": "disabled"}
    else:
        body["reasoning_effort"] = args.effort
    last_error = None
    for attempt in range(4):
        try:
            r = client.post(API_URL, json=body, timeout=600)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"].get("content") or ""
            return json.loads(content), data.get("usage", {})
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            time.sleep(2**attempt * 3)
    raise RuntimeError(f"DeepSeek request failed after retries: {last_error!r}")


def label_batch(client, args, prefix: list[dict], batch: list[Quote], mains: set[str], usage: Usage) -> list[dict]:
    """Labels one batch. IDs that come back missing or with an off-roster
    speaker are re-asked once on their own before being recorded as invalid."""
    wanted = {q.quote_id: q for q in batch}
    results: dict[str, dict] = {}
    pending = list(batch)
    for _ in range(2):
        if not pending:
            break
        listing = "\n".join(f'{q.quote_id}: "{q.subs[0][:70]}"' for q in pending)
        ask = {"role": "user", "content": f"Annotate these quotation IDs, in this order, as json:\n{listing}"}
        data, u = call(client, args, prefix + [ask])
        usage.add(u)
        for item in data.get("results", []):
            qid = str(item.get("id", ""))
            if qid in wanted and qid not in results and (item.get("speaker") in mains or item.get("speaker") == "Unknown"):
                results[qid] = item
        pending = [q for q in pending if q.quote_id not in results]
    for q in pending:
        results[q.quote_id] = {"id": q.quote_id, "speaker": None, "invalid": True}
    return [results[q.quote_id] for q in batch]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pdnc", type=Path, required=True)
    parser.add_argument("--novel", default="PrideAndPrejudice")
    parser.add_argument("--env-file", type=Path, help="File with DEEPSEEK_API_KEY=... (or set it in the environment)")
    parser.add_argument("--model", default="deepseek-flash", choices=sorted(PRICES))
    parser.add_argument("--effort", default="low", choices=["none", "low", "high", "max"])
    parser.add_argument("--batch", type=int, default=40, help="Quotation IDs per request")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Only the first N quotations (a pilot run)")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "llm_runs")
    args = parser.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key and args.env_file:
        for line in args.env_file.read_text().splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise SystemExit("Set DEEPSEEK_API_KEY or pass --env-file")

    novel_dir = args.pdnc / "data" / args.novel
    quotes = load_quotes(novel_dir)
    tagged, missed = tag_novel((novel_dir / "novel_text.txt").read_text(encoding="utf-8"), quotes)
    roster_text, mains = roster(novel_dir)
    alias_index = load_alias_index(novel_dir)
    if args.limit:
        quotes = quotes[: args.limit]

    run_dir = args.out / f"{args.novel}_{args.model}_{args.effort}"
    run_dir.mkdir(parents=True, exist_ok=True)
    labels_path = run_dir / "labels.jsonl"
    done = set()
    if labels_path.exists():
        done = {json.loads(line)["quote_id"] for line in labels_path.read_text().splitlines() if line.strip()}
    todo = [q for q in quotes if q.quote_id not in done]
    batches = [todo[i : i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"{args.novel}: {len(quotes)} quotations ({len(done)} already labeled), {len(batches)} batches; "
          f"{missed} sub-quotation parts could not be tagged", flush=True)

    # The cacheable prefix: identical bytes on every request.
    prefix = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CHARACTERS:\n{roster_text}\n\nNOVEL TEXT:\n{tagged}"},
        {"role": "assistant", "content": "Ready. Send the quotation IDs to annotate."},
    ]
    novel_text = re.sub(r"\s+", " ", (novel_dir / "novel_text.txt").read_text(encoding="utf-8"))
    usage = Usage(args.model)
    write_lock = threading.Lock()
    t0 = time.monotonic()

    def run(batch: list[Quote]) -> int:
        items = label_batch(client, args, prefix, batch, mains, usage)
        rows = []
        for q, item in zip(batch, items):
            evidence = item.get("evidence") or ""
            rows.append({
                "novel": args.novel,
                "quote_id": q.quote_id,
                "quote_text": q.text,
                "quote_type": q.quote_type,
                "gold_speaker": q.speaker,
                "gold_addressees": q.addressees,
                "pred_speaker": item.get("speaker"),
                "pred_addressees": item.get("addressees", []),
                "speaker_correct": item.get("speaker") is not None
                and canonical(item["speaker"], alias_index) == canonical(q.speaker, alias_index),
                "addressee_overlap": bool(
                    {canonical(a, alias_index) for a in item.get("addressees") or []}
                    & {canonical(a, alias_index) for a in q.addressees}
                ),
                "question": item.get("question"),
                "answer": item.get("answer"),
                "evidence": evidence,
                "evidence_verbatim": bool(evidence) and re.sub(r"\s+", " ", evidence).strip() in novel_text,
                "model": args.model,
                "effort": args.effort,
            })
        with write_lock, open(labels_path, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        return len(rows)

    with httpx.Client(headers={"Authorization": f"Bearer {key}"}) as client:
        if batches:  # first request alone, so the novel prefix is cached before fanning out
            run(batches[0])
            print(f"  batch 1/{len(batches)} done (cache warm-up): {usage.totals}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run, b) for b in batches[1:]]
            for n, future in enumerate(as_completed(futures), start=2):
                future.result()
                print(f"  batch {n}/{len(batches)} done, ${usage.totals['usd']:.4f} so far", flush=True)

    rows = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
    summary = summarize(rows, usage.totals, time.monotonic() - t0)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def summarize(rows: list[dict], totals: dict, seconds: float) -> dict:
    by_type: dict[str, list[bool]] = {}
    for r in rows:
        by_type.setdefault(r["quote_type"], []).append(r["speaker_correct"])
    n = len(rows)
    addressee_rows = [r for r in rows if r["gold_addressees"]]
    return {
        "quotations": n,
        "speaker_accuracy": round(sum(r["speaker_correct"] for r in rows) / n, 4) if n else None,
        "speaker_accuracy_by_type": {t: round(sum(v) / len(v), 4) for t, v in sorted(by_type.items())},
        "unknown": sum(r["pred_speaker"] == "Unknown" for r in rows),
        "invalid": sum(r["pred_speaker"] is None for r in rows),
        "addressee_any_overlap": round(
            sum(r["addressee_overlap"] for r in addressee_rows)
            / len(addressee_rows), 4) if addressee_rows else None,
        "qa_pairs": sum(bool(r["question"] and r["answer"]) for r in rows),
        "evidence_verbatim_rate": round(sum(r["evidence_verbatim"] for r in rows) / n, 4) if n else None,
        "usage_this_run": {**totals, "usd": round(totals["usd"], 4)},
        "wall_seconds_this_run": round(seconds, 1),
    }


if __name__ == "__main__":
    sys.exit(main())
