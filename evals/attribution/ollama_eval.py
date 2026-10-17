"""Local (Ollama) models vs. DeepSeek on speaker attribution + comprehension.

For each sampled PDNC quotation, a local model gets a ~1.2k-token passage
around it (what the interactive narrator would see via retrieval, unlike
DeepSeek's whole-novel context in deepseek_label.py) and must return:
- the speaker, constrained by Ollama structured output to a name from the
  novel's character list (or "Unknown");
- an answer to the comprehension question DeepSeek wrote for that quote.

Speakers are scored against PDNC's gold labels and against DeepSeek's.
Answers are saved beside DeepSeek's answer with a token-F1 overlap score as
a rough first signal. Judging their correctness is a separate step, done
once DeepSeek's questions themselves have been checked.

Sampling and caching: the novel is cut into overlapping passages
(`--passage-chars`, stride half that). Only quotes in a passage's second
half are evaluated, so each has at least half a passage of conversation
before it. Every request for a passage shares one prefix (instructions,
roster, passage) and only the final message changes, so Ollama reuses the
processed prefix across that passage's quotes.

    python evals/attribution/ollama_eval.py --pdnc project-dialogism-novel-corpus \
        --novel PrideAndPrejudice --models hermes3:8b llama3.1:8b
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from deepseek_label import canonical, load_quotes, roster  # noqa: E402
from run_pdnc import load_alias_index  # noqa: E402

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
# Fastest first, so partial results arrive early on CPU (8B models run at
# ~13 prompt tok/s and ~5 output tok/s on a 16-core CPU; 1.5B is ~6x faster).
DEFAULT_MODELS = [
    "qwen2.5:1.5b-instruct",
    "llama3.1:8b",
    "richardyoung/llama-3.1-8b-instruct-abliterated:latest",
    "hermes3:8b",
]

SYSTEM_PROMPT = """You are reading a passage from a novel. Quotations in the
passage are tagged with ID markers like ⟦Q12⟧ at their start; a quotation
interrupted by narration carries the same ID on each part.

You will be asked about one quotation ID. Return json with:
- "speaker": who speaks that quotation, exactly as one of the main names in
  the character list, or "Unknown". Use the conversation's turn-taking, not
  only nearby speech verbs.
- "answer": a concise answer to the question asked, based on the passage,
  in at most 25 words."""


def passages(text: str, size: int) -> list[tuple[int, int]]:
    stride = size // 2
    return [(start, min(start + size, len(text))) for start in range(0, max(1, len(text) - stride), stride)]


def build_prompt_prefix(text: str, quotes, span: tuple[int, int], roster_text: str) -> list[dict]:
    start, end = span
    inserts = sorted(
        (s, f"⟦{q.quote_id}⟧") for q in quotes for (s, _e) in q.spans if start <= s < end
    )
    pieces, cursor = [], start
    for pos, marker in inserts:
        pieces.append(text[cursor:pos])
        pieces.append(marker)
        cursor = pos
    pieces.append(text[cursor:end])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CHARACTERS:\n{roster_text}\n\nPASSAGE:\n{''.join(pieces)}"},
        {"role": "assistant", "content": "Ready."},
    ]


def token_f1(pred: str, gold: str) -> float:
    p, g = re.findall(r"\w+", pred.lower()), re.findall(r"\w+", gold.lower())
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def ask(
    client: httpx.Client, model: str, messages: list[dict], names: list[str], num_ctx: int, timeout: float = 300
) -> tuple[dict, dict]:
    schema = {
        "type": "object",
        "properties": {"speaker": {"type": "string", "enum": names}, "answer": {"type": "string"}},
        "required": ["speaker", "answer"],
    }
    body = {
        "model": model,
        "messages": messages,
        "format": schema,
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx, "num_predict": 160},
    }
    for attempt in range(3):
        try:
            r = client.post(OLLAMA_URL, json=body, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return json.loads(data["message"]["content"]), data
        except httpx.TimeoutException:
            raise
        except (httpx.HTTPError, json.JSONDecodeError, KeyError):
            if attempt == 2:
                raise
            time.sleep(5)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pdnc", type=Path, required=True)
    parser.add_argument("--novel", default="PrideAndPrejudice")
    parser.add_argument("--deepseek-run", default="deepseek-flash_low", help="llm_runs/<novel>_<this>/labels.jsonl")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--passages", type=int, default=5, help="Random passages to sample")
    parser.add_argument("--passage-chars", type=int, default=4800, help="~1.2k tokens")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--request-timeout", type=float, default=300,
                        help="Seconds; a request over this skips the rest of that model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "llm_runs" / "ollama")
    args = parser.parse_args()

    novel_dir = args.pdnc / "data" / args.novel
    text = (novel_dir / "novel_text.txt").read_text(encoding="utf-8")
    quotes = load_quotes(novel_dir)
    _, mains = roster(novel_dir)
    names = sorted(mains) + ["Unknown"]
    # Main names only: the alias list roughly doubles the prefix for a big
    # cast, and on CPU every prefix token costs ~75 ms for an 8B model.
    roster_text = "\n".join(f"- {name}" for name in sorted(mains))
    alias_index = load_alias_index(novel_dir)
    ds_path = Path(__file__).parent / "llm_runs" / f"{args.novel}_{args.deepseek_run}" / "labels.jsonl"
    deepseek = {r["quote_id"]: r for r in map(json.loads, ds_path.read_text().splitlines()) if r.get("question")}

    # Sample passages that have quotes (with a DeepSeek question) in their second half.
    candidates = []
    for span in passages(text, args.passage_chars):
        half = span[0] + (span[1] - span[0]) // 2
        targets = [q for q in quotes if half <= q.spans[0][0] < span[1] and q.quote_id in deepseek]
        if len(targets) >= 4:
            candidates.append((span, targets))
    rng = random.Random(args.seed)
    sample = sorted(rng.sample(candidates, min(args.passages, len(candidates))), key=lambda c: c[0][0])
    n_quotes = sum(len(t) for _, t in sample)
    print(f"{args.novel}: {len(sample)} passages, {n_quotes} quotations, models: {', '.join(args.models)}", flush=True)

    out_dir = args.out / args.novel
    out_dir.mkdir(parents=True, exist_ok=True)
    # DeepSeek on the exact same quotations, for a like-for-like row.
    same = [deepseek[q.quote_id] for _, targets in sample for q in targets]
    # Merged into any earlier summary, so models can be run one per invocation.
    summary_path = out_dir / "summary.json"
    summaries = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summaries["deepseek-flash (whole novel)"] = {
        "quotations": len(same),
        "speaker_accuracy": round(sum(r["speaker_correct"] for r in same) / len(same), 4),
    }
    with httpx.Client() as client:
        for model in args.models:
            # Checkpointed per quotation: each row is appended as soon as it's
            # answered, and a re-run skips quotations already in the file, so
            # an interrupted run resumes instead of starting over.
            path = out_dir / f"{model.replace('/', '_').replace(':', '_')}.jsonl"
            sampled_ids = {q.quote_id for _, targets in sample for q in targets}
            done = {}
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        if row["quote_id"] in sampled_ids:
                            done[row["quote_id"]] = row
            rows, t0, timed_out = list(done.values()), time.monotonic(), False
            if done:
                print(f"  {model}: resuming, {len(done)}/{n_quotes} quotations already done", flush=True)
            for span, targets in sample:
                targets = [q for q in targets if q.quote_id not in done]
                if timed_out:
                    break
                if not targets:
                    continue
                prefix = build_prompt_prefix(text, quotes, span, roster_text)
                passage_text = re.sub(r"\s+", " ", text[span[0] : span[1]])
                for q in targets:
                    ds = deepseek[q.quote_id]
                    ask_msg = {"role": "user", "content": (
                        f'Quotation {q.quote_id}: "{q.subs[0][:120]}"\n'
                        f"1) Who speaks quotation {q.quote_id}?\n2) {ds['question']}\nRespond in json."
                    )}
                    t_req = time.monotonic()
                    try:
                        pred, raw = ask(client, model, prefix + [ask_msg], names, args.num_ctx, args.request_timeout)
                    except httpx.TimeoutException:
                        print(f"  {model}: request over {args.request_timeout:.0f}s on {q.quote_id}; "
                              "skipping the rest of this model", flush=True)
                        timed_out = True
                        break
                    evidence = re.sub(r"\s+", " ", ds.get("evidence") or "").strip()
                    rows.append({
                        "quote_id": q.quote_id,
                        "quote_type": q.quote_type,
                        "gold_speaker": q.speaker,
                        "deepseek_speaker": ds["pred_speaker"],
                        "pred_speaker": pred.get("speaker"),
                        "speaker_correct": canonical(pred.get("speaker") or "", alias_index)
                        == canonical(q.speaker, alias_index),
                        "agrees_with_deepseek": pred.get("speaker") == ds["pred_speaker"],
                        "question": ds["question"],
                        "deepseek_answer": ds["answer"],
                        "pred_answer": pred.get("answer"),
                        "answer_token_f1": round(token_f1(pred.get("answer") or "", ds["answer"] or ""), 4),
                        "evidence_in_passage": bool(evidence) and evidence in passage_text,
                        "seconds": round(time.monotonic() - t_req, 2),
                        "prompt_eval_seconds": round(raw.get("prompt_eval_duration", 0) / 1e9, 2),
                        "output_tokens": raw.get("eval_count"),
                    })
                    r = rows[-1]
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    print(f"    {model} {q.quote_id}: {r['seconds']:.1f}s speaker={r['pred_speaker']} "
                          f"gold={r['gold_speaker']} {'OK' if r['speaker_correct'] else 'x'}", flush=True)
                print(f"  {model}: passage at char {span[0]} done ({len(rows)}/{n_quotes})", flush=True)

            if rows:
                summaries[model] = {**summarize(rows, time.monotonic() - t0), "timed_out": timed_out}
                print(f"  {model}: {json.dumps(summaries[model])}", flush=True)
            # Written after every model, so a long run yields partial results.
            summary_path.write_text(json.dumps(summaries, indent=2))

    print(json.dumps(summaries, indent=2))
    return 0


def summarize(rows: list[dict], seconds: float) -> dict:
    n = len(rows)
    in_passage = [r for r in rows if r["evidence_in_passage"]]
    by_type: dict[str, list[bool]] = {}
    for r in rows:
        by_type.setdefault(r["quote_type"], []).append(r["speaker_correct"])
    return {
        "quotations": n,
        "speaker_accuracy": round(sum(r["speaker_correct"] for r in rows) / n, 4),
        "speaker_accuracy_by_type": {t: round(sum(v) / len(v), 4) for t, v in sorted(by_type.items())},
        "agrees_with_deepseek": round(sum(r["agrees_with_deepseek"] for r in rows) / n, 4),
        "answer_token_f1": round(sum(r["answer_token_f1"] for r in rows) / n, 4),
        "answer_token_f1_evidence_in_passage": round(
            sum(r["answer_token_f1"] for r in in_passage) / len(in_passage), 4) if in_passage else None,
        "evidence_in_passage_rate": round(len(in_passage) / n, 4),
        "mean_seconds_per_quote": round(sum(r["seconds"] for r in rows) / n, 2),
        "mean_prompt_eval_seconds": round(sum(r["prompt_eval_seconds"] for r in rows) / n, 2),
        "wall_seconds": round(seconds, 1),
    }


if __name__ == "__main__":
    sys.exit(main())
