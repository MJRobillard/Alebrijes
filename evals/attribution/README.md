# Speaker-attribution eval (PDNC)

How often does Alebrijes give a line of dialogue to the right character?
Measured against the **Project Dialogism Novel Corpus** (Vishnubhotla, Hammond
& Hirst, LREC 2022; CC-BY 4.0): **48,803 hand-annotated quotation segments
across 28 public-domain novels**, from *Pride and Prejudice* to *The Sun Also
Rises*.

`run_pdnc.py` runs each novel's raw text through the real pipeline, chapter
by chapter (`ingest.parse_book` → `segment_paragraph` →
`RuleBasedAttributor`, the same path `cast/build.py` uses). It then aligns
the predicted dialogue spans against PDNC's gold sub-quotations and scores
the speaker of each one.

```bash
git clone --depth 1 https://github.com/Priya22/project-dialogism-novel-corpus.git
python evals/attribution/run_pdnc.py --pdnc project-dialogism-novel-corpus   # ~9 min, CPU
```

## Results

Overall numbers for all 28 novels. Per-novel and per-tier breakdowns are in
`results.json` (current) and `results_baseline.json` (before the fixes
below).

| | Baseline | Current |
|---|---|---|
| Dialogue segments detected (recall) | 43.1% | **82.7%** |
| Speaker correct, of detected | 42.7% | 38.4% |
| **Speaker correct, end to end** (of all 48,803) | 18.4% | **31.8%** |
| Named speech-tag tier ("said Darcy"): accuracy | 63.4% | **76.0%** |
| Auto-accepted (confidence ≥ 0.5): accuracy | 49.4% | 56.8% |
| Sent to the human cast-review step (< 0.5) | 22.4% | 56.1% |
| Ambiguous name ("Bennet" for two Bennets) | 4.0% | 0.8% |

Every one of the 28 novels improved end to end. The gains range from
+2.5 points (*The Invisible Man*) to +29.5 points (*Winnie-the-Pooh*).

## What the eval found and fixed

1. **Untagged dialogue was read by the narrator.** A quote counted as
   dialogue only if a speech verb ("said") appeared nearby, so 98% of PDNC's
   *implicit* quotes (untagged back-and-forth) were missed. Now a paragraph
   that opens with a quote is a dialogue turn
   (`segment/__init__.py`). Implicit-quote recall went from 2.2% to 85.4%.
2. **Titles were dropped from names.** The parser returned only the head
   token, so "Mrs. Bennet said" became "Bennet", and Mr. and Mrs. Bennet
   collapsed into one voice. Speaker names now keep their compound tokens,
   and PERSON entities keep a preceding title (`attribute/rules.py`).
3. **Common nouns were cast as characters.** "cried his wife" produced a
   character called "wife". Only proper nouns and the known pronouns are
   accepted as subjects now.
4. **`.txt` books parsed as one chapter.** Gutenberg's "CHAPTER I." headings
   weren't recognized, so all 28 novels came out as a single chapter.
   Roman numerals are now accepted (`ingest/txt.py`).

## LLM attribution + comprehension dataset (DeepSeek)

`deepseek_label.py` asks `deepseek-flash` (V4.1 Flash, `reasoning_effort:
low`) to attribute every PDNC quotation. The model sees the **whole novel**
with each quote tagged by ID, and for each quote it also writes a
comprehension question, an answer, and a verbatim evidence span. The Q&A is
a test set for the smaller local models that run the interactive narrator.

DeepSeek has no batch API, but it caches repeated prompt prefixes
automatically. Every request shares one prefix (instructions, then the
character list, then the full tagged novel). Only the final message, the
next 40 quote IDs, changes. After the first request, about 95% of input
tokens are cache hits at 2% of the uncached price.

| | *Pride and Prejudice* | *The Sport of the Gods* |
|---|---|---|
| Quotations | 1,270 | 640 |
| **Speaker accuracy** | **99.5%** | **100%** |
| Implicit (untagged) quotes | 99.2% | 100% |
| Addressee overlap with gold | 97.5% | 96.1% |
| Q&A pairs (evidence found verbatim in the novel) | 1,270 (93.7%) | 640 (97.2%) |
| Input tokens served from cache | 96.4% | 92.9% |
| Cost / wall time (off-peak) | $0.23 / 6.3 min | $0.11 / 4.0 min |

For comparison, the rule-based pipeline above gets 34.3% of *Pride and
Prejudice*'s quote segments right end to end. That isn't quite the same
measurement: the LLM is given PDNC's gold quote boundaries, while the
pipeline has to find the quotes first. *The Sport of the Gods* (Dunbar, 1902) is a lesser-known control
for memorization of *Pride and Prejudice*. One caveat: PDNC itself has been
public since 2022, so training-data exposure to the annotations can't be
ruled out. Of the 6 *Pride and Prejudice* misses, 4 come from turn-swaps in
the rapid Darcy/Elizabeth exchanges and 1 is a PDNC naming quirk
("Denny" vs. "Mr. Denney").

Labels land in `llm_runs/<novel>_<model>_<effort>/labels.jsonl`, one
quotation per line: gold and predicted speaker and addressees, question,
answer, and evidence. A per-run `summary.json` records accuracy and token
cost.

```bash
python evals/attribution/deepseek_label.py --pdnc project-dialogism-novel-corpus \
    --novel PrideAndPrejudice --env-file path/to/.env   # DEEPSEEK_API_KEY=...
```

## Local models vs. DeepSeek (Ollama, CPU)

`ollama_eval.py` asks local models the same two things: who speaks the
quote, and DeepSeek's comprehension question for it. Unlike DeepSeek, a
local model sees only a ~1.2k-token passage around the quote, which is what
the interactive narrator would retrieve. Ollama's structured output limits
the speaker to names on the character list. Each passage's prompt prefix is
reused across its quotes, so after the first quote only the question gets
processed (an 8B model's first quote takes 143 s; later ones take 7 s of
prompt processing). Results are saved per quote, and
`run_ollama_evals.sh` resumes an interrupted run.

Sample: 5 random passages per novel (seed 0), 85 quotes. Everything ran on
a 16-core CPU with no GPU.

| Model | Quotes | Speaker accuracy | Answer token-F1 vs. DeepSeek | s / quote |
|---|---|---|---|---|
| `deepseek-flash` (whole novel) | 85 | 100% | — | — |
| `qwen2.5:1.5b-instruct` | 85 | **65.9%** (P&P 63.2%, Sport 68.1%) | 0.28 | 7.7 |
| `llama3.1:8b` (preliminary, run stopped early) | 17 | **94.1%** (Qwen on the same 17: 52.9%) | 0.30 | 32.1 |

Token-F1 measures word overlap only. Answer correctness gets judged once
DeepSeek's questions themselves have been verified. Still to run: the
`llama3.1:8b` vs. abliterated comparison (does refusal removal cost
accuracy?) and `hermes3:8b`, the narrator's default model.

## Reading the numbers

- **The confidence tiers are calibrated.** Named speech tags are right 76% of
  the time, pronoun resolution 34%, and turn alternation 28%. Everything
  below 0.5 confidence is routed to the cast-review screen instead of being
  silently trusted. The fixes raised recall at the cost of more review: most
  newly detected untagged quotes land in the 28%-accurate alternation tier.
- **Rules have hit their ceiling on untagged turns.** Attributing those
  needs conversational context: who was addressed, and who spoke two turns
  ago. That is the job for a model-based attributor behind the same
  `Attributor` protocol (`attribute/__init__.py`), and this harness is how
  it would be measured.
- **The scoring is strict.** A prediction counts only if it resolves to
  exactly the annotated character via PDNC's alias lists. A bare surname
  shared by two characters scores as wrong, even when a listener could
  probably tell from context.
- **Alignment is by text.** Predicted spans are matched to gold
  sub-quotations in order, by normalized text. A gold quote the pipeline
  never detected counts against end-to-end accuracy, not against "correct,
  of detected".
