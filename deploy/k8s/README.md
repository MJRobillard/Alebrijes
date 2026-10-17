# Distributed rendering on Kubernetes

Rendering a book is slow, CPU-bound TTS work, and every chapter is
independent. This setup spreads one book across a cluster with **one pod
per chapter**, then stitches the results back into a single chaptered
`.m4b`. It runs on a local [kind](https://kind.sigs.k8s.io) cluster
(1 control plane + 3 workers). The same Jobs run on any cluster with a
shared `ReadWriteMany` volume.

```
                  ┌─ render pod (chapter 0) ─┐
cast Job ──► Indexed Job ── render pod (chapter 1) ──► assemble Job ──► book.m4b
(speakers)        └─ render pod (chapter N) ─┘        (+ transcript.json)
                         shared /data volume (parts/)
```

| Step | Kubernetes object | CLI it runs |
|---|---|---|
| Speaker attribution | `Job` (`cast-job.yaml`) | `alebrijes cast` |
| Per-chapter synthesis | Indexed `Job`, `parallelism: N` (`render-job.yaml`) | `alebrijes render-chapter` (chapter = `JOB_COMPLETION_INDEX`) |
| Stitch + transcript | `Job` (`assemble-job.yaml`) | `alebrijes assemble` |

## Reliability

- **Per-chapter retries.** `backoffLimitPerIndex: 2` retries a failed
  chapter on its own. One bad chapter never restarts the book.
- **Idempotent parts.** `render_chapter_part` writes each chapter's WAV and
  timing sidecar under temporary names and renames them only when done. A
  re-run pod finds the finished part and exits in under a second, before it
  even loads the model.
- **Offline pods.** The image includes the model weights, every built-in
  voice, and the spaCy model, and it runs with `HF_HUB_OFFLINE=1`. Render
  pods make no network calls.
- **One core per pod.** Synthesis is single-threaded, so the system scales
  out with more pods rather than more threads. Each pod requests 1 CPU and
  2 GiB.

The local path (`alebrijes render`) runs the same `render_chapter_part`
loop followed by `assemble_parts`. `tests/test_render.py` checks that a
distributed render and a local render produce identical transcripts.

## Run it

Prerequisites: Docker Desktop with **WSL integration** enabled (if on WSL),
plus `kind` and `kubectl` on your `PATH`.

```bash
# 1. a public-domain benchmark book: the first 8 chapters of Pride and Prejudice
curl -L -o /tmp/pg1342.txt https://www.gutenberg.org/cache/epub/1342/pg1342.txt
python deploy/k8s/make_bench_book.py /tmp/pg1342.txt --chapters 8

# 2. cluster + image + cast -> render -> assemble (prints timings and pod placement)
PARALLELISM=8 deploy/k8s/run.sh

# 3. scaling benchmark at 1/2/4/8 pods -> deploy/k8s/bench.csv
deploy/k8s/bench.sh            # or pick pod counts: deploy/k8s/bench.sh 1 4 12
```

Output: `deploy/k8s/data/out/bench.m4b` plus `bench.transcript.json`.
Tear down with `kind delete cluster --name alebrijes`.

## Results

**First end-to-end run** (2026-10-17): kind 4-node cluster on one 16-core
WSL2 machine, 8-chapter *Pride and Prejudice* benchmark book (11.6k words),
`PARALLELISM=8`.

| Step | Wall time |
|---|---|
| Cast Job (attribution) | 9 s |
| Render Job: 8 pods across 3 workers (3/3/2) | 735 s |
| Assemble Job (8 parts → `.m4b` + transcript) | 65 s |
| **Output** | **60.8 min of audio, 8 chapter markers, 4.97× real time** |

With one pod per chapter, the wall time is set by the longest chapter
(Chapter 6, 11.8 min of audio), and each pod synthesizes at roughly real-time
speed on its one core. The scaling benchmark below measures the speedup
directly instead of estimating it.

**Scaling benchmark:** _pending; `bench.sh` fills this in from `bench.csv`._

| Pods | Render wall time | Speedup | Audio rendered / wall time |
|---|---|---|---|
| 1 | | 1.00× | |
| 2 | | | |
| 4 | | | |
| 8 | | | |

`render_seconds` is the render Job's full wall time: scheduling, model
load, and synthesis. On one machine, speedup is capped by its physical
cores and memory bandwidth, and by the slowest chapter once there are as
many pods as chapters. Adding nodes to a real cluster raises the core cap;
it does not change the code.
