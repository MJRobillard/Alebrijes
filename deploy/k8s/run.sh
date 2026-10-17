#!/usr/bin/env bash
# Renders one book end to end on a local kind cluster:
#   cast Job -> Indexed render Job (one pod per chapter) -> assemble Job
#
# Usage: BOOK=bench.txt PARALLELISM=8 deploy/k8s/run.sh [--skip-build]
# The book must already be in deploy/k8s/data/ (see make_bench_book.py).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
export PARALLELISM="${PARALLELISM:-8}"

[[ -f "$DATA_DIR/$BOOK" ]] || { echo "No $DATA_DIR/$BOOK -- see make_bench_book.py"; exit 1; }

ensure_cluster
[[ "${1:-}" == "--skip-build" ]] || build_image
load_image

export CHAPTERS="$(chapter_count)"
log "'$BOOK': $CHAPTERS chapters, up to $PARALLELISM render pods at once"

run_job cast-job.yaml alebrijes-cast
run_job render-job.yaml alebrijes-render
render_seconds=$JOB_SECONDS
kubectl -n "$NS" get pods -l app=alebrijes-render -o wide
run_job assemble-job.yaml alebrijes-assemble

audio=$(audio_seconds)
log "done: $DATA_DIR/out/$STEM.m4b -- ${audio}s of audio, chapters rendered in ${render_seconds}s" \
  "($(python3 -c "print(f'{$audio / $render_seconds:.2f}')")x realtime)"
